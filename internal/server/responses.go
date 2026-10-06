// OpenAI Responses API 兼容层（POST /v1/responses），供 Codex 等只会说 Responses 协议的客户端直连。
//
// 为什么做成转接层而不是新链路：选号、粘性、6004/冷却策略、预算闸、统计与请求日志
// 全在 chatCompletions 里。这里只做两件事——把 Responses 请求翻成等价的
// chat/completions 请求体交给 chatCompletions，再把它写出的结果（SSE chunk 或聚合后的
// chat.completion）翻回 Responses 的事件流 / 响应对象。上游始终只见到 chat/completions，
// 账号侧行为与直接调用完全一致，两条入口不会在调度策略上分叉。
//
// 不支持（显式拒绝或忽略）：
//   - previous_response_id：网关不存响应，客户端须每轮带全量 input（Codex 默认
//     store=false 即如此）。带了直接 400——静默忽略等于悄悄丢掉上文。
//   - 内置工具（web_search / file_search / computer_use / image_generation / local_shell …）：
//     上游没有对应能力，转发只会被拒，故丢弃并记日志。function 与 custom 工具照常转换。
//   - reasoning 的 encrypted_content：上游无法消费，只取可读文本回填 reasoning_content。
package server

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"

	"workbuddy2api/internal/logfmt"
)

// responsesRequest Responses 请求里本层会用到的字段（其余字段如 store / include /
// service_tier / text.verbosity 对上游无意义，直接忽略）。
type responsesRequest struct {
	Model             string            `json:"model"`
	Input             json.RawMessage   `json:"input"`
	Instructions      string            `json:"instructions"`
	Stream            bool              `json:"stream"`
	Tools             []json.RawMessage `json:"tools"`
	ToolChoice        json.RawMessage   `json:"tool_choice"`
	ParallelToolCalls *bool             `json:"parallel_tool_calls"`
	MaxOutputTokens   *int64            `json:"max_output_tokens"`
	Temperature       *float64          `json:"temperature"`
	TopP              *float64          `json:"top_p"`
	Reasoning         *struct {
		Effort  string `json:"effort"`
		Summary string `json:"summary"`
	} `json:"reasoning"`
	Text *struct {
		Format json.RawMessage `json:"format"`
	} `json:"text"`
	// Messages 非标准字段：部分客户端（如 Nagram）把 chat 形态的 messages 发到
	// /v1/responses 而不带 input。没有 input 时按 chat messages 原样采用，见 responsesToChat。
	Messages           []map[string]any `json:"messages"`
	PromptCacheKey     string           `json:"prompt_cache_key"`
	User               string           `json:"user"`
	PreviousResponseID string           `json:"previous_response_id"`
	Metadata           map[string]any   `json:"metadata"`
}

// responsesMeta 一次请求的转换上下文：响应对象的回显字段 + custom 工具名表
// （上游只认 function 工具，custom 工具被包成 function，回程靠这张表还原成
// custom_tool_call）。
type responsesMeta struct {
	id          string
	created     int64
	req         *responsesRequest
	customTools map[string]bool
}

// responses 处理 POST /v1/responses。
func (h *Handler) responses(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request", "read body: "+err.Error())
		return
	}
	var req responsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request", "invalid JSON: "+err.Error())
		return
	}
	if req.PreviousResponseID != "" {
		writeOpenAIError(w, http.StatusBadRequest, "unsupported_parameter",
			"previous_response_id is not supported: this gateway does not store responses; send the full conversation in input")
		return
	}
	chat, meta, err := responsesToChat(&req)
	if err != nil {
		// 只记结构不记内容：客户端发来的形态是排查这类 400 的唯一线索（下游中转通常不存请求体）。
		log.Printf("WARN: [responses] 400 %v; ua=%q body keys=%s input=%s", err,
			logfmt.Truncate(r.UserAgent(), 60), bodyKeys(raw), jsonKind(req.Input))
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	chatBody, err := json.Marshal(chat)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, "internal_error", err.Error())
		return
	}

	// 复用 chat 链路：克隆请求（保留 ctx 与会话头族，供粘性/聚合键沿用），换上翻译后的 body。
	r2 := r.Clone(r.Context())
	r2.Body = io.NopCloser(bytes.NewReader(chatBody))
	r2.ContentLength = int64(len(chatBody))
	r2.URL.Path = "/v1/chat/completions"

	sw := newResponsesWriter(w, meta)
	h.chatCompletions(sw, r2)
	sw.finish()
}

// newRespID 生成带前缀的随机 id（resp_ / msg_ / rs_ / fc_ / ctc_ / call_）。
func newRespID(prefix string) string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return prefix + hex.EncodeToString(b[:])
}

// ---------------------------------------------------------------------------
// 请求方向：Responses → chat/completions
// ---------------------------------------------------------------------------

// responsesToChat 把 Responses 请求翻成 chat/completions 请求体。
func responsesToChat(req *responsesRequest) (map[string]any, *responsesMeta, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, nil, fmt.Errorf("model is required")
	}
	meta := &responsesMeta{
		id:          newRespID("resp_"),
		created:     time.Now().Unix(),
		req:         req,
		customTools: map[string]bool{},
	}
	tools, dropped := convertResponsesTools(req.Tools, meta.customTools)
	if len(dropped) > 0 {
		log.Printf("[responses] dropped unsupported built-in tools: %s", strings.Join(dropped, ", "))
	}

	b := &chatMsgBuilder{}
	if s := strings.TrimSpace(req.Instructions); s != "" {
		b.msgs = append(b.msgs, map[string]any{"role": "system", "content": req.Instructions})
	}
	if err := b.addInput(req.Input); err != nil {
		return nil, nil, err
	}
	if len(bytes.TrimSpace(req.Input)) == 0 && len(req.Messages) > 0 {
		// 没有 input、却带了 chat 形态的 messages：本来就是上游要的格式，原样采用，
		// 而不是回 400——那会让下游中转（claude-code-hub 等）把整个供应商熔断。
		for _, m := range req.Messages {
			b.msgs = append(b.msgs, m)
		}
	}
	if len(b.msgs) == 0 {
		return nil, nil, fmt.Errorf("input is required")
	}

	chat := map[string]any{
		"model":    req.Model,
		"messages": b.messages(),
		"stream":   req.Stream,
	}
	if len(tools) > 0 {
		chat["tools"] = tools
		if tc := convertResponsesToolChoice(req.ToolChoice); tc != nil {
			chat["tool_choice"] = tc
		}
		if req.ParallelToolCalls != nil {
			chat["parallel_tool_calls"] = *req.ParallelToolCalls
		}
	}
	if req.MaxOutputTokens != nil && *req.MaxOutputTokens > 0 {
		chat["max_tokens"] = *req.MaxOutputTokens
	}
	if req.Temperature != nil {
		chat["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		chat["top_p"] = *req.TopP
	}
	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		effort := strings.ToLower(strings.TrimSpace(req.Reasoning.Effort))
		if effort == "none" {
			effort = "off" // Responses 的 none 对应本网关档位表里的 off（payload.effortRank）
		}
		chat["reasoning_effort"] = effort
	}
	if req.Text != nil {
		if rf := convertResponsesTextFormat(req.Text.Format); rf != nil {
			chat["response_format"] = rf
		}
	}
	if req.PromptCacheKey != "" {
		chat["prompt_cache_key"] = req.PromptCacheKey // 也是粘性键来源（同会话落同一账号）
	}
	if req.User != "" {
		chat["user"] = req.User
	}
	return chat, meta, nil
}

// chatMsgBuilder 把 Responses 的 input 条目序列折叠成 chat messages。
//
// 折叠规则：Responses 把「一轮助手输出」拆成多个平级条目（message / function_call /
// reasoning），chat 则是一条 assistant 消息带 content + tool_calls + reasoning_content。
// 故 function_call 若紧跟在 assistant 消息后（中间没有 user/tool 消息），并入那条消息；
// reasoning 条目暂存，回填给下一条 assistant 消息的 reasoning_content（DeepSeek 等思考
// 模型在工具调用轮要求回传思考内容）。
type chatMsgBuilder struct {
	msgs             []map[string]any
	pendingReasoning string
}

func (b *chatMsgBuilder) messages() []any {
	out := make([]any, len(b.msgs))
	for i, m := range b.msgs {
		out[i] = m
	}
	return out
}

// assistantTail 返回可并入工具调用的末条 assistant 消息；没有则新建一条。
func (b *chatMsgBuilder) assistantTail() map[string]any {
	if n := len(b.msgs); n > 0 {
		if last := b.msgs[n-1]; last["role"] == "assistant" {
			return last
		}
	}
	m := map[string]any{"role": "assistant", "content": nil}
	b.attachReasoning(m)
	b.msgs = append(b.msgs, m)
	return m
}

func (b *chatMsgBuilder) attachReasoning(m map[string]any) {
	if b.pendingReasoning == "" {
		return
	}
	// 同一轮里先后有两段思考（思考 → 工具 → 思考 → 工具）时串接，不覆盖也不丢。
	if s, _ := m["reasoning_content"].(string); s != "" {
		m["reasoning_content"] = s + "\n\n" + b.pendingReasoning
	} else {
		m["reasoning_content"] = b.pendingReasoning
	}
	b.pendingReasoning = ""
}

func (b *chatMsgBuilder) addInput(raw json.RawMessage) error {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return fmt.Errorf("input: %w", err)
		}
		b.msgs = append(b.msgs, map[string]any{"role": "user", "content": s})
		return nil
	}
	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		return fmt.Errorf("input must be a string or an array of items: %w", err)
	}
	for i, it := range items {
		if err := b.addItem(it); err != nil {
			return fmt.Errorf("input[%d]: %w", i, err)
		}
	}
	return nil
}

func (b *chatMsgBuilder) addItem(it map[string]any) error {
	typ, _ := it["type"].(string)
	switch typ {
	case "", "message":
		role, _ := it["role"].(string)
		if role == "" {
			return fmt.Errorf("message item missing role")
		}
		if role == "assistant" {
			m := map[string]any{"role": "assistant", "content": textOfParts(it["content"])}
			b.attachReasoning(m)
			b.msgs = append(b.msgs, m)
			return nil
		}
		content, err := convertInputContent(it["content"])
		if err != nil {
			return err
		}
		// 思考只属于紧随其后的助手输出；中间插进非助手消息（如被打断的回合），丢弃，
		// 免得挂到后面不相干的 assistant 上。
		b.pendingReasoning = ""
		b.msgs = append(b.msgs, map[string]any{"role": role, "content": content})
	case "function_call", "custom_tool_call":
		callID, _ := it["call_id"].(string)
		name, _ := it["name"].(string)
		args, _ := it["arguments"].(string)
		if typ == "custom_tool_call" {
			input, _ := it["input"].(string)
			enc, _ := json.Marshal(map[string]string{"input": input})
			args = string(enc)
		}
		if args == "" {
			args = "{}"
		}
		m := b.assistantTail()
		b.attachReasoning(m)
		calls, _ := m["tool_calls"].([]any)
		m["tool_calls"] = append(calls, map[string]any{
			"id":       callID,
			"type":     "function",
			"function": map[string]any{"name": name, "arguments": args},
		})
	case "function_call_output", "custom_tool_call_output":
		b.pendingReasoning = ""
		callID, _ := it["call_id"].(string)
		b.msgs = append(b.msgs, map[string]any{
			"role":         "tool",
			"tool_call_id": callID,
			"content":      textOfParts(it["output"]),
		})
	case "reasoning":
		if s := reasoningText(it); s != "" {
			b.pendingReasoning = s
		}
	default:
		// item_reference / local_shell_call / web_search_call 等：上游无对应语义，跳过。
		log.Printf("[responses] skipped unsupported input item type=%q", typ)
	}
	return nil
}

// convertInputContent 把 user/system/developer 消息的 content 转成 chat 形态：
// 纯文本收敛成字符串；含图片时保留分段数组。
func convertInputContent(c any) (any, error) {
	switch v := c.(type) {
	case string:
		return v, nil
	case nil:
		return "", nil
	case []any:
		var parts []any
		hasImage := false
		for _, p := range v {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			switch pt, _ := pm["type"].(string); pt {
			case "input_text", "output_text", "text":
				t, _ := pm["text"].(string)
				parts = append(parts, map[string]any{"type": "text", "text": t})
			case "input_image":
				url, _ := pm["image_url"].(string)
				if url == "" {
					log.Printf("[responses] skipped input_image without image_url (file_id is not supported)")
					continue
				}
				img := map[string]any{"url": url}
				if d, _ := pm["detail"].(string); d != "" && d != "auto" {
					img["detail"] = d
				}
				parts = append(parts, map[string]any{"type": "image_url", "image_url": img})
				hasImage = true
			default:
				log.Printf("[responses] skipped unsupported content part type=%q", pt)
			}
		}
		if !hasImage {
			var sb strings.Builder
			for i, p := range parts {
				if i > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(p.(map[string]any)["text"].(string))
			}
			return sb.String(), nil
		}
		return parts, nil
	default:
		return nil, fmt.Errorf("unsupported content shape %T", c)
	}
}

// textOfParts 取 content / output 的纯文本：字符串原样；分段数组拼接其中的文本段。
func textOfParts(c any) string {
	switch v := c.(type) {
	case string:
		return v
	case []any:
		var sb strings.Builder
		for _, p := range v {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			t, _ := pm["text"].(string)
			if t == "" {
				t, _ = pm["refusal"].(string)
			}
			if t == "" {
				continue
			}
			if sb.Len() > 0 {
				sb.WriteString("\n")
			}
			sb.WriteString(t)
		}
		return sb.String()
	}
	return ""
}

// reasoningText 取 reasoning 条目的可读文本：优先原始思考（content[].reasoning_text），
// 否则拼摘要（summary[].summary_text）。encrypted_content 无法回传，忽略。
func reasoningText(it map[string]any) string {
	collect := func(key string) string {
		arr, _ := it[key].([]any)
		var sb strings.Builder
		for _, p := range arr {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if t, _ := pm["text"].(string); t != "" {
				if sb.Len() > 0 {
					sb.WriteString("\n\n")
				}
				sb.WriteString(t)
			}
		}
		return sb.String()
	}
	if s := collect("content"); s != "" {
		return s
	}
	return collect("summary")
}

// convertResponsesTools 把 Responses 工具表翻成 chat tools。
//   - function：扁平字段收进 function 子对象（strict 不转发，上游未必认）。
//   - custom（Codex 的 apply_patch 等自由文本工具）：包成只有一个 string 参数 input 的
//     function，语法定义写进描述；名字记入 custom 表，回程还原成 custom_tool_call。
//   - 其余内置工具：上游无对应能力，返回名单由调用方记日志。
func convertResponsesTools(raw []json.RawMessage, custom map[string]bool) (tools []any, dropped []string) {
	for _, r := range raw {
		var t map[string]any
		if json.Unmarshal(r, &t) != nil {
			continue
		}
		typ, _ := t["type"].(string)
		name, _ := t["name"].(string)
		desc, _ := t["description"].(string)
		switch typ {
		case "function":
			params := t["parameters"]
			if params == nil {
				params = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			fn := map[string]any{"name": name, "parameters": params}
			if desc != "" {
				fn["description"] = desc
			}
			tools = append(tools, map[string]any{"type": "function", "function": fn})
		case "custom":
			custom[name] = true
			if f, ok := t["format"].(map[string]any); ok && f["type"] == "grammar" {
				syntax, _ := f["syntax"].(string)
				def, _ := f["definition"].(string)
				desc += fmt.Sprintf("\n\nThe `input` argument is raw free-form text (not JSON) and must follow this %s grammar:\n%s", syntax, def)
			}
			tools = append(tools, map[string]any{"type": "function", "function": map[string]any{
				"name":        name,
				"description": strings.TrimSpace(desc),
				"parameters": map[string]any{
					"type":                 "object",
					"properties":           map[string]any{"input": map[string]any{"type": "string", "description": "Raw free-form input for this tool."}},
					"required":             []any{"input"},
					"additionalProperties": false,
				},
			}})
		default:
			dropped = append(dropped, typ)
		}
	}
	return tools, dropped
}

// convertResponsesToolChoice 翻译 tool_choice；无法表达的形态回落 auto，未提供返回 nil。
func convertResponsesToolChoice(raw json.RawMessage) any {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var o map[string]any
	if json.Unmarshal(raw, &o) != nil {
		return nil
	}
	switch o["type"] {
	case "function", "custom":
		if name, _ := o["name"].(string); name != "" {
			return map[string]any{"type": "function", "function": map[string]any{"name": name}}
		}
	}
	return "auto"
}

// convertResponsesTextFormat 把 text.format 翻成 chat 的 response_format；纯文本返回 nil。
func convertResponsesTextFormat(raw json.RawMessage) any {
	var f map[string]any
	if len(raw) == 0 || json.Unmarshal(raw, &f) != nil {
		return nil
	}
	switch f["type"] {
	case "json_object":
		return map[string]any{"type": "json_object"}
	case "json_schema":
		js := map[string]any{"name": f["name"], "schema": f["schema"]}
		if v, ok := f["strict"]; ok {
			js["strict"] = v
		}
		if v, ok := f["description"]; ok {
			js["description"] = v
		}
		return map[string]any{"type": "json_schema", "json_schema": js}
	}
	return nil
}

// ---------------------------------------------------------------------------
// 响应方向：chat/completions → Responses（非流式与流式共用的组装件）
// ---------------------------------------------------------------------------

// baseResponse 组装 Response 对象骨架（回显请求参数），output/usage 由调用方填。
func (m *responsesMeta) baseResponse(status string) map[string]any {
	req := m.req
	resp := map[string]any{
		"id":                   m.id,
		"object":               "response",
		"created_at":           m.created,
		"status":               status,
		"error":                nil,
		"incomplete_details":   nil,
		"instructions":         nil,
		"max_output_tokens":    nil,
		"model":                req.Model,
		"output":               []any{},
		"parallel_tool_calls":  true,
		"previous_response_id": nil,
		"reasoning":            map[string]any{"effort": nil, "summary": nil},
		"store":                false,
		"temperature":          nil,
		"text":                 map[string]any{"format": map[string]any{"type": "text"}},
		"tool_choice":          "auto",
		"tools":                []any{},
		"top_p":                nil,
		"truncation":           "disabled",
		"usage":                nil,
		"user":                 nil,
		"metadata":             map[string]any{},
	}
	if req.Instructions != "" {
		resp["instructions"] = req.Instructions
	}
	if req.MaxOutputTokens != nil {
		resp["max_output_tokens"] = *req.MaxOutputTokens
	}
	if req.ParallelToolCalls != nil {
		resp["parallel_tool_calls"] = *req.ParallelToolCalls
	}
	if req.Reasoning != nil {
		resp["reasoning"] = map[string]any{"effort": nilIfEmpty(req.Reasoning.Effort), "summary": nilIfEmpty(req.Reasoning.Summary)}
	}
	if req.Temperature != nil {
		resp["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		resp["top_p"] = *req.TopP
	}
	if req.Text != nil && len(req.Text.Format) > 0 {
		resp["text"] = map[string]any{"format": req.Text.Format}
	}
	if len(bytes.TrimSpace(req.ToolChoice)) > 0 {
		resp["tool_choice"] = req.ToolChoice
	}
	if len(req.Tools) > 0 {
		resp["tools"] = req.Tools
	}
	if req.User != "" {
		resp["user"] = req.User
	}
	if req.Metadata != nil {
		resp["metadata"] = req.Metadata
	}
	return resp
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// finishStatus 把 chat 的 finish_reason 映射成 Response 的 status 与 incomplete_details。
func finishStatus(finish string) (string, any) {
	switch finish {
	case "length":
		return "incomplete", map[string]any{"reason": "max_output_tokens"}
	case "content_filter":
		return "incomplete", map[string]any{"reason": "content_filter"}
	}
	return "completed", nil
}

// responsesUsage 把 chat usage 换算成 Responses usage（缓存/思考 token 兼容多种上游字段名）。
func responsesUsage(u map[string]any) map[string]any {
	if u == nil {
		return nil
	}
	num := func(v any) int64 {
		switch n := v.(type) {
		case float64:
			return int64(n)
		case int64:
			return n
		case int:
			return int64(n)
		case json.Number:
			i, _ := n.Int64()
			return i
		}
		return 0
	}
	sub := func(key, field string) int64 {
		if m, ok := u[key].(map[string]any); ok {
			return num(m[field])
		}
		return 0
	}
	in, out := num(u["prompt_tokens"]), num(u["completion_tokens"])
	total := num(u["total_tokens"])
	if total == 0 {
		total = in + out
	}
	cached := sub("prompt_tokens_details", "cached_tokens")
	if cached == 0 {
		cached = num(u["cached_tokens"])
	}
	if cached == 0 {
		cached = num(u["prompt_cache_hit_tokens"])
	}
	reasoning := sub("completion_tokens_details", "reasoning_tokens")
	if reasoning == 0 {
		reasoning = num(u["completion_thinking_tokens"])
	}
	return map[string]any{
		"input_tokens":          in,
		"input_tokens_details":  map[string]any{"cached_tokens": cached},
		"output_tokens":         out,
		"output_tokens_details": map[string]any{"reasoning_tokens": reasoning},
		"total_tokens":          total,
	}
}

// customToolInput 从包装后的 function 参数（{"input": "..."}）取回 custom 工具的原始输入；
// 模型没按约定输出 JSON 时退回原文。
func customToolInput(args string) string {
	var v struct {
		Input *string `json:"input"`
	}
	if json.Unmarshal([]byte(args), &v) == nil && v.Input != nil {
		return *v.Input
	}
	return args
}

// 输出条目构造：流式 output_item.done 与非流式 output 数组用同一套形状。

func reasoningItem(id, text string) map[string]any {
	return map[string]any{
		"type":    "reasoning",
		"id":      id,
		"summary": []any{map[string]any{"type": "summary_text", "text": text}},
	}
}

func outputTextPart(text string) map[string]any {
	return map[string]any{"type": "output_text", "text": text, "annotations": []any{}}
}

func messageItem(id, status, text string) map[string]any {
	content := []any{}
	if status == "completed" {
		content = []any{outputTextPart(text)}
	}
	return map[string]any{"type": "message", "id": id, "status": status, "role": "assistant", "content": content}
}

func toolCallItem(m *responsesMeta, id, callID, name, args, status string) map[string]any {
	if m.customTools[name] {
		input := ""
		if status == "completed" {
			input = customToolInput(args)
		}
		return map[string]any{"type": "custom_tool_call", "id": id, "call_id": callID, "name": name, "input": input, "status": status}
	}
	return map[string]any{"type": "function_call", "id": id, "call_id": callID, "name": name, "arguments": args, "status": status}
}

func toolItemPrefix(m *responsesMeta, name string) string {
	if m.customTools[name] {
		return "ctc_"
	}
	return "fc_"
}

// chatCompletionToResponse 把聚合后的 chat.completion 翻成 Response 对象（非流式）。
func chatCompletionToResponse(m *responsesMeta, cc map[string]any) map[string]any {
	var (
		content, reasoning, finish string
		toolCalls                  []any
	)
	if choices, ok := cc["choices"].([]any); ok && len(choices) > 0 {
		if c0, ok := choices[0].(map[string]any); ok {
			finish, _ = c0["finish_reason"].(string)
			if msg, ok := c0["message"].(map[string]any); ok {
				content, _ = msg["content"].(string)
				reasoning, _ = msg["reasoning_content"].(string)
				toolCalls, _ = msg["tool_calls"].([]any)
			}
		}
	}
	var output []any
	if reasoning != "" {
		output = append(output, reasoningItem(newRespID("rs_"), reasoning))
	}
	if content != "" {
		output = append(output, messageItem(newRespID("msg_"), "completed", content))
	}
	for _, tc := range toolCalls {
		tm, ok := tc.(map[string]any)
		if !ok {
			continue
		}
		callID, _ := tm["id"].(string)
		if callID == "" {
			callID = newRespID("call_")
		}
		fn, _ := tm["function"].(map[string]any)
		name, _ := fn["name"].(string)
		args, _ := fn["arguments"].(string)
		if args == "" && !m.customTools[name] {
			args = "{}"
		}
		output = append(output, toolCallItem(m, newRespID(toolItemPrefix(m, name)), callID, name, args, "completed"))
	}
	status, incomplete := finishStatus(finish)
	resp := m.baseResponse(status)
	resp["incomplete_details"] = incomplete
	if output == nil {
		output = []any{}
	}
	resp["output"] = output
	if u, ok := cc["usage"].(map[string]any); ok {
		resp["usage"] = responsesUsage(u)
	}
	return resp
}

// sortedKeys 让工具调用按 chat 的 index 顺序收尾（map 迭代无序，事件顺序必须确定）。
func sortedKeys(m map[int]*respItem) []int {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	return keys
}

// responsesNotSupported 兜住 /v1/responses 的其余形态：非 POST 方法与子路径
// （/{id}、/{id}/cancel、/compact、/input_tokens …）。网关不存响应，这些都无从实现；
// 回 JSON 错误信封而不是 mux 默认的纯文本，客户端才能解析出原因。
func responsesNotSupported(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/v1/responses" {
		w.Header().Set("Allow", http.MethodPost)
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "use POST /v1/responses")
		return
	}
	writeOpenAIError(w, http.StatusNotFound, "not_found",
		"not supported: this gateway does not store responses, so retrieve/cancel/compact/input_tokens endpoints are unavailable")
}

// bodyKeys 请求体的顶层键（排序后逗号拼接），供 400 诊断日志用。
func bodyKeys(raw []byte) string {
	var m map[string]json.RawMessage
	if json.Unmarshal(raw, &m) != nil {
		return "<not an object>"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return strings.Join(keys, ",")
}

// jsonKind 描述一段 JSON 的形态（缺失 / null / 字符串 / 数组长度 / 对象），不含内容。
func jsonKind(raw json.RawMessage) string {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return "missing"
	}
	switch raw[0] {
	case 'n':
		return "null"
	case '"':
		return "string"
	case '{':
		return "object"
	case '[':
		var arr []json.RawMessage
		_ = json.Unmarshal(raw, &arr)
		return fmt.Sprintf("array(len=%d)", len(arr))
	}
	return "other"
}
