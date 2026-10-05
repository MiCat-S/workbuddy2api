package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// responsesWriter 夹在 chatCompletions 与真实 ResponseWriter 之间，按 chat 链路的输出形态分三路：
//   - 非 200（错误信封）：原样透传——Responses 客户端认的错误也是 {"error":{...}} 信封；
//   - 200 + text/event-stream：逐帧把 chat.completion.chunk 翻成 Responses 事件流；
//   - 200 + JSON（非流式聚合结果）：缓存，finish 时整体翻成 Response 对象。
//
// Header() 直接共用底层的头表：chat 链路设置的 Content-Type（SSE / JSON）、
// X-Wb-Account 等对 Responses 客户端同样成立，无需二次转换。
type responsesWriter struct {
	w    http.ResponseWriter
	fl   http.Flusher
	meta *responsesMeta
	mode rwMode

	line bytes.Buffer // SSE 未成行的残片
	body bytes.Buffer // 非流式 JSON

	seq      int
	started  bool
	finished bool
	writeErr error // 客户端断开等写失败：记住后让 chat 链路尽早停读上游

	nextIndex int
	open      *respItem         // 当前打开的文本类条目（reasoning / message），同一时刻至多一个
	tools     map[int]*respItem // chat tool_calls 的 index → 条目（并行调用可交错到达，收尾统一关闭）
	outputs   map[int]any       // output_index → 已完成条目，拼 response.completed 的 output

	finishReason string
	usage        map[string]any
	failure      map[string]any // 流中 error 帧
}

type rwMode int

const (
	rwPending rwMode = iota
	rwPassthrough
	rwSSE
	rwJSON
)

// respItem 一个输出条目的累积状态。kind：reasoning / message / tool。
type respItem struct {
	kind   string
	id     string
	index  int
	text   strings.Builder // 思考文本 / 正文 / 工具参数
	callID string
	name   string
}

func newResponsesWriter(w http.ResponseWriter, meta *responsesMeta) *responsesWriter {
	fl, _ := w.(http.Flusher)
	return &responsesWriter{
		w:       w,
		fl:      fl,
		meta:    meta,
		tools:   map[int]*respItem{},
		outputs: map[int]any{},
	}
}

func (s *responsesWriter) Header() http.Header { return s.w.Header() }

func (s *responsesWriter) WriteHeader(code int) {
	if s.mode == rwPending {
		s.decide(code)
	}
}

func (s *responsesWriter) decide(code int) {
	switch {
	case code != http.StatusOK:
		s.mode = rwPassthrough
		s.w.WriteHeader(code)
	case strings.Contains(s.w.Header().Get("Content-Type"), "text/event-stream"):
		s.mode = rwSSE
	default:
		s.mode = rwJSON
	}
}

func (s *responsesWriter) Write(p []byte) (int, error) {
	if s.mode == rwPending {
		s.decide(http.StatusOK)
	}
	switch s.mode {
	case rwPassthrough:
		return s.w.Write(p)
	case rwJSON:
		return s.body.Write(p)
	}
	if s.writeErr != nil {
		return 0, s.writeErr
	}
	s.line.Write(p)
	for {
		i := bytes.IndexByte(s.line.Bytes(), '\n')
		if i < 0 {
			break
		}
		ln := strings.TrimRight(string(s.line.Next(i+1)), "\r\n")
		s.handleLine(ln)
		if s.writeErr != nil {
			return 0, s.writeErr
		}
	}
	return len(p), nil
}

// Flush 只在透传模式下转发；SSE 模式每个事件写出后已即时 flush。
func (s *responsesWriter) Flush() {
	if s.mode == rwPassthrough && s.fl != nil {
		s.fl.Flush()
	}
}

func (s *responsesWriter) handleLine(ln string) {
	switch {
	case ln == "":
	case strings.HasPrefix(ln, "data: [DONE]"):
		s.complete()
	case strings.HasPrefix(ln, "data: "):
		s.handleChunk(strings.TrimPrefix(ln, "data: "))
	case strings.HasPrefix(ln, ":"):
		s.raw(ln + "\n\n") // SSE 注释（保活）原样转发
	}
}

func (s *responsesWriter) handleChunk(payload string) {
	var obj map[string]any
	if json.Unmarshal([]byte(payload), &obj) != nil {
		return
	}
	s.start()
	if e, ok := obj["error"]; ok && e != nil {
		s.failure, _ = e.(map[string]any)
		if s.failure == nil {
			s.failure = map[string]any{"message": payload}
		}
		return
	}
	if u, ok := obj["usage"].(map[string]any); ok {
		s.usage = u
	}
	choices, _ := obj["choices"].([]any)
	if len(choices) == 0 {
		return
	}
	c0, _ := choices[0].(map[string]any)
	if fr, _ := c0["finish_reason"].(string); fr != "" {
		s.finishReason = fr
	}
	delta, _ := c0["delta"].(map[string]any)
	if delta == nil {
		return
	}
	if d, _ := delta["reasoning_content"].(string); d != "" {
		s.reasoningDelta(d)
	}
	if d, _ := delta["content"].(string); d != "" {
		s.textDelta(d)
	}
	if tcs, _ := delta["tool_calls"].([]any); len(tcs) > 0 {
		for _, tc := range tcs {
			if tm, ok := tc.(map[string]any); ok {
				s.toolDelta(tm)
			}
		}
	}
}

// start 首个有效帧到达时发 response.created / response.in_progress。
func (s *responsesWriter) start() {
	if s.started {
		return
	}
	s.started = true
	resp := s.meta.baseResponse("in_progress")
	s.emit("response.created", map[string]any{"response": resp})
	s.emit("response.in_progress", map[string]any{"response": resp})
}

func (s *responsesWriter) newItem(kind, id string) *respItem {
	it := &respItem{kind: kind, id: id, index: s.nextIndex}
	s.nextIndex++
	return it
}

func (s *responsesWriter) reasoningDelta(d string) {
	if s.open == nil || s.open.kind != "reasoning" {
		s.closeOpen()
		it := s.newItem("reasoning", newRespID("rs_"))
		s.open = it
		s.emit("response.output_item.added", map[string]any{
			"output_index": it.index,
			"item":         map[string]any{"type": "reasoning", "id": it.id, "summary": []any{}},
		})
		s.emit("response.reasoning_summary_part.added", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": ""},
		})
	}
	s.open.text.WriteString(d)
	s.emit("response.reasoning_summary_text.delta", map[string]any{
		"item_id": s.open.id, "output_index": s.open.index, "summary_index": 0, "delta": d,
	})
}

func (s *responsesWriter) textDelta(d string) {
	if s.open == nil || s.open.kind != "message" {
		s.closeOpen()
		it := s.newItem("message", newRespID("msg_"))
		s.open = it
		s.emit("response.output_item.added", map[string]any{
			"output_index": it.index, "item": messageItem(it.id, "in_progress", ""),
		})
		s.emit("response.content_part.added", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0, "part": outputTextPart(""),
		})
	}
	s.open.text.WriteString(d)
	s.emit("response.output_text.delta", map[string]any{
		"item_id": s.open.id, "output_index": s.open.index, "content_index": 0, "delta": d, "logprobs": []any{},
	})
}

func (s *responsesWriter) toolDelta(tm map[string]any) {
	idx := 0
	if f, ok := tm["index"].(float64); ok {
		idx = int(f)
	}
	fn, _ := tm["function"].(map[string]any)
	name, _ := fn["name"].(string)
	args, _ := fn["arguments"].(string)
	it := s.tools[idx]
	if it == nil {
		s.closeOpen() // 模型转入工具调用：先收尾正在输出的思考/正文
		callID, _ := tm["id"].(string)
		if callID == "" {
			callID = newRespID("call_")
		}
		it = s.newItem("tool", newRespID(toolItemPrefix(s.meta, name)))
		it.callID, it.name = callID, name
		s.tools[idx] = it
		s.emit("response.output_item.added", map[string]any{
			"output_index": it.index,
			"item":         toolCallItem(s.meta, it.id, it.callID, it.name, "", "in_progress"),
		})
	}
	if args == "" {
		return
	}
	it.text.WriteString(args)
	// custom 工具的参数是 {"input": "..."} 包装，流式片段无法逐字还原成原始输入，
	// 只在 output_item.done 一次性给出完整 input。
	if !s.meta.customTools[it.name] {
		s.emit("response.function_call_arguments.delta", map[string]any{
			"item_id": it.id, "output_index": it.index, "delta": args,
		})
	}
}

func (s *responsesWriter) closeOpen() {
	if s.open != nil {
		s.closeItem(s.open)
		s.open = nil
	}
}

func (s *responsesWriter) closeItem(it *respItem) {
	text := it.text.String()
	var item map[string]any
	switch it.kind {
	case "reasoning":
		s.emit("response.reasoning_summary_text.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0, "text": text,
		})
		s.emit("response.reasoning_summary_part.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": text},
		})
		item = reasoningItem(it.id, text)
	case "message":
		s.emit("response.output_text.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0, "text": text, "logprobs": []any{},
		})
		s.emit("response.content_part.done", map[string]any{
			"item_id": it.id, "output_index": it.index, "content_index": 0, "part": outputTextPart(text),
		})
		item = messageItem(it.id, "completed", text)
	default:
		args := text
		if !s.meta.customTools[it.name] {
			if args == "" {
				args = "{}"
			}
			s.emit("response.function_call_arguments.done", map[string]any{
				"item_id": it.id, "output_index": it.index, "arguments": args,
			})
		}
		item = toolCallItem(s.meta, it.id, it.callID, it.name, args, "completed")
	}
	s.outputs[it.index] = item
	s.emit("response.output_item.done", map[string]any{"output_index": it.index, "item": item})
}

// complete 收尾：关闭所有打开的条目，发终态事件（completed / incomplete / failed）。
func (s *responsesWriter) complete() {
	if s.finished {
		return
	}
	s.finished = true
	s.start()
	s.closeOpen()
	for _, k := range sortedKeys(s.tools) {
		s.closeItem(s.tools[k])
	}
	output := make([]any, 0, s.nextIndex)
	for i := 0; i < s.nextIndex; i++ {
		if it, ok := s.outputs[i]; ok {
			output = append(output, it)
		}
	}
	resp := s.meta.baseResponse("completed")
	resp["output"] = output
	if s.usage != nil {
		resp["usage"] = responsesUsage(s.usage)
	}
	if s.failure != nil {
		resp["status"] = "failed"
		resp["error"] = responsesError(s.failure)
		s.emit("response.failed", map[string]any{"response": resp})
		return
	}
	status, incomplete := finishStatus(s.finishReason)
	resp["status"] = status
	resp["incomplete_details"] = incomplete
	if status == "incomplete" {
		s.emit("response.incomplete", map[string]any{"response": resp})
		return
	}
	s.emit("response.completed", map[string]any{"response": resp})
}

// finish 在 chatCompletions 返回后调用：非流式翻译并写出；流式补齐未收尾的事件流。
func (s *responsesWriter) finish() {
	switch s.mode {
	case rwPending:
		// chat 链路什么都没写（正常路径不会发生）：按 502 兜底，免得客户端空等。
		writeOpenAIError(s.w, http.StatusBadGateway, "upstream_parse", "empty response from chat pipeline")
	case rwJSON:
		var cc map[string]any
		if err := json.Unmarshal(s.body.Bytes(), &cc); err != nil {
			writeOpenAIError(s.w, http.StatusBadGateway, "upstream_parse", "decode chat completion: "+err.Error())
			return
		}
		writeJSON(s.w, http.StatusOK, chatCompletionToResponse(s.meta, cc))
	case rwSSE:
		if s.finished {
			return
		}
		// 没等到 [DONE] 流就结束了（上游读错误 / 客户端断开）：已拿到 finish_reason
		// 按正常收尾，否则判失败——Responses 客户端要求事件流必须以终态事件结束。
		if s.finishReason == "" && s.failure == nil {
			s.failure = map[string]any{"code": "stream_interrupted", "message": "upstream stream ended before completion"}
		}
		s.complete()
	}
}

func (s *responsesWriter) emit(typ string, ev map[string]any) {
	ev["type"] = typ
	ev["sequence_number"] = s.seq
	s.seq++
	raw, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.raw("event: " + typ + "\ndata: " + string(raw) + "\n\n")
}

func (s *responsesWriter) raw(str string) {
	if s.writeErr != nil {
		return
	}
	if _, err := io.WriteString(s.w, str); err != nil {
		s.writeErr = err
		return
	}
	if s.fl != nil {
		s.fl.Flush()
	}
}

// responsesError 把 chat 错误帧（{"message","type","code"}）收敛成 Response.error 的 {code, message}。
func responsesError(e map[string]any) map[string]any {
	code, _ := e["code"].(string)
	if code == "" {
		code, _ = e["type"].(string)
	}
	if code == "" {
		code = "upstream_error"
	}
	msg, _ := e["message"].(string)
	if hint, _ := e["gateway_hint"].(string); hint != "" {
		msg += " (" + hint + ")"
	}
	return map[string]any{"code": code, "message": msg}
}
