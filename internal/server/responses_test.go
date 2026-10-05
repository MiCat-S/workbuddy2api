package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"workbuddy2api/internal/auth"
	"workbuddy2api/internal/upstream"
)

// chatSSE 拼一段 chat.completion.chunk 流：每个 delta 一帧，末帧带 finish_reason（与可选 usage）。
func chatSSE(finish, usage string, deltas ...string) string {
	const head = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,`
	var sb strings.Builder
	for _, d := range deltas {
		sb.WriteString(head + `"delta":` + d + "}]}\n\n")
	}
	fin := head + fmt.Sprintf(`"delta":{},"finish_reason":%q}]`, finish)
	if usage != "" {
		fin += `,"usage":` + usage
	}
	sb.WriteString(fin + "}\n\ndata: [DONE]\n\n")
	return sb.String()
}

// newCapturingUpstream 所有上游调用回同一段 SSE，并记录发往上游的请求体。
func newCapturingUpstream(sse string) (*upstream.Client, func() []map[string]any) {
	var mu sync.Mutex
	var bodies []map[string]any
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			raw, _ := io.ReadAll(r.Body)
			var m map[string]any
			_ = json.Unmarshal(raw, &m)
			mu.Lock()
			bodies = append(bodies, m)
			mu.Unlock()
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(sse)),
			}, nil
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	return up, func() []map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return append([]map[string]any(nil), bodies...)
	}
}

func newResponsesHandler(up *upstream.Client) *Handler {
	return NewHandler(Config{
		Pool:     testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999}),
		Upstream: up,
	})
}

func postResponses(h *Handler, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

type respEvent struct {
	name string
	data map[string]any
}

// parseResponsesSSE 解析 Responses 事件流，并校验 event 名与 data.type 一致、sequence_number 递增。
func parseResponsesSSE(t *testing.T, body string) []respEvent {
	t.Helper()
	var out []respEvent
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" || strings.HasPrefix(block, ":") {
			continue
		}
		var ev respEvent
		for _, ln := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(ln, "event: "):
				ev.name = strings.TrimPrefix(ln, "event: ")
			case strings.HasPrefix(ln, "data: "):
				if err := json.Unmarshal([]byte(strings.TrimPrefix(ln, "data: ")), &ev.data); err != nil {
					t.Fatalf("事件 data 不是 JSON: %q (%v)", ln, err)
				}
			}
		}
		if ev.data["type"] != ev.name {
			t.Errorf("event 名 %q 与 data.type %v 不一致", ev.name, ev.data["type"])
		}
		if seq, _ := ev.data["sequence_number"].(float64); int(seq) != len(out) {
			t.Errorf("sequence_number=%v want %d（事件 %s）", ev.data["sequence_number"], len(out), ev.name)
		}
		out = append(out, ev)
	}
	return out
}

func eventNames(evs []respEvent) []string {
	names := make([]string, len(evs))
	for i, e := range evs {
		names[i] = e.name
	}
	return names
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// TestResponsesToChatConversion 请求翻译：instructions、各类 input 条目的折叠、工具与参数映射。
func TestResponsesToChatConversion(t *testing.T) {
	var req responsesRequest
	if err := json.Unmarshal([]byte(`{
	  "model":"glm-5.2","instructions":"be brief","stream":true,
	  "input":[
	    {"type":"message","role":"developer","content":[{"type":"input_text","text":"dev note"}]},
	    {"role":"user","content":[{"type":"input_text","text":"look"},{"type":"input_image","image_url":"data:image/png;base64,AAA"}]},
	    {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"think A"}],"encrypted_content":"opaque"},
	    {"type":"message","role":"assistant","content":[{"type":"output_text","text":"calling"}]},
	    {"type":"function_call","call_id":"call_1","name":"shell","arguments":"{\"cmd\":\"ls\"}"},
	    {"type":"custom_tool_call","call_id":"call_2","name":"apply_patch","input":"*** Begin Patch"},
	    {"type":"function_call_output","call_id":"call_1","output":"a.txt"},
	    {"type":"custom_tool_call_output","call_id":"call_2","output":[{"type":"input_text","text":"ok"}]},
	    {"type":"web_search_call","id":"ws_1"}
	  ],
	  "tools":[
	    {"type":"function","name":"shell","description":"run","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}},"strict":true},
	    {"type":"custom","name":"apply_patch","description":"patch","format":{"type":"grammar","syntax":"lark","definition":"start: x"}},
	    {"type":"web_search"}
	  ],
	  "tool_choice":{"type":"function","name":"shell"},
	  "parallel_tool_calls":false,"max_output_tokens":100,
	  "reasoning":{"effort":"high","summary":"auto"},
	  "text":{"format":{"type":"json_schema","name":"out","schema":{"type":"object"},"strict":true}},
	  "prompt_cache_key":"conv-1"
	}`), &req); err != nil {
		t.Fatal(err)
	}
	chat, meta, err := responsesToChat(&req)
	if err != nil {
		t.Fatal(err)
	}
	msgs := chat["messages"].([]any)
	wantMsgs := []string{
		`{"content":"be brief","role":"system"}`,
		`{"content":"dev note","role":"developer"}`,
		`{"content":[{"text":"look","type":"text"},{"image_url":{"url":"data:image/png;base64,AAA"},"type":"image_url"}],"role":"user"}`,
		`{"content":"calling","reasoning_content":"think A","role":"assistant","tool_calls":[` +
			`{"function":{"arguments":"{\"cmd\":\"ls\"}","name":"shell"},"id":"call_1","type":"function"},` +
			`{"function":{"arguments":"{\"input\":\"*** Begin Patch\"}","name":"apply_patch"},"id":"call_2","type":"function"}]}`,
		`{"content":"a.txt","role":"tool","tool_call_id":"call_1"}`,
		`{"content":"ok","role":"tool","tool_call_id":"call_2"}`,
	}
	if len(msgs) != len(wantMsgs) {
		t.Fatalf("messages=%d want %d: %s", len(msgs), len(wantMsgs), mustJSON(t, msgs))
	}
	for i, w := range wantMsgs {
		if got := mustJSON(t, msgs[i]); got != w {
			t.Errorf("messages[%d]\n got %s\nwant %s", i, got, w)
		}
	}

	tools := chat["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools=%d want 2（web_search 应被丢弃）: %s", len(tools), mustJSON(t, tools))
	}
	if got := mustJSON(t, tools[0]); got != `{"function":{"description":"run","name":"shell","parameters":{"properties":{"cmd":{"type":"string"}},"type":"object"}},"type":"function"}` {
		t.Errorf("function 工具: %s", got)
	}
	fn := tools[1].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "apply_patch" || !strings.Contains(fn["description"].(string), "lark grammar:\nstart: x") {
		t.Errorf("custom 工具未包装成带语法说明的 function: %s", mustJSON(t, fn))
	}
	if got := mustJSON(t, fn["parameters"]); !strings.Contains(got, `"required":["input"]`) {
		t.Errorf("custom 工具参数: %s", got)
	}
	if !meta.customTools["apply_patch"] || meta.customTools["shell"] {
		t.Errorf("customTools=%v", meta.customTools)
	}

	checks := map[string]string{
		"tool_choice":         `{"function":{"name":"shell"},"type":"function"}`,
		"parallel_tool_calls": `false`,
		"max_tokens":          `100`,
		"reasoning_effort":    `"high"`,
		"response_format":     `{"json_schema":{"name":"out","schema":{"type":"object"},"strict":true},"type":"json_schema"}`,
		"prompt_cache_key":    `"conv-1"`,
		"stream":              `true`,
		"model":               `"glm-5.2"`,
	}
	for k, w := range checks {
		if got := mustJSON(t, chat[k]); got != w {
			t.Errorf("%s=%s want %s", k, got, w)
		}
	}
}

// TestResponsesStringInput input 为字符串时等价于一条 user 消息；effort=none 映射为 off。
func TestResponsesStringInput(t *testing.T) {
	req := responsesRequest{Model: "m", Input: json.RawMessage(`"hi"`)}
	req.Reasoning = &struct {
		Effort  string `json:"effort"`
		Summary string `json:"summary"`
	}{Effort: "none"}
	chat, _, err := responsesToChat(&req)
	if err != nil {
		t.Fatal(err)
	}
	if got := mustJSON(t, chat["messages"]); got != `[{"content":"hi","role":"user"}]` {
		t.Errorf("messages=%s", got)
	}
	if chat["reasoning_effort"] != "off" {
		t.Errorf("reasoning_effort=%v want off", chat["reasoning_effort"])
	}
	if _, err := (func() (map[string]any, error) {
		c, _, e := responsesToChat(&responsesRequest{Model: "m"})
		return c, e
	})(); err == nil {
		t.Error("空 input 应报错")
	}
}

const usageJSON = `{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":2}}`

// TestResponsesStreamReasoningAndText 流式：思考 + 正文两个条目的完整事件序列、终态 output 与 usage。
func TestResponsesStreamReasoningAndText(t *testing.T) {
	up, bodies := newCapturingUpstream(chatSSE("stop", usageJSON,
		`{"role":"assistant","reasoning_content":"想"}`,
		`{"reasoning_content":"一想"}`,
		`{"content":"你"}`,
		`{"content":"好"}`,
	))
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"input":"hi"}`)
	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Errorf("Content-Type=%q", ct)
	}
	evs := parseResponsesSSE(t, rec.Body.String())
	want := []string{
		"response.created", "response.in_progress",
		"response.output_item.added", "response.reasoning_summary_part.added",
		"response.reasoning_summary_text.delta", "response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.done", "response.reasoning_summary_part.done", "response.output_item.done",
		"response.output_item.added", "response.content_part.added",
		"response.output_text.delta", "response.output_text.delta",
		"response.output_text.done", "response.content_part.done", "response.output_item.done",
		"response.completed",
	}
	if got := eventNames(evs); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("事件序列\n got %v\nwant %v", got, want)
	}
	resp := evs[len(evs)-1].data["response"].(map[string]any)
	if resp["status"] != "completed" || resp["object"] != "response" || resp["model"] != "glm-5.2" {
		t.Errorf("终态 response: %s", mustJSON(t, resp))
	}
	output := resp["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output=%s", mustJSON(t, output))
	}
	if got := mustJSON(t, output[0].(map[string]any)["summary"]); got != `[{"text":"想一想","type":"summary_text"}]` {
		t.Errorf("reasoning summary=%s", got)
	}
	msg := output[1].(map[string]any)
	if msg["type"] != "message" || mustJSON(t, msg["content"]) != `[{"annotations":[],"text":"你好","type":"output_text"}]` {
		t.Errorf("message=%s", mustJSON(t, msg))
	}
	if got := mustJSON(t, resp["usage"]); got != `{"input_tokens":10,"input_tokens_details":{"cached_tokens":4},"output_tokens":5,"output_tokens_details":{"reasoning_tokens":2},"total_tokens":15}` {
		t.Errorf("usage=%s", got)
	}
	// 上游收到的是 chat/completions 形态。
	sent := bodies()
	if len(sent) != 1 {
		t.Fatalf("上游调用次数=%d", len(sent))
	}
	msgs, _ := sent[0]["messages"].([]any)
	if len(msgs) == 0 || mustJSON(t, msgs[len(msgs)-1]) != `{"content":"hi","role":"user"}` {
		t.Errorf("上游 messages=%s", mustJSON(t, sent[0]["messages"]))
	}
}

// TestResponsesStreamToolCalls 流式工具调用：并行两路交错到达，function 逐片发参数增量，
// custom 工具（apply_patch）只在 done 时给出还原后的原始 input。
func TestResponsesStreamToolCalls(t *testing.T) {
	up, _ := newCapturingUpstream(chatSSE("tool_calls", "",
		`{"role":"assistant","content":"ok "}`,
		`{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"shell","arguments":""}}]}`,
		`{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":"}}]}`,
		`{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"apply_patch","arguments":"{\"input\":\"*** Begin"}}]}`,
		`{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]}`,
		`{"tool_calls":[{"index":1,"function":{"arguments":" Patch\"}"}}]}`,
	))
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"input":"go",
	  "tools":[{"type":"function","name":"shell","parameters":{"type":"object"}},{"type":"custom","name":"apply_patch"}]}`)
	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	evs := parseResponsesSSE(t, rec.Body.String())
	var argDeltas []string
	for _, e := range evs {
		if e.name == "response.function_call_arguments.delta" {
			argDeltas = append(argDeltas, e.data["delta"].(string))
		}
	}
	if strings.Join(argDeltas, "") != `{"cmd":"ls"}` || len(argDeltas) != 2 {
		t.Errorf("function 参数增量=%q（custom 工具不应发参数增量）", argDeltas)
	}
	resp := evs[len(evs)-1].data["response"].(map[string]any)
	if evs[len(evs)-1].name != "response.completed" {
		t.Fatalf("终态事件=%s", evs[len(evs)-1].name)
	}
	output := resp["output"].([]any)
	if len(output) != 3 {
		t.Fatalf("output=%s", mustJSON(t, output))
	}
	fc := output[1].(map[string]any)
	if fc["type"] != "function_call" || fc["call_id"] != "call_a" || fc["name"] != "shell" || fc["arguments"] != `{"cmd":"ls"}` || fc["status"] != "completed" {
		t.Errorf("function_call=%s", mustJSON(t, fc))
	}
	ctc := output[2].(map[string]any)
	if ctc["type"] != "custom_tool_call" || ctc["call_id"] != "call_b" || ctc["input"] != "*** Begin Patch" {
		t.Errorf("custom_tool_call=%s", mustJSON(t, ctc))
	}
}

// TestResponsesNonStream 非流式：返回完整 Response 对象。
func TestResponsesNonStream(t *testing.T) {
	up, _ := newCapturingUpstream(chatSSE("stop", usageJSON,
		`{"role":"assistant","reasoning_content":"想"}`,
		`{"content":"你好"}`,
	))
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","input":[{"role":"user","content":"hi"}],"metadata":{"k":"v"}}`)
	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Errorf("Content-Type=%q", ct)
	}
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("body 不是 JSON: %s", rec.Body)
	}
	if resp["object"] != "response" || resp["status"] != "completed" || !strings.HasPrefix(resp["id"].(string), "resp_") {
		t.Errorf("response=%s", rec.Body)
	}
	output := resp["output"].([]any)
	if len(output) != 2 || output[0].(map[string]any)["type"] != "reasoning" || output[1].(map[string]any)["type"] != "message" {
		t.Errorf("output=%s", mustJSON(t, output))
	}
	if got := mustJSON(t, output[1].(map[string]any)["content"]); got != `[{"annotations":[],"text":"你好","type":"output_text"}]` {
		t.Errorf("message content=%s", got)
	}
	if mustJSON(t, resp["metadata"]) != `{"k":"v"}` {
		t.Errorf("metadata 未回显: %s", mustJSON(t, resp["metadata"]))
	}
	if u := resp["usage"].(map[string]any); u["input_tokens"] != float64(10) || u["output_tokens"] != float64(5) {
		t.Errorf("usage=%s", mustJSON(t, u))
	}
}

// TestResponsesIncompleteOnLength finish_reason=length → response.incomplete（max_output_tokens）。
func TestResponsesIncompleteOnLength(t *testing.T) {
	up, _ := newCapturingUpstream(chatSSE("length", "", `{"role":"assistant","content":"截断"}`))
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"input":"hi","max_output_tokens":1}`)
	evs := parseResponsesSSE(t, rec.Body.String())
	last := evs[len(evs)-1]
	if last.name != "response.incomplete" {
		t.Fatalf("终态事件=%s want response.incomplete", last.name)
	}
	resp := last.data["response"].(map[string]any)
	if resp["status"] != "incomplete" || mustJSON(t, resp["incomplete_details"]) != `{"reason":"max_output_tokens"}` {
		t.Errorf("response=%s", mustJSON(t, resp))
	}
}

// TestResponsesMidStreamError 流中 error 帧 → 先收尾已打开的条目，再以 response.failed 结束。
func TestResponsesMidStreamError(t *testing.T) {
	sse := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"半"}}]}` + "\n\n" +
		`data: {"error":{"message":"boom","type":"upstream_error","code":"upstream_broken"}}` + "\n\n" +
		"data: [DONE]\n\n"
	up, _ := newCapturingUpstream(sse)
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"input":"hi"}`)
	evs := parseResponsesSSE(t, rec.Body.String())
	names := eventNames(evs)
	if names[len(names)-1] != "response.failed" || names[len(names)-2] != "response.output_item.done" {
		t.Fatalf("事件序列=%v", names)
	}
	resp := evs[len(evs)-1].data["response"].(map[string]any)
	if resp["status"] != "failed" || mustJSON(t, resp["error"]) != `{"code":"upstream_broken","message":"boom"}` {
		t.Errorf("response=%s", mustJSON(t, resp))
	}
}

// TestResponsesUpstreamErrorPassthrough 选号失败等非 200 结果原样透传错误信封，不开事件流。
func TestResponsesUpstreamErrorPassthrough(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		return 429, `{"code":6004,"msg":"usage exceeds frequency limit, but don't worry, your usage will reset at 2099-01-01 00:00:00 UTC+8"}`, false
	})
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"input":"hi"}`)
	if rec.Code < 400 {
		t.Fatalf("code=%d want 4xx/5xx", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); strings.Contains(ct, "event-stream") {
		t.Errorf("错误响应不应是事件流: Content-Type=%q", ct)
	}
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil || !strings.Contains(env.Error.Message, "6004") {
		t.Errorf("body=%s", rec.Body)
	}
}

// TestResponsesRejectsPreviousResponseID 网关不存响应：previous_response_id 明确 400，而不是静默丢上文。
func TestResponsesRejectsPreviousResponseID(t *testing.T) {
	up, bodies := newCapturingUpstream(chatSSE("stop", ""))
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","input":"hi","previous_response_id":"resp_x"}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("code=%d want 400", rec.Code)
	}
	assertJSONErrorCode(t, rec.Body.String(), "unsupported_parameter")
	if n := len(bodies()); n != 0 {
		t.Errorf("被拒请求不应打上游: calls=%d", n)
	}
}

// TestResponsesRequiresAuth 与 chat 同一鉴权。
func TestResponsesRequiresAuth(t *testing.T) {
	up, _ := newCapturingUpstream(chatSSE("stop", ""))
	h := NewHandler(Config{
		Pool:     testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999}),
		Upstream: up,
		APIKey:   "sk-main",
	})
	if rec := postResponses(h, `{"model":"glm-5.2","input":"hi"}`); rec.Code != http.StatusUnauthorized {
		t.Errorf("无 key code=%d want 401", rec.Code)
	}
}
