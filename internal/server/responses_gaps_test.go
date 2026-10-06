package server

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"workbuddy2api/internal/auth"
	"workbuddy2api/internal/upstream"
)

// TestResponsesStreamMissingFinishIsFailed 上游 EOF 时 StreamHint 会自动补 [DONE]；
// 没有 finish_reason 就结束的流是被截断的，不能报 response.completed。
func TestResponsesStreamMissingFinishIsFailed(t *testing.T) {
	sse := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"半截"}}]}` + "\n\n"
	up, _ := newCapturingUpstream(sse)
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"input":"hi"}`)
	evs := parseResponsesSSE(t, rec.Body.String())
	last := evs[len(evs)-1]
	if last.name != "response.failed" {
		t.Fatalf("终态事件=%s want response.failed（事件序列 %v）", last.name, eventNames(evs))
	}
	if code := last.data["response"].(map[string]any)["error"].(map[string]any)["code"]; code != "stream_interrupted" {
		t.Errorf("error.code=%v", code)
	}
}

// TestResponsesDropsTruncatedToolCall finish_reason=length 时参数残缺的工具调用不发 done、
// 不进 output（与非流式 Aggregate 同口径）；参数完整的照常保留。
func TestResponsesDropsTruncatedToolCall(t *testing.T) {
	up, _ := newCapturingUpstream(chatSSE("length", "",
		`{"role":"assistant","tool_calls":[{"index":0,"id":"call_ok","type":"function","function":{"name":"shell","arguments":"{\"cmd\":\"ls\"}"}}]}`,
		`{"tool_calls":[{"index":1,"id":"call_cut","type":"function","function":{"name":"shell","arguments":"{\"cmd\":\"rm -"}}]}`,
	))
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"input":"go","tools":[{"type":"function","name":"shell"}]}`)
	evs := parseResponsesSSE(t, rec.Body.String())
	for _, e := range evs {
		if e.name == "response.output_item.done" {
			if item := e.data["item"].(map[string]any); item["call_id"] == "call_cut" {
				t.Errorf("截断的工具调用不应发 output_item.done: %s", mustJSON(t, item))
			}
		}
	}
	last := evs[len(evs)-1]
	if last.name != "response.incomplete" {
		t.Fatalf("终态事件=%s", last.name)
	}
	output := last.data["response"].(map[string]any)["output"].([]any)
	if len(output) != 1 || output[0].(map[string]any)["call_id"] != "call_ok" {
		t.Errorf("output 应只剩完整的 call_ok: %s", mustJSON(t, output))
	}
}

// TestResponsesContextLengthCode 网关的 prompt_too_long 在 Responses 入口改写成
// context_length_exceeded（Codex 据此触发上下文压缩），状态码与原文不变。
func TestResponsesContextLengthCode(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		return 400, `{"code":11115,"msg":"prompt is too long: 120000 tokens > 65536 maximum"}`, false
	})
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"input":"hi"}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	assertJSONErrorCode(t, rec.Body.String(), "context_length_exceeded")
	if !strings.Contains(rec.Body.String(), "120000 tokens") {
		t.Errorf("上游原文丢失: %s", rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Errorf("Content-Type=%q", ct)
	}
}

// TestResponsesUnsupportedPathsJSON 网关不存响应：其余方法与子路径回 JSON 错误信封。
func TestResponsesUnsupportedPathsJSON(t *testing.T) {
	up, _ := newCapturingUpstream(chatSSE("stop", ""))
	h := newResponsesHandler(up)
	cases := []struct {
		method, path string
		code         int
		errCode      string
	}{
		{"GET", "/v1/responses", http.StatusMethodNotAllowed, "method_not_allowed"},
		{"GET", "/v1/responses/resp_abc", http.StatusNotFound, "not_found"},
		{"POST", "/v1/responses/resp_abc/cancel", http.StatusNotFound, "not_found"},
		{"POST", "/v1/responses/compact", http.StatusNotFound, "not_found"},
	}
	for _, c := range cases {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(c.method, c.path, strings.NewReader(`{}`)))
		if rec.Code != c.code {
			t.Errorf("%s %s code=%d want %d", c.method, c.path, rec.Code, c.code)
		}
		assertJSONErrorCode(t, rec.Body.String(), c.errCode)
	}
}

// TestResponsesLateToolNameAndID 首片既没 id 也没 name、后续分片才补：done 条目要用迟到的值。
func TestResponsesLateToolNameAndID(t *testing.T) {
	up, _ := newCapturingUpstream(chatSSE("tool_calls", "",
		`{"role":"assistant","tool_calls":[{"index":0,"type":"function","function":{"arguments":""}}]}`,
		`{"tool_calls":[{"index":0,"id":"call_late","function":{"name":"apply_patch","arguments":"{\"input\":\"x\"}"}}]}`,
	))
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"input":"go","tools":[{"type":"custom","name":"apply_patch"}]}`)
	evs := parseResponsesSSE(t, rec.Body.String())
	output := evs[len(evs)-1].data["response"].(map[string]any)["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output=%s", mustJSON(t, output))
	}
	it := output[0].(map[string]any)
	if it["type"] != "custom_tool_call" || it["name"] != "apply_patch" || it["call_id"] != "call_late" || it["input"] != "x" {
		t.Errorf("迟到的 name/id 未生效: %s", mustJSON(t, it))
	}
}

// TestResponsesReasoningPlacement 思考只挂到紧随其后的 assistant：中间插进 user 消息就丢弃；
// 同一轮多段思考串接而非覆盖。
func TestResponsesReasoningPlacement(t *testing.T) {
	var req responsesRequest
	_ = json.Unmarshal([]byte(`{"model":"m","input":[
	  {"type":"reasoning","summary":[{"type":"summary_text","text":"orphan"}]},
	  {"role":"user","content":"interrupted, new question"},
	  {"type":"reasoning","summary":[{"type":"summary_text","text":"A"}]},
	  {"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
	  {"type":"reasoning","summary":[{"type":"summary_text","text":"B"}]},
	  {"type":"function_call","call_id":"c2","name":"f","arguments":"{}"},
	  {"type":"function_call_output","call_id":"c1","output":"1"},
	  {"type":"function_call_output","call_id":"c2","output":"2"},
	  {"type":"message","role":"assistant","content":"done"}
	]}`), &req)
	chat, _, err := responsesToChat(&req)
	if err != nil {
		t.Fatal(err)
	}
	msgs := chat["messages"].([]any)
	asst := msgs[1].(map[string]any)
	if asst["reasoning_content"] != "A\n\nB" {
		t.Errorf("同轮多段思考应串接: %v", asst["reasoning_content"])
	}
	final := msgs[len(msgs)-1].(map[string]any)
	if _, has := final["reasoning_content"]; has {
		t.Errorf("被 user 打断的思考不应挂到后面的 assistant: %s", mustJSON(t, final))
	}
}

// TestChatClientCancelNotPenalized 客户端在等首字节时取消：传输层错误只是取消的回声，
// 不罚号（不记连败）、不换号重试。
func TestChatClientCancelNotPenalized(t *testing.T) {
	var calls atomic.Int32
	entered := make(chan struct{}, 4)
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			calls.Add(1)
			entered <- struct{}{}
			<-r.Context().Done() // 模拟上游迟迟不回首字节，直到请求被取消
			return nil, r.Context().Err()
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	p := testPoolWith(
		&auth.Auth{UID: "a1", AccessToken: "at1", ExpiresAt: 9999999999},
		&auth.Auth{UID: "a2", AccessToken: "at2", ExpiresAt: 9999999999},
	)
	h := NewHandler(Config{Pool: p, Upstream: up})
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-entered
		cancel() // 请求已打到上游、等首字节期间客户端断开
	}()
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}]}`)).WithContext(ctx)
	h.ServeHTTP(httptest.NewRecorder(), req)
	if n := calls.Load(); n != 1 {
		t.Errorf("客户端已断开仍换号重试: upstream calls=%d want 1", n)
	}
	for _, uid := range []string{"a1", "a2"} {
		if st, _ := p.Status(uid); st.ConsecutiveFails != 0 {
			t.Errorf("%s 因客户端取消被记连败: %d", uid, st.ConsecutiveFails)
		}
	}
}

// TestResponsesAcceptsChatMessages 客户端（如 Nagram）把 chat 形态的 messages 发到 /v1/responses、
// 不带 input：原样采用 messages，而不是回 400（下游中转会因此把整个供应商熔断）。
func TestResponsesAcceptsChatMessages(t *testing.T) {
	up, bodies := newCapturingUpstream(chatSSE("stop", "", `{"role":"assistant","content":"hi there"}`))
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,
	  "messages":[{"role":"system","content":"be nice"},{"role":"user","content":"hello"}]}`)
	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	evs := parseResponsesSSE(t, rec.Body.String())
	if last := evs[len(evs)-1]; last.name != "response.completed" {
		t.Fatalf("终态事件=%s", last.name)
	}
	sent := bodies()
	msgs, _ := sent[0]["messages"].([]any)
	got := mustJSON(t, msgs[len(msgs)-1])
	if got != `{"content":"hello","role":"user"}` {
		t.Errorf("上游末条消息=%s", got)
	}
}

// TestResponsesEmptyInputStill400 既没有 input 也没有 messages：仍然 400，并记录诊断日志。
func TestResponsesEmptyInputStill400(t *testing.T) {
	up, _ := newCapturingUpstream(chatSSE("stop", ""))
	var buf strings.Builder
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)
	rec := postResponses(newResponsesHandler(up), `{"model":"glm-5.2","stream":true,"foo":1}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("code=%d", rec.Code)
	}
	if !strings.Contains(buf.String(), "body keys=foo,model,stream") || !strings.Contains(buf.String(), "input=missing") {
		t.Errorf("诊断日志缺结构信息: %q", buf.String())
	}
}
