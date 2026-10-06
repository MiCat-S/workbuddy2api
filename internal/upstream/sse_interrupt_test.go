package upstream

import (
	"errors"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestStreamHintMidStreamReadError 流已开始后中途读失败：必须在流里写一帧 error（客户端
// 才知道回答被截断），不补 [DONE]，并返回可识别的 errStreamInterrupted。
func TestStreamHintMidStreamReadError(t *testing.T) {
	frame := `data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"半"}}]}` + "\n\n"
	r := io.MultiReader(strings.NewReader(frame), errorReader{errors.New("connection reset by peer")})
	rec := httptest.NewRecorder()
	err := StreamHint(rec, r, nil)
	if !IsStreamInterrupted(err) {
		t.Fatalf("err=%v，应为 errStreamInterrupted", err)
	}
	if IsEmptyStreamError(err) {
		t.Error("中途读失败不应被判成空流")
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"code":"stream_interrupted"`) || !strings.Contains(body, "connection reset by peer") {
		t.Errorf("缺少 error 帧: %q", body)
	}
	if strings.Contains(body, "[DONE]") {
		t.Errorf("中途读失败不应补 [DONE]: %q", body)
	}
}

// TestStreamHintLateToolName 首片不带 name、后续分片才补：迟到的 name 必须保留（只出现一次）。
func TestStreamHintLateToolName(t *testing.T) {
	sse := `data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"arguments":""}}]}}]}` + "\n\n" +
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"shell","arguments":"{}"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"shell","arguments":""}}]}}]}` + "\n\n" +
		"data: [DONE]\n\n"
	rec := httptest.NewRecorder()
	if err := StreamHint(rec, strings.NewReader(sse), nil); err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(rec.Body.String(), `"name":"shell"`); n != 1 {
		t.Errorf("name 出现 %d 次 want 1（迟到的 name 被删或重复下发）: %s", n, rec.Body)
	}
}

type errorReader struct{ err error }

func (r errorReader) Read([]byte) (int, error) { return 0, r.err }
