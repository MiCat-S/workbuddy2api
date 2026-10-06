package upstream

import (
	"errors"
	"io"
	"strings"
	"testing"
)

// TestPeekFirstFrameReplays 偷看首帧后，返回的 reader 必须逐字节重放原始流。
func TestPeekFirstFrameReplays(t *testing.T) {
	raw := ": ping\n\ndata: {\"id\":\"c1\"}\n\ndata: {\"id\":\"c2\"}\n\ndata: [DONE]\n\n"
	r, first := PeekFirstFrame(strings.NewReader(raw))
	if first != `{"id":"c1"}` {
		t.Errorf("first=%q", first)
	}
	got, err := io.ReadAll(r)
	if err != nil || string(got) != raw {
		t.Errorf("重放不一致: err=%v\n got %q\nwant %q", err, got, raw)
	}

	// 没有 data 帧：payload 为空，重放原样内容。
	r, first = PeekFirstFrame(strings.NewReader(": only comment\n"))
	if got, _ := io.ReadAll(r); first != "" || string(got) != ": only comment\n" {
		t.Errorf("first=%q replay=%q", first, got)
	}

	// 首帧前读错误：重放已读内容后再给出同一个错误。
	boom := errors.New("boom")
	r, _ = PeekFirstFrame(io.MultiReader(strings.NewReader(": x\n"), errorReader{boom}))
	if _, err := io.ReadAll(r); !errors.Is(err, boom) {
		t.Errorf("err=%v want boom", err)
	}
}

func TestIsErrorFrame(t *testing.T) {
	cases := map[string]bool{
		`{"error":{"code":"6004"}}`:                       true,
		`{"error":"boom"}`:                                true,
		`{"error":null}`:                                  false,
		`{"choices":[{"delta":{"content":"错误"}}]}`:        false,
		`{"choices":[{"delta":{"content":"\"error\""}}]}`: false,
		`[DONE]`: false,
		``:       false,
	}
	for in, want := range cases {
		if got := IsErrorFrame(in); got != want {
			t.Errorf("IsErrorFrame(%q)=%v want %v", in, got, want)
		}
	}
}

// TestAggregateReturnsErrorFrame 非流式聚合遇到 error 帧必须交回错误，不能产出 200 + 空 content。
func TestAggregateReturnsErrorFrame(t *testing.T) {
	raw := `data: {"id":"c1","choices":[{"index":0,"delta":{"content":"半"}}]}` + "\n\n" +
		`data: {"error":{"message":"rate limited","code":"6004"}}` + "\n\n" +
		"data: [DONE]\n\n"
	resp, err := Aggregate(strings.NewReader(raw))
	var ef *StreamErrorFrame
	if !errors.As(err, &ef) || resp != nil {
		t.Fatalf("resp=%v err=%v，应返回 *StreamErrorFrame", resp, err)
	}
	if !strings.Contains(ef.Payload, `"code":"6004"`) {
		t.Errorf("payload=%q", ef.Payload)
	}
}
