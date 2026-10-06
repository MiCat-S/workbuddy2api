package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"workbuddy2api/internal/auth"
	"workbuddy2api/internal/pool"
	"workbuddy2api/internal/session"
)

// 上游 HTTP 200 开流、流里下发的 6004 error 帧（带重置时间，模型级限流）。
const frame6004 = `data: {"error":{"message":"usage exceeds frequency limit, but don't worry, your usage will reset at 2099-01-01 00:00:00 UTC+8","code":"6004"}}` + "\n\n"

const contentFrame = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"partial"}}]}` + "\n\n"

func modelCooled(p *pool.Pool, uid, model string) bool {
	st, _ := p.Status(uid)
	for _, m := range st.RateLimitedModels {
		if m.Model == model {
			return true
		}
	}
	return false
}

// TestChatFirstFrameErrorRotates 首帧即 error：按 HTTP 错误同等处置——该号写模型冷却、
// 不算成功，请求换号由健康号服务；客户端只看到成功号的输出。
func TestChatFirstFrameErrorRotates(t *testing.T) {
	calls := map[string]int{}
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		calls[authz]++
		if authz == "Bearer at-bad" {
			return 200, frame6004 + "data: [DONE]\n\n", true
		}
		return 200, sseOK, true
	})
	p := testPoolWith(
		&auth.Auth{UID: "bad", AccessToken: "at-bad", ExpiresAt: 9999999999},
		&auth.Auth{UID: "good", AccessToken: "at-good", ExpiresAt: 9999999999},
	)
	h := NewHandler(Config{Pool: p, Upstream: up, SoftCooldown: time.Minute})
	for _, stream := range []string{"true", "false"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
			strings.NewReader(`{"model":"glm-5.2","stream":`+stream+`,"messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != 200 || !strings.Contains(rec.Body.String(), "你好") {
			t.Fatalf("stream=%s code=%d body=%s", stream, rec.Code, rec.Body)
		}
		if strings.Contains(rec.Body.String(), "6004") {
			t.Errorf("stream=%s: 首帧 error 不应透给客户端（已换号成功）: %s", stream, rec.Body)
		}
		if got := rec.Header().Get("X-Wb-Account"); got != "good" {
			t.Errorf("stream=%s: X-Wb-Account=%q want good", stream, got)
		}
	}
	if !modelCooled(p, "bad", "glm-5.2") {
		t.Error("首帧 6004 应写入模型级冷却")
	}
	if calls["Bearer at-bad"] != 1 {
		t.Errorf("冷却后不应再选中 bad: calls=%v", calls)
	}
}

// TestChatFirstFrameErrorAllAccounts 所有号都首帧 6004：回 429 错误信封（不开事件流）。
func TestChatFirstFrameErrorAllAccounts(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		return 200, frame6004 + "data: [DONE]\n\n", true
	})
	h := NewHandler(Config{
		Pool:         testPoolWith(&auth.Auth{UID: "a1", AccessToken: "at1", ExpiresAt: 9999999999}),
		Upstream:     up,
		SoftCooldown: time.Minute,
	})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code=%d want 429 body=%s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); strings.Contains(ct, "event-stream") {
		t.Errorf("全部失败不应开事件流: %q", ct)
	}
	if !strings.Contains(rec.Body.String(), "6004") {
		t.Errorf("应透传上游原文: %s", rec.Body)
	}
}

// TestChatMidStreamErrorFrame 开流后才出现的 error 帧：流照常透传（头已发出），但账号侧
// 写模型冷却并解除本次粘性绑定。
func TestChatMidStreamErrorFrame(t *testing.T) {
	store := newBindStore()
	sess := session.New(session.Config{
		TTL:       time.Minute,
		Store:     store,
		Available: func() []string { return []string{"a1"} },
	})
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		return 200, contentFrame + frame6004 + "data: [DONE]\n\n", true
	})
	p := testPoolWith(&auth.Auth{UID: "a1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, Session: sess, SoftCooldown: time.Minute})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}],"metadata":{"conversation_id":"conv-e"}}`)))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "partial") || !strings.Contains(rec.Body.String(), "6004") {
		t.Fatalf("流应原样透传（含 error 帧）: code=%d body=%s", rec.Code, rec.Body)
	}
	if !modelCooled(p, "a1", "glm-5.2") {
		t.Error("流中途 6004 应写入模型级冷却")
	}
	if uid, ok := store.lastUID("conv-e"); ok {
		t.Errorf("流中途 error 后粘性绑定应解除，仍绑定 %s", uid)
	}
}

// TestChatNonStreamMidStreamErrorFrame 非流式：流中途 error 帧不再变成「200 + 空 content」
// 的假成功，而是按分类回错误（6004 → 429），并写模型冷却。
func TestChatNonStreamMidStreamErrorFrame(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		return 200, contentFrame + frame6004 + "data: [DONE]\n\n", true
	})
	p := testPoolWith(&auth.Auth{UID: "a1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, SoftCooldown: time.Minute})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","stream":false,"messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code=%d want 429 body=%s", rec.Code, rec.Body)
	}
	assertJSONErrorCode(t, rec.Body.String(), "rate_limit_exceeded")
	if !strings.Contains(rec.Body.String(), "usage exceeds frequency limit") {
		t.Errorf("应带上游原文: %s", rec.Body)
	}
	if !modelCooled(p, "a1", "glm-5.2") {
		t.Error("应写入模型级冷却")
	}
}
