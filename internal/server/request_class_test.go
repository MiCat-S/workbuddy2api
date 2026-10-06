package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"workbuddy2api/internal/auth"
)

func threeAccounts() []*auth.Auth {
	return []*auth.Auth{
		{UID: "a1", AccessToken: "at1", ExpiresAt: 9999999999},
		{UID: "a2", AccessToken: "at2", ExpiresAt: 9999999999},
		{UID: "a3", AccessToken: "at3", ExpiresAt: 9999999999},
	}
}

// TestChatBadParamsStopsAfterSecondAccount 11101 请求体解析失败：换一个号确认后即停（不打满
// 全部账号），末端回 400 + invalid_request_error（不是 503，免得 SDK 重试与下游中转熔断）。
func TestChatBadParamsStopsAfterSecondAccount(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		mu.Lock()
		calls++
		mu.Unlock()
		return 400, `{"code":11101,"msg":"Unmarshal chat params failed with error: unexpected EOF"}`, false
	})
	h := NewHandler(Config{Pool: testPoolWith(threeAccounts()...), Upstream: up, MaxRotate: 3})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("code=%d want 400 body=%s", rec.Code, rec.Body)
	}
	assertJSONErrorCode(t, rec.Body.String(), "invalid_request_error")
	if calls != 2 {
		t.Errorf("11101 应只换一次号确认：upstream calls=%d want 2", calls)
	}
}

// TestChatMixedFailuresStay503 失败中混有上游故障（5xx）时不能归因为请求问题：维持 503。
func TestChatMixedFailuresStay503(t *testing.T) {
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		if authz == "Bearer at1" {
			return 500, `{"code":500,"msg":"internal error"}`, false
		}
		return 400, `{"code":400,"msg":"bad request"}`, false
	})
	h := NewHandler(Config{Pool: testPoolWith(threeAccounts()...), Upstream: up, MaxRotate: 3})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("code=%d want 503（混有上游故障）body=%s", rec.Code, rec.Body)
	}
}

// TestResponsesRequestClassIs4xx Responses 入口同样拿到 4xx（下游中转按错误规则处理，不熔断）。
func TestResponsesRequestClassIs4xx(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		return 400, `{"code":400,"msg":"bad request"}`, false
	})
	h := NewHandler(Config{Pool: testPoolWith(threeAccounts()...), Upstream: up})
	rec := postResponses(h, `{"model":"glm-5.2","stream":true,"input":"hi"}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("code=%d want 400 body=%s", rec.Code, rec.Body)
	}
	assertJSONErrorCode(t, rec.Body.String(), "invalid_request_error")
}

// TestChatAllModelCooledExplains 所有可用账号对请求模型都处于 6004 模型级冷却、本次一个上游都
// 没打：回 429 并说明是哪个模型、最早几点恢复，而不是笼统的「所有账号暂时不可用」。
func TestChatAllModelCooledExplains(t *testing.T) {
	calls := 0
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		calls++
		return 429, `{"code":6004,"msg":"usage exceeds frequency limit, your usage will reset at 2099-01-02 03:04:05 UTC+8"}`, false
	})
	h := NewHandler(Config{
		Pool:         testPoolWith(&auth.Auth{UID: "a1", AccessToken: "at1", ExpiresAt: 9999999999}),
		Upstream:     up,
		SoftCooldown: time.Minute,
	})
	req := func() *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
			strings.NewReader(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}]}`)))
		return rec
	}
	req() // 首个请求撞 6004，写模型级冷却
	rec := req()
	if calls != 1 {
		t.Errorf("模型冷却期间不应再打上游: calls=%d", calls)
	}
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code=%d want 429 body=%s", rec.Code, rec.Body)
	}
	assertJSONErrorCode(t, rec.Body.String(), "rate_limit_exceeded")
	for _, want := range []string{"glm-5.2", "rate-limited", "2099-01-02 03:04:05"} {
		if !strings.Contains(rec.Body.String(), want) {
			t.Errorf("message 缺少 %q: %s", want, rec.Body)
		}
	}
}
