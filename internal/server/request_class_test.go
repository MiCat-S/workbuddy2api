package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

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
