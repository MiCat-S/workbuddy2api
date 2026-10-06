package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"workbuddy2api/internal/auth"
	"workbuddy2api/internal/upstream"
)

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
