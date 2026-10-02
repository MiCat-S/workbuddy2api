package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"workbuddy2api/internal/pool"
)

// TestOpsEndpointsRejectGroupKeys 全池视图 / 全局操作端点只对不限分组的密钥开放：
// 分组密钥是发给外部调用方的，不能借 /status 看到其他分组账号、清全局统计、
// 或对全池触发签到。主密钥（控制台）不受影响。
func TestOpsEndpointsRejectGroupKeys(t *testing.T) {
	h := NewHandler(Config{
		Pool: pool.New(""),
		AuthKeys: []AuthKey{
			{Key: "sk-main", Name: "主密钥"},
			{Key: "sk-ext", Name: "外部", Groups: []string{"external"}},
		},
		MetricsEnabled: true,
		CheckinFn: func() (CheckinReport, bool, error) {
			return CheckinReport{}, false, nil
		},
	})
	routes := []struct{ method, path string }{
		{"GET", "/status"},
		{"GET", "/v1/stats"},
		{"POST", "/v1/stats/reset"},
		{"POST", "/v1/checkin"},
		{"GET", "/metrics"},
	}
	for _, rt := range routes {
		req := httptest.NewRequest(rt.method, rt.path, nil)
		req.Header.Set("Authorization", "Bearer sk-ext")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden {
			t.Errorf("分组密钥 %s %s code=%d want 403", rt.method, rt.path, rec.Code)
		}

		req = httptest.NewRequest(rt.method, rt.path, nil)
		req.Header.Set("Authorization", "Bearer sk-main")
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code == http.StatusForbidden || rec.Code == http.StatusUnauthorized {
			t.Errorf("主密钥 %s %s code=%d，不应被拒", rt.method, rt.path, rec.Code)
		}
	}
}

// TestCheckinCooldownStampsCompletion 冷却按签到**完成**时刻起算：一轮签到耗时超过
// 冷却窗口时，按开始时刻盖章会让冷却在返回时已过期，脚本可背靠背连续触发。
func TestCheckinCooldownStampsCompletion(t *testing.T) {
	h := NewHandler(Config{CheckinFn: func() (CheckinReport, bool, error) {
		time.Sleep(1100 * time.Millisecond)
		return CheckinReport{Total: 1}, false, nil
	}})
	start := time.Now().Unix()
	if rec := checkinPost(t, h, ""); rec.Code != 200 {
		t.Fatalf("code=%d want 200", rec.Code)
	}
	if got := h.lastCheckinUnix.Load(); got < start+1 {
		t.Errorf("冷却盖章=%d，早于完成时刻（开始=%d，签到耗时 >1s）", got, start)
	}
}
