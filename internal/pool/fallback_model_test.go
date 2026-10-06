package pool

import (
	"testing"
	"time"

	"workbuddy2api/internal/auth"
)

// TestFallbackSkipsModelCooled 全冷却兜底按账号级截止挑最早到期的号，但对请求模型处于
// 独立冷却（6004）的号必须跳过——挑它只会再撞同一个模型级错误。其他模型不受影响。
func TestFallbackSkipsModelCooled(t *testing.T) {
	p := New("")
	p.SetSoftRateMax(24 * time.Hour)
	p.Add(&auth.Auth{UID: "a", AccessToken: "ta", ExpiresAt: 9999999999})
	p.Add(&auth.Auth{UID: "b", AccessToken: "tb", ExpiresAt: 9999999999})
	now := time.Now()
	// 两个号都处于账号级软冷却，a 更早到期（兜底本该优先挑 a）。
	p.CooldownSoftRate("a", time.Minute, now.Add(5*time.Minute), "429 rate limit")
	p.CooldownSoftRate("b", time.Minute, now.Add(30*time.Minute), "429 rate limit")
	// a 的 flash 另有 6004 模型级冷却。
	p.CooldownSoftForModel("a", time.Minute, now.Add(time.Hour), "flash", "6004 model rate limit")

	if got := p.PickExcludingForRealmGroups(nil, "flash", "", nil); got == nil || got.UID != "b" {
		t.Fatalf("flash 兜底应跳过模型冷却中的 a、挑 b，实得 %v", got)
	}
	p.Release("b")
	if got := p.PickExcludingForRealmGroups(nil, "other", "", nil); got == nil || got.UID != "a" {
		t.Fatalf("其他模型不受 a 的 flash 冷却影响，兜底应挑最早到期的 a，实得 %v", got)
	}
	p.Release("a")
	// 唯一可兜底的号对该模型也在冷却：返回 nil（由调用方回限流错误），而不是硬挑它。
	p.CooldownSoftForModel("b", time.Minute, now.Add(time.Hour), "flash", "6004 model rate limit")
	if got := p.PickExcludingForRealmGroups(nil, "flash", "", nil); got != nil {
		t.Fatalf("所有号对 flash 都在模型冷却，兜底应返回 nil，实得 %v", got.UID)
	}
}
