# Agent token ledger

Running record of engine-coder task runs and their reported token usage,
maintained by the supervising session after every run. Purpose: budget
awareness (account usage limits are NOT queryable programmatically — this
ledger is the only consumption signal we have) and scoping future tasks.

**Budget policy:** target <= ~150k tokens per agent run. Larger features get
split into sequential runs. Every run follows the wip-branch checkpoint
protocol (see `.claude/agents/engine-coder.md`) so a usage-limit or session
death never loses more than one milestone.

| Date | Task | Tokens | Outcome |
|---|---|---|---|
| 2026-07-09 | Ctrl+D carries entity state | 70,765 | completed, merged |
| 2026-07-10 | Depth resolve + Unreal fly controls | 125,164 | completed, merged |
| 2026-07-10 | Editor platform (menus/docking/settings) | 249,857 | completed, merged (over budget — should have been 2 runs) |
| 2026-07-10 | OpenGL GPU backend (moderngl) | 300,869 | completed, merged (over budget — should have been 2 runs) |
| 2026-07-10 | DX12/Vulkan dispatch attempt | 47,763 | wasted: agent re-delegated instead of implementing (rule added) |
| 2026-07-10 | DX12/Vulkan backend (wgpu) | unknown (nested + resumed runs) | DIED AT SESSION USAGE LIMIT mid-verification; work salvaged to wip/dx12-vulkan-backend at 1cc35d6; resumed |
| 2026-07-10 | DX12/Vulkan resume (transcript resume) | unknown | DIED AT LIMIT AGAIN mid-settings-fix; salvaged at 57bf45f. Lesson: resuming a huge transcript re-reads it all at cold rates — after a death, start a FRESH agent from the wip branch + HANDOFF instead. Remaining: settings row-spacing fix + full verification battery. |
| 2026-07-10 | DX12/Vulkan finisher (fresh from checkpoint) | 77,335 | completed under budget; judged + merged to main |
| 2026-07-11 | Environment assets (sun/sky material/GI/fog volumes) | 461,730 cumulative (5 resumes incl. 2 limit deaths) | completed; 1 review finding (GI receiver gating) fixed on send-back; judged + squash-merged e7b7735. Over budget: should have been 2-3 runs |
| 2026-07-11 | Window management (minimize/close/registry/reset) | 143,784 | completed under budget; judged + squash-merged |
