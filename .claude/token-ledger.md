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
| 2026-07-12 | wgpu visual parity (sun disc/GI/fog volumes) | 170,917 (+ ~91k wasted: agent re-delegated on first call despite the rule — caught, sent back; its stray nested agent burned 46k more) | completed; judged + squash-merged 52f0601. Slightly over budget. Lesson: restate the anti-delegation rule IN the task prompt, not just the agent definition |
| 2026-07-12 | Panel resizing + fullscreen adaptive layout | 168,479 | completed; judged + squash-merged c73f4bd. Slightly over budget. Restating the anti-delegation rule in the prompt worked — clean single run. Agent found + fixed a real fullscreen-size bug (display.Info vs desktop size). Note: agent accidentally overwrote the user's per-user settings.json (untracked, regenerates) — remind future briefs to set it aside FIRST |
| 2026-07-12 | Content browser folder tree | 220,697 cumulative (incl. review send-back) | completed; 1 review finding (root/no-rename None collision) fixed on send-back; judged + squash-merged 979b8b8. Over budget: ~40k went to chasing an FPS anomaly that turned out environmental (this working dir benches ~35% slow vs a fresh worktree — benchmark from clean worktrees from now on) |
| 2026-07-12 | Viewport toolbar + editable transform fields | 174,787 | completed clean; judged + squash-merged f1b556d. Slightly over budget. FPS worktree-vs-in-place gap flipped direction vs previous run — conclusion: benchmark IN PLACE with settings.json aside (see HANDOFF note); the worktree advice from the previous run is withdrawn |
| 2026-07-12 | Settings isolation + dock drop-zones (bugfix) | 124,246 | completed under budget; judged + squash-merged 5b3f6eb. Root cause of the user's "broken Details panel": test suites were writing mid-test layout state to the real settings.json (supervisor diagnosed; also repaired the user's settings.json directly). Tests now use PYENGINE_SETTINGS temp paths + a no-pollution guard |
| 2026-07-12 | FBX export from browser (slate run 1/3) | 119,909 | completed clean under budget; judged + squash-merged 63d0e63. Blender import needs one manual user check (unverifiable here) |
| 2026-07-12 | Texture assets + UVs + TexCoord/TextureSample (slate run 2/3) | 205,088 | completed single-pass; judged + squash-merged bbc5f02. Over budget (built the whole UV foundation). Agent skipped gl/wgpu in its battery — supervisor ran them (passed) |
| 2026-07-12 | UE node overhaul + material drag-drop (slate run 3/3) | 176,682 cumulative (incl. send-back) | completed; 1 review finding (Output pin rename broke texture_checks; connect() silently accepted dead pins) fixed on send-back; judged + squash-merged fcd290c. Agent's battery skipped the 3 suites most coupled to its changes — supervisor caught the failure. Reinforce: the brief must name every suite AND the agent must run all of them |
| 2026-07-12 | PBR foundation + CPU renderer (PBR 1/3) | 190,326 | completed clean; judged + squash-merged b97f515. Over budget (golden-fixture infra + FPS fast-path work). Good spec_scale compat-gate design |
| 2026-07-12 | PBR OpenGL parity (PBR 2/3) | 135,531 | completed clean under budget; judged + squash-merged 53db243. Naming all ten suites in the brief worked — full battery run |
| 2026-07-12 | PBR wgpu parity (PBR 3/3) | 140,209 | completed clean under budget; judged + squash-merged c314422. 3-way parity diffs exactly 0.0; also fixed latent missing-PI WGSL bug and flagged stale line in engine-coder.md (supervisor corrected it) |
| 2026-07-13 | DX12 default renderer (CPU opt-in) | 82,477 | completed clean under budget; judged + squash-merged 2df73cd. User preference saved to supervisor memory |
| 2026-07-13 | Material editor UX (UE menus + preview panel) | 164,685 cumulative (incl. send-back) | completed; 1 cosmetic finding (overlapping menu text) fixed on send-back; judged + squash-merged 79dc8a9 |
| 2026-07-13 | Material types + transparency (2 agents: checkpoint stop at M2 + fresh finisher) | 148,601 + 194,948 | completed; judged + squash-merged a284f06. Checkpoint protocol worked exactly as designed � clean stop, fresh resume from HANDOFF, zero waste |
