# UPSTREAM.md — Lila ↔ public SkyRL sync ledger

Tracks how `fl97inc/SkyRL` (private, standalone) relates to `NovaSky-AI/SkyRL` (public). The
recurring-sync SOP (`upstream-sync/02-recurring-workflow.md`) reads and updates this file every
cycle. Keep it accurate — the "drop on contact" step is only deterministic if the drop-list below
is current.

Upstream remote: `https://github.com/NovaSky-AI/SkyRL.git` · sync target: **`upstream/main`**
(release tags are months apart; `main` is the operative cadence).

> **Branching model (load-bearing):** `main` is a **zero-commit mirror** of `upstream/main` and
> advances only by `git merge --ff-only`. Any non-upstream commit on `main` breaks the
> fast-forward. All bespoke work — including this ledger, `CODEOWNERS`, the sync workflow, and the
> `upstream-sync/` runbooks — lives on **`develop`**, the long-lived integration branch.

## 1. Last sync

| upstream SHA | upstream tag (if any) | sync date | `lila-sync-*` tag |
|---|---|---|---|
| `f508739f` | — | 2026-06-10 | `lila-sync-f508739f` |

> Seed value: the `upstream/main` SHA fork `main` was reset to in the one-time catch-up
> (2026-06-10). `main` was verified byte-identical to `upstream/main` (`f508739f`) — 0 ahead /
> 0 behind. Update this row every cycle (SOP step 7).

## 2. Carried patches (what diverges from a pure upstream mirror)

`main` carries **zero** commits. The delta lives entirely on topic branches that PR into
`develop`. Goal: keep this list short. `category` ∈ {`keep-fork-only`, `pending-upstream`,
`evaluate`}.

| branch | internal PR# | category | public PR# (if pending) | hotspot files touched |
|---|---|---|---|---|
| `feat/torch-profiler-driving` | #16 | pending-upstream | _TODO: NovaSky PR# once opened_ | `trainer.py`, `config.py`, worker dispatch, profiler |
| `vdinh/wandb-tags-tracking` | #14 | pending-upstream | _TODO: NovaSky PR#_ | `config.py`, `sft_config.py`, `tracking.py` |
| `feat/sync-weights-subtimers` | #15 | keep-fork-only | — | `broadcast_strategy.py`, `fully_async_trainer.py`, worker dispatch |
| `vdinh/fix-max-training-steps-and-hang` | #5 | keep-fork-only | — | `config.py`, `trainer.py` |
| `tito_example` | #6 | keep-fork-only | — | examples only |
| `achaloo/mfu` | #4 | evaluate | — | `trainer.py`, `flops_tracker.py` (true MFU %, distinct from upstream's tokens/sec) |
| `log_val_samples` | #7 | evaluate | — | `evaluate.py`, `tracking.py`, `trajectory_logging.py` |
| `optimize-sft-tokenization` | #9 | evaluate | — | `sft_trainer.py`, `sft_config.py` |

> **`evaluate` = author decision pending.** These were *not* cleanly superseded by upstream
> (verified 2026-06-10): #4 adds a true MFU-% tracker (`flops_tracker.py`), distinct from
> upstream's `tokens_per_second_per_gpu` (#1711); #7's would-be twin `c60a1b91` is on an
> *unmerged* NovaSky branch (`upstream/log_val_samples`), not in `main`; #9 overlaps upstream
> #1695 in intent but uses a different chunk-worker implementation (~93/377 added lines absent
> from `main`). Authors decide: rebase onto `develop`, upstream, or drop.

## 2.5. Live `develop` ↔ `main` delta (AUTO-GENERATED)

<!--
  Everything between the BEGIN/END AUTO markers below is regenerated on every merge to `develop`
  by .github/workflows/develop-delta.yaml (script: upstream-sync/gen_develop_delta.sh).
  DO NOT edit inside the markers by hand — your changes will be overwritten on the next merge.
  §2 above (curated categories / public PR#s) is hand-maintained and is NEVER touched by the bot.
-->

<!-- BEGIN AUTO:develop-delta -->
_Last generated for `origin/main` (fd79ceec) ↔ `HEAD` (250cdd85); merge-base `7f453704`. **Auto-generated — do not edit by hand.**_

`develop` is **72 commit(s)** ahead of `main`.

### Commits on `develop` not on `main`

| commit | subject | author | date |
|---|---|---|---|
| `250cdd85` | fix(r3): align routed-expert metadata and padding (#56) | dyurk-lila | 2026-07-20 |
| `63fbf69f` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-14 |
| `ded01538` | chore(sync): upstream 02700539 (CONFLICTS — manual resolve)  | lila-ci-bot[bot] | 2026-07-14 |
| `a7df0d99` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-13 |
| `f141d150` | [bugfix] Fused LM Head Compatibility with HybridModel (#53) | dyurk-lila | 2026-07-13 |
| `1b3f2bc3` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-13 |
| `ef991ff1` | ci(sync): preserve pre-pushed PR branches (#54) | dyurk-lila | 2026-07-13 |
| `62393b91` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-09 |
| `8327e4fd` | chore(sync): upstream e17527a1 (CONFLICTS — manual resolve)  | dyurk-lila | 2026-07-09 |
| `23e475f6` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-08 |
| `ac182bd2` | ci: add agent conflict resolver and ml.train e2e tests for u | dyurk-lila | 2026-07-08 |
| `56eaedaa` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-30 |
| `e5d43639` | feat(algo): add cispo_anchor to select old vs rollout IS-rat | tbalestri-lila | 2026-06-30 |
| `31860caf` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-29 |
| `f00c895a` | chore(sync): upstream 76f5f467 (CONFLICTS — manual resolve)  | lila-ci-bot[bot] | 2026-06-29 |
| `81a992bc` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-29 |
| `343bde7b` | ci(sync): use native GitHub conflicts for sync PRs | lila-sync-bot | 2026-06-29 |
| `d2f3b6e6` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-29 |
| `70d925aa` | ci(sync): publish conflicted sync results | lila-sync-bot | 2026-06-29 |
| `39907cab` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-29 |
| `a99938a2` | ci(sync): allow conflict sync PR creation | lila-sync-bot | 2026-06-29 |
| `cde2d714` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-29 |
| `a83c467c` | ci(sync): create sync PRs through REST API | lila-sync-bot | 2026-06-29 |
| `59f27167` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-29 |
| `1123cdf9` | ci(sync): request app permissions for sync PRs | lila-sync-bot | 2026-06-29 |
| `450c4b02` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-29 |
| `33247944` | ci(sync): use workflow token for sync PRs | lila-sync-bot | 2026-06-29 |
| `8cccd284` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-29 |
| `7a6a21a9` | ci(sync): tolerate missing upstream-sync label permission | lila-sync-bot | 2026-06-29 |
| `4c64c7c8` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-26 |
| `0cde1434` | ci(sync): use lila bot token for upstream sync (#42) | dyurk-lila | 2026-06-26 |
| `ed1e4c1f` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-22 |
| `cb7e187a` | feat(profiler): drive torch.profiler around the training loo | dyurk-lila | 2026-06-22 |
| `b0f52733` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-22 |
| `f027fe88` | [megatron] Stream ChunkedDistributedLogprob.backward into a  | dyurk-lila | 2026-06-22 |
| `b12818bc` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-22 |
| `34886ebe` | [train] Skip building unused per-token loss_fn_outputs when  | dyurk-lila | 2026-06-22 |
| `31c72fae` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-22 |
| `01ba1cd3` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-22 |
| `64dc91d4` | [train] Save HF processor on checkpoint export for VLMs (#29 | Vu Dinh | 2026-06-22 |
| `a6357e3d` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-22 |
| `461ff450` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-22 |
| `dd8ad3fa` | [train] Async batch prefetch (double-buffering) for the SFT  | dyurk-lila | 2026-06-22 |
| `bca9e5dd` | [train] Vectorize controller-side training-batch collation ( | dyurk-lila | 2026-06-22 |
| `ff5d8dae` | [megatron] Accept dtype-string optimizer_config_kwargs (coer | dyurk-lila | 2026-06-22 |
| `8199475f` | [megatron] Fused LM-head log-prob + entropy (avoid full [*,  | dyurk-lila | 2026-06-22 |
| `3a29acc0` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-22 |
| `76c366df` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-16 |
| `a5158404` | test: move loguru deadlock regression test into skyrl-train  | Vu Dinh | 2026-06-16 |
| `2fd5b66e` | fix: prevent loguru/stdlib logging deadlock in Ray workers | Vu Dinh | 2026-06-11 |
| `f706ed79` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-15 |
| `db513d40` | ci: disable auto-trigger of SkyRL-GPU job (missing fork ANYS | dyurk-lila | 2026-06-15 |
| `414bb535` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-15 |
| `a36839b9` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-15 |
| `55849207` | Revert "ci(upstream-sync): grant workflows:write so main fas | dyurk-lila | 2026-06-15 |
| `dc9bf5d3` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-15 |
| `58c6d609` | ci(upstream-sync): grant workflows:write so main fast-forwar | dyurk-lila | 2026-06-15 |
| `b577a50a` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-10 |
| `684e4e66` | ci: also exclude heavy tx LoRA-training parity tests from CP | dyurk-lila | 2026-06-10 |
| `e4775051` | ci: exclude slow server-spawning tinker tests from CPU CI | dyurk-lila | 2026-06-10 |
| `503b2c2d` | ci: run heavy CPU test jobs on ubuntu-latest-8-core (fix OOM | dyurk-lila | 2026-06-10 |
| `cb955daf` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-06-10 |
| `186fd6cc` | feat(upstream-sync): auto-regenerate develop↔main delta on m | dyurk-lila | 2026-06-10 |
| `237a022f` | chore(upstream-sync): install sync tooling on develop (ledge | dyurk-lila | 2026-06-10 |

<details><summary>merge commits (8)</summary>

| commit | subject | author | date |
|---|---|---|---|
| `8d529ee5` | Merge pull request #35 from fl97inc/sync/develop-2026-06-22  | dyurk-lila | 2026-06-22 |
| `95829fb0` | Merge commit '7f45370462952b85443366adc3d785fb905a2d9b' in.. | lila-sync-bot | 2026-06-22 |
| `fdafc434` | Merge pull request #28 from fl97inc/vdinh/fix-loguru-loggi.. | dyurk-lila | 2026-06-16 |
| `c42b3fa9` | Merge pull request #31 from fl97inc/sync/develop-2026-06-15  | dyurk-lila | 2026-06-15 |
| `0c19095c` | Merge commit 'c4eaded781fd291b4e624be7d0c2d5843377815e' in.. | dyurk-lila | 2026-06-15 |
| `751c765c` | Merge pull request #27 from fl97inc/ci/beefier-cpu-runner    | dyurk-lila | 2026-06-10 |
| `ea2ea70b` | Merge pull request #19 from fl97inc/chore/auto-develop-delta | dyurk-lila | 2026-06-10 |
| `281b7298` | Merge pull request #18 from fl97inc/chore/install-upstream.. | dyurk-lila | 2026-06-10 |

</details>

### Files changed (`origin/main...HEAD`)

```
 .claude/docs/architecture.md                       |    3 +-
 .claude/docs/ci.md                                 |    2 +-
 .claude/docs/contributing.md                       |    2 +-
 .claude/docs/inference.md                          |   20 +-
 .claude/docs/weight_sync.md                        |   37 +-
 .github/CODEOWNERS                                 |   26 +
 .github/actionlint.yaml                            |    3 +
 .github/workflows/cpu_skyrl.yaml                   |   14 +-
 .github/workflows/cpu_skyrl_train.yaml             |    4 +-
 .github/workflows/develop-delta.yaml               |   91 +
 .github/workflows/gpu_skyrl.yaml                   |   23 +-
 .github/workflows/gpu_skyrl_train_megatron.yaml    |    4 +-
 .../workflows/gpu_skyrl_train_old_inference.yaml   |   71 -
 .github/workflows/sync-upstream.yaml               |  654 ++++
 NOTICE                                             |   21 +
 README.md                                          |    3 -
 UPSTREAM.md                                        |  577 +++
 ci/anyscale_gpu_ci.yaml                            |    4 +-
 ci/anyscale_gpu_ci_h100.yaml                       |    4 +-
 ci/anyscale_gpu_ci_skyrl_train.yaml                |    4 +-
 ci/anyscale_gpu_ci_skyrl_train_megatron.yaml       |    4 +-
 ...nyscale_gpu_ci_skyrl_train_megatron_models.yaml |    4 +-
 ci/anyscale_gpu_ci_skyrl_train_old_inference.yaml  |   10 -
 ci/anyscale_gpu_e2e_test.yaml                      |    4 +-
 ci/anyscale_gpu_e2e_test_fully_async.yaml          |    4 +-
 ci/anyscale_gpu_e2e_test_megatron.yaml             |    4 +-
 ci/anyscale_gpu_e2e_test_sft.yaml                  |    4 +-
 ci/anyscale_gpu_e2e_test_tinker.yaml               |    4 +-
 ci/anyscale_gpu_e2e_test_tinker_fully_async.yaml   |    4 +-
 ci/anyscale_tinker_skyrl_train_backend_gpu.yaml    |    4 +-
 ci/gpu_ci_run_skyrl_train.sh                       |   10 -
 ci/gpu_ci_run_skyrl_train_old_inference.sh         |   17 -
 docker/Dockerfile                                  |    2 +-
 docker/Dockerfile.megatron                         |   16 +-
 docs/api-pages.yaml                                |    6 +-
 .../docs/checkpointing-logging/vllm-metrics.mdx    |   18 -
 docs/content/docs/configuration/config.mdx         |   81 +-
 docs/content/docs/configuration/placement.mdx      |    4 +-
 docs/content/docs/examples/flash_rl.mdx            |  109 -
 docs/content/docs/examples/geometry3k.mdx          |    3 +-
 docs/content/docs/examples/megatron.mdx            |   16 +-
 docs/content/docs/examples/meta.json               |    2 +-
 docs/content/docs/examples/multi_turn_text2sql.mdx |    4 +-
 docs/content/docs/examples/ppo.mdx                 |    2 +-
 docs/content/docs/examples/quantized_rollouts.mdx  |   80 +
 docs/content/docs/examples/remote_server.mdx       |   82 +-
 docs/content/docs/examples/search.mdx              |    2 -
 docs/content/docs/examples/visgym.mdx              |    5 +-
 .../getting-started/inference_architecture.mdx     |   21 +-
 docs/content/docs/getting-started/installation.mdx |   16 +-
 docs/content/docs/harbor/index.mdx                 |    5 +-
 docs/content/docs/tinker/architecture.mdx          |    9 +-
 docs/content/docs/tutorials/agent-integration.mdx  |    7 +-
 docs/content/docs/tutorials/fully_async.mdx        |   13 +-
 docs/content/docs/tutorials/new_env.mdx            |    4 +-
 .../content/docs/tutorials/skyrl_gym_generator.mdx |    2 +-
 docs/content/docs/tutorials/vision_language_rl.mdx |    2 +-
 docs/lib/sort-tree.ts                              |    2 +-
 examples/tinker/ppo/run_tinker_server.sh           |    2 +-
 .../session_based_routing/run_tinker_server.sh     |    2 +-
 examples/train/README.md                           |    7 +-
 examples/train/algorithms/cispo/run_cispo_gsm8k.sh |    1 -
 .../algorithms/clip_cov_kl_cov/run_clip_cov.sh     |    1 -
 .../train/algorithms/clip_cov_kl_cov/run_kl_cov.sh |    1 -
 .../run_custom_adv_est.sh                          |    1 -
 .../custom_policy_loss/run_custom_policy_loss.sh   |    1 -
 .../train/algorithms/dapo/main_dapo_fully_async.py |   38 +-
 .../algorithms/dapo/run_dapo_aime_qwen3_4b_aime.sh |    3 +-
 examples/train/algorithms/dapo/run_dapo_gsm8k.sh   |    3 +-
 .../algorithms/dapo/run_dapo_qwen2.5_32b_aime.sh   |    3 +-
 .../dapo/run_dapo_qwen2.5_math_7b_aime.sh          |    3 +-
 .../algorithms/dapo/run_dapo_qwen3_1.7b_aime.sh    |    3 +-
 .../dapo/run_dapo_qwen3_1.7b_aime_fully_async.sh   |   10 +-
 ...run_dapo_qwen3_1.7b_aime_fully_async_onestep.sh |   12 +-
 .../run_dapo_qwen3_30b_a3b_lora_megatron_aime.sh   |    3 +-
 .../dapo/run_dapo_qwen3_30b_a3b_megatron_aime.sh   |    3 +-
 .../train/algorithms/drgrpo/run_drgrpo_gsm8k.sh    |    1 -
 examples/train/algorithms/maxrl/run_maxrl_gsm8k.sh |    1 -
 .../algorithms/reinforce++/run_reinforce++.sh      |    1 -
 examples/train/algorithms/rloo/run_rloo.sh         |    1 -
 .../algorithms/sapo/run_sapo_qwen3_4b_aime.sh      |    3 +-
 examples/train/async/async_run_gsm8k.sh            |    1 -
 examples/train/async/async_trainer.py              |  138 +-
 examples/train/async/main_async.py                 |    5 -
 examples/train/flash_rl/.env.0.5b_int8             |    4 -
 examples/train/flash_rl/.env.fp8                   |    2 -
 examples/train/flash_rl/.env.int8                  |    4 -
 examples/train/flash_rl/__init__.py                |    0
 examples/train/flash_rl/flash_rl_engine.py         |  135 -
 examples/train/flash_rl/main_dapo_flashrl.py       |  163 -
 .../flash_rl/run_dapo_gsm8k_flashrl_0.5b_fp8.sh    |   96 -
 .../flash_rl/run_dapo_gsm8k_flashrl_0.5b_int8.sh   |   96 -
 .../flash_rl/run_dapo_gsm8k_flashrl_32b_int8.sh    |   97 -
 .../flash_rl/run_dapo_repro_flashrl_0.5b_int8.sh   |   93 -
 .../flash_rl/run_dapo_repro_flashrl_32b_int8.sh    |   99 -
 .../train/fully_async/fully_async_run_gsm8k.sh     |    7 +-
 .../fully_async_run_gsm8k_megatron_lora.sh         |    6 +-
 examples/train/fully_async/main_fully_async.py     |    5 -
 examples/train/fully_async/main_fully_async_sim.py |    5 -
 .../sim_trainer/run_fully_async_sim_gsm8k_e2e.sh   |    6 +-
 .../run_fully_async_sim_gsm8k_external.sh          |    6 +-
 examples/train/geometry3k/run_geometry3k.sh        |    3 +-
 examples/train/geometry3k/run_geometry3k_lora.sh   |    3 +-
 examples/train/gptoss/run_gsm8k_gptoss.sh          |    4 +-
 examples/train/gsm8k/gsm8k-grpo-skypilot.yaml      |    1 -
 examples/train/gsm8k/run_32b_gsm8k.sh              |    1 -
 examples/train/gsm8k/run_gsm8k.sh                  |    1 -
 examples/train/gsm8k/run_gsm8k_modal.sh            |    1 -
 examples/train/gsm8k/run_gsm8k_pd.sh               |    1 -
 examples/train/livecodebench/run_lcb.sh            |    1 -
 examples/train/llm_as_a_judge/run_llm_judge.sh     |    1 -
 .../train/lora/run_qwen2_5_0.5b_gsm8k_grpo_lora.sh |    1 -
 .../train/lora/run_qwen2_5_0.5b_gsm8k_ppo_lora.sh  |    1 -
 examples/train/megatron/run_fsdp_baseline.sh       |    1 -
 examples/train/megatron/run_megatron.sh            |   11 +-
 .../megatron/run_megatron_dapo_qwen3.5_35b_a3b.sh  |    4 +-
 .../run_megatron_dapo_qwen3.6_35b_a3b_lora.sh      |    2 -
 ..._megatron_dapo_qwen3.6_35b_a3b_lora_int4_qat.sh |  180 +
 .../run_megatron_dapo_qwen3_235b_a22b_lora.sh      |    1 -
 .../megatron/run_megatron_dapo_qwen3_30b_a3b.sh    |    3 +-
 .../run_megatron_dapo_qwen3_30b_a3b_lora.sh        |    3 +-
 .../train/megatron/run_megatron_dapo_qwen3_4b.sh   |    3 +-
 .../megatron/run_megatron_dapo_qwen3_4b_lora.sh    |    3 +-
 .../train/megatron/run_megatron_grpo_glm4_7_30b.sh |    5 +-
 .../train/megatron/run_megatron_lora_qwen3-0.6b.sh |    1 -
 .../megatron/run_megatron_lora_qwen3-30b-a3b.sh    |    1 -
 examples/train/megatron/run_megatron_moonlight.sh  |    1 -
 .../megatron/run_megatron_nemotron_mini_4b.sh      |   11 +-
 .../train/megatron/run_megatron_qwen3-235b-a22b.sh |    1 -
 .../train/megatron/run_megatron_qwen3-30b-a3b.sh   |    1 -
 examples/train/megatron/run_megatron_qwen3.5.sh    |    1 -
 .../train/megatron/run_megatron_qwen3.5_35b_a3b.sh |    1 -
 examples/train/megatron/run_search_megatron.sh     |    1 -
 examples/train/mini_swe_agent/.env.miniswe         |    2 -
 .../train/mini_swe_agent/mini_swe_generator.py     |   53 +-
 examples/train/mini_swe_agent/run_mini_swe_30B.sh  |    4 -
 examples/train/mini_swe_agent/run_mini_swe_8B.sh   |    4 -
 examples/train/models/run_qwen3.5_0.8b.sh          |    1 -
 examples/train/moe/run_qwen1_5_MoE_A2_7B.sh        |    1 -
 examples/train/multiply/run_multiply.sh            |    1 -
 .../nemotron_3/run_nemotron_3_nano_4b_gsm8k.sh     |    2 -
 .../run_on_policy_distill_math_qwen3_1.7b.sh       |    3 +-
 .../run_on_policy_distill_math_qwen3_4b.sh         |    3 +-
 examples/train/ppo/run_ppo.sh                      |    1 -
 .../remote_inference_engine/run_vllm_server.sh     |   21 -
 .../run_remote.sh                                  |   16 +-
 .../remote_inference_server/run_vllm_server.sh     |   16 +
 examples/train/rlm/openrouter_client.py            |    4 +-
 examples/train/rlm/rlm_generator.py                |    5 +-
 examples/train/rlm/run_multi_paper_rlm.sh          |    1 -
 .../router_replay/run_dapo_moonlight_16b_a3b.sh    |    3 +-
 .../run_moonlight16b_router_replay.sh              |    1 -
 examples/train/search/run_search.sh                |    1 -
 examples/train/search/run_search_fully_async.sh    |    3 +-
 examples/train/sft/README.md                       |   83 +
 examples/train/sft/curriculum_sampler.py           |  130 +
 examples/train/sft/data_mixing_sampler.py          |  118 +
 examples/train/sft/prepare_cauldron_vlm.py         |   88 +
 examples/train/sft/run_sft_megatron_vlm.sh         |   63 +
 .../train/step_wise/run_skyrl_sql_step_wise.sh     |    1 -
 .../step_wise/run_skyrl_sql_step_wise_qwen3.sh     |    1 -
 examples/train/text_to_sql/run_skyrl_sql.sh        |    1 -
 .../run_skyrl_sql_conversation_format.sh           |    1 -
 examples/train/text_to_sql/run_skyrl_sql_fp8.sh    |    1 -
 .../text_to_sql/run_skyrl_sql_megatron_lora.sh     |    1 -
 examples/train/text_to_sql/run_sql_fsdp.sh         |    1 -
 examples/train/text_to_sql/run_sql_fsdp_2node.sh   |    1 -
 examples/train/thunder_agent/README.md             |    2 +-
 examples/train/thunder_agent/main_thunder_agent.py |   15 +-
 .../thunder_agent/scripts/r2egym_32b/run_sbatch.sh |    3 +-
 .../thunder_agent/scripts/r2egym_32b/run_stages.sh |    5 +-
 .../scripts/r2egym_32b/run_trainer.sh              |    5 +-
 .../thunder_agent/scripts/r2egym_32b/setup_env.sh  |    6 +-
 .../scripts/r2egym_32b/start_rollout_servers.sh    |   35 +-
 .../thunder_agent/skyrl_integration/generator.py   |    8 +-
 .../tests/test_thunder_agent_router.py             |   14 +-
 examples/train/tis_correction/run_dapo_tis.sh      |    3 +-
 examples/train/training_backends/fsdp/run_fsdp.sh  |    1 -
 .../train/training_backends/run_no_seq_pack.sh     |    1 -
 .../turn_level_rewards/run_gsm8k_multi_turn.sh     |    1 -
 examples/train/visgym/run_visgym_from_instruct.sh  |    3 +-
 examples/train/visgym/run_visgym_from_sft.sh       |    3 +-
 .../harbor/entrypoints/main_harbor_fully_async.py  |    4 -
 .../train_integrations/harbor/harbor_generator.py  |   45 +-
 .../train_integrations/harbor/run_codecontest.sh   |    4 -
 .../harbor/run_codecontest_fully_async.sh          |    6 +-
 .../train_integrations/harbor/run_harbor_gen.sh    |    4 -
 examples/train_integrations/modal/README.md        |    2 +-
 examples/train_integrations/modal/main.py          |    2 +-
 examples/train_integrations/openenv/run_openenv.sh |    1 -
 .../openreward/run_openreward.sh                   |    1 -
 .../verifiers/entrypoints/main_verifiers.py        |   11 +-
 .../train_integrations/verifiers/run_verifiers.sh  |    1 -
 .../verifiers/verifiers_generator.py               |   22 +-
 .../train_scripts/full_context/run_full_ctx.sh     |    1 -
 .../full_context/run_full_ctx_megatron.sh          |    1 -
 .../train_scripts/full_context/trainer_full_ctx.py |  143 +-
 .../launch_multiple_remote_servers.py              |  250 --
 .../train_scripts/test_new_vs_old_inference.py     |  340 --
 integrations/arctic_rl/README.md                   |  207 ++
 integrations/arctic_rl/__init__.py                 |   17 +
 integrations/arctic_rl/config.py                   |  603 ++++
 integrations/arctic_rl/entrypoint.py               |  190 +
 integrations/arctic_rl/envs/__init__.py            |   19 +
 integrations/arctic_rl/envs/bird.py                |   88 +
 integrations/arctic_rl/envs/bird_reward.py         |  299 ++
 integrations/arctic_rl/envs/preprocess_bird.py     |  748 ++++
 integrations/arctic_rl/examples/fsdp_bird_entry.py |   53 +
 .../arctic_rl/examples/run_bird_grpo_32b_32gpu.sh  |  182 +
 .../examples/run_bird_grpo_32b_32gpu_fsdp.sh       |  136 +
 .../arctic_rl/examples/run_bird_grpo_8b_32gpu.sh   |  164 +
 .../arctic_rl/examples/run_bird_grpo_smoke.sh      |  124 +
 .../arctic_rl/examples/run_gsm8k_grpo_4gpu.sh      |   94 +
 integrations/arctic_rl/generator.py                |  178 +
 integrations/arctic_rl/trainer.py                  |  853 +++++
 pyproject.toml                                     |   69 +-
 skyrl-agent/examples/run_skyrl/run_skyrl_swe.sh    |    1 -
 .../examples/run_skyrl/skyrl_web_research_hle.sh   |    1 -
 skyrl-agent/pyproject.toml                         |    2 +-
 .../integrations/skyrl_train/skyrl_train_main.py   |    6 +-
 .../integrations/skyrl_train/trainer.py            |    7 +-
 skyrl-train/README.md                              |    2 +-
 .../skyrl_train/distributed/fsdp_strategy.py       |   22 +-
 .../megatron/fused_linear_logprob_triton.py        | 1959 ++++++++++
 .../distributed/megatron/fused_lm_head.py          |   95 +
 .../distributed/megatron/megatron_utils.py         |   65 +-
 .../distributed/megatron/model_utils.py            |  528 ++-
 .../skyrl_train/distributed/megatron/optimizer.py  |    6 +-
 .../distributed/megatron/optimizer_dtype.py        |   64 +
 .../distributed/megatron/packing_utils.py          |   27 +
 skyrl/backends/skyrl_train/distributed/strategy.py |   14 +
 .../skyrl_train/inference_engines/__init__.py      |    0
 .../inference_engines/inference_engine_client.py   |  466 ---
 .../inference_engine_client_http_endpoint.py       |  363 --
 .../ray_wrapped_inference_engine.py                |  343 --
 .../inference_engines/remote_inference_engine.py   |  334 --
 .../skyrl_train/inference_engines/utils.py         |  267 --
 .../skyrl_train/inference_engines/vllm/utils.py    |   33 -
 .../inference_engines/vllm/vllm_engine.py          |  776 ----
 .../inference_engines/vllm/vllm_server.py          |  164 -
 .../base.py                                        |  107 +-
 .../skyrl_train/inference_servers/engine_utils.py  |   89 +
 .../inference_servers/layerwise_reload.py          |  113 +-
 .../inference_servers/new_inference_worker_wrap.py |   12 +-
 .../inference_servers/remote_inference_client.py   |   43 +-
 .../skyrl_train/inference_servers/server_group.py  |    6 +-
 .../skyrl_train/inference_servers/setup.py         |    7 +-
 .../skyrl_train/inference_servers/utils.py         |    4 +-
 .../inference_servers/vllm_server_actor.py         |  181 +-
 .../skyrl_train/inference_servers/vllm_worker.py   |  127 -
 skyrl/backends/skyrl_train/training_batch.py       |   17 +-
 skyrl/backends/skyrl_train/utils/io/io.py          |   34 +
 skyrl/backends/skyrl_train/utils/ppo_utils.py      |   67 +-
 skyrl/backends/skyrl_train/utils/profiler.py       |  156 +-
 skyrl/backends/skyrl_train/utils/replay_utils.py   |  258 +-
 skyrl/backends/skyrl_train/weight_sync/__init__.py |    8 -
 skyrl/backends/skyrl_train/weight_sync/base.py     |    2 +-
 .../skyrl_train/weight_sync/broadcast_strategy.py  |  258 +-
 .../skyrl_train/weight_sync/cuda_ipc_strategy.py   |  162 +-
 .../skyrl_train/weight_sync/transfer_strategy.py   |  118 +-
 .../weight_sync/weight_extractor_utils.py          |    6 +-
 .../skyrl_train/weight_sync/weight_loader.py       |   25 -
 .../skyrl_train/workers/fsdp/fsdp_worker.py        |   23 +-
 .../skyrl_train/workers/megatron/fake_int4_qat.py  |  146 +
 .../workers/megatron/megatron_model_wrapper.py     |  442 ++-
 .../workers/megatron/megatron_worker.py            |  184 +-
 .../backends/skyrl_train/workers/model_wrapper.py  |    8 +-
 skyrl/backends/skyrl_train/workers/worker.py       |  192 +-
 .../skyrl_train/workers/worker_dispatch.py         |   57 +-
 skyrl/backends/skyrl_train/workers/worker_utils.py |   26 +-
 skyrl/backends/skyrl_train_backend.py              |  234 +-
 skyrl/benchmarks/bench_fused_linear_logprob.py     |  221 ++
 skyrl/benchmarks/load_test_concurrency.py          |    1 -
 skyrl/env_vars.py                                  |   21 -
 skyrl/tinker/api.py                                |    3 +-
 skyrl/tinker/types.py                              |    3 +-
 skyrl/train/config/__init__.py                     |    4 +-
 skyrl/train/config/config.py                       |  338 +-
 skyrl/train/config/sft_config.py                   |   34 +
 skyrl/train/dataset/collators.py                   |  106 +-
 skyrl/train/dataset/preprocess.py                  |  197 +-
 skyrl/train/dataset/replay_buffer.py               |    7 +-
 skyrl/train/dataset/samplers.py                    |   84 +
 skyrl/train/entrypoints/main_base.py               |  139 +-
 skyrl/train/entrypoints/main_generate.py           |    2 +-
 skyrl/train/entrypoints/serve.py                   |   11 +-
 skyrl/train/evaluate.py                            |   32 +-
 skyrl/train/fully_async_trainer.py                 |  353 +-
 skyrl/train/generators/base.py                     |    2 +-
 skyrl/train/generators/skyrl_gym_generator.py      |   45 +-
 skyrl/train/generators/skyrl_vlm_generator.py      |    8 +-
 skyrl/train/generators/utils.py                    |   20 +-
 skyrl/train/sft_trainer.py                         |  814 +++--
 skyrl/train/trainer.py                             |  459 ++-
 skyrl/train/utils/async_batch_collator.py          |   58 +
 skyrl/train/utils/trainer_utils.py                 |   45 +-
 skyrl/train/utils/utils.py                         |   82 +-
 skyrl/train/utils/vllm_metrics_scraper.py          |  164 +-
 skyrl/utils/routed_experts.py                      |   14 +
 skyrl/utils/tok.py                                 |   32 +-
 skyrl/utils/token_metadata.py                      |  204 ++
 .../backends/skyrl_train/_fake_int4_qat_golden.py  |   36 +
 .../test_chunked_logprob_backward_streaming.py     |  116 +
 .../skyrl_train/distributed/test_fused_lm_head.py  |  228 ++
 .../distributed/test_optimizer_dtype_coercion.py   |   57 +
 .../skyrl_train/distributed/test_packing_utils.py  |   39 +
 .../distributed/test_preprocess_packed_seqs_cp.py  |   11 +-
 .../test_preprocess_packed_seqs_multiseq.py        |  144 +-
 .../distributed/test_save_hf_configs_processor.py  |  111 +
 tests/backends/skyrl_train/gpu/gpu_ci/conftest.py  |    3 +-
 .../test_inplace_lora_reload_no_pause.py           |    3 -
 .../inference_servers/test_multi_lora_serving.py   |    5 +-
 .../test_new_inference_generation.py               |    6 +-
 .../test_remote_inference_client_chat_template.py  |    4 +-
 .../test_vllm_server_entrypoint.py                 |  137 +
 .../test_vlm_inference_generation.py               |    7 -
 .../gpu_ci/inference_servers/test_weight_sync.py   |   75 +-
 .../inference_servers/test_weight_sync_moe.py      |  365 --
 .../gpu/gpu_ci/integrations/test_pd_routing.py     |    1 -
 .../megatron/test_chunked_logprob_backward.py      |    2 +-
 .../gpu/gpu_ci/megatron/test_fake_int4_qat.py      |  273 ++
 .../gpu_ci/megatron/test_fused_linear_logprob.py   |  230 ++
 .../megatron/test_fused_linear_logprob_triton.py   |  175 +
 .../gpu/gpu_ci/megatron/test_megatron_models.py    |    2 +-
 .../gpu/gpu_ci/megatron/test_megatron_vlm_init.py  |  358 ++
 .../gpu/gpu_ci/megatron/test_megatron_worker.py    |   13 +-
 .../gpu/gpu_ci/megatron/test_router_replay.py      |   21 +-
 .../gpu/gpu_ci/megatron/test_sft_packing_parity.py |   11 +-
 .../gpu/gpu_ci/test_engine_generation.py           |    9 +-
 .../gpu/gpu_ci/test_expert_parallel_inference.py   |    3 +-
 .../test_inference_engine_client_http_endpoint.py  | 1102 ------
 tests/backends/skyrl_train/gpu/gpu_ci/test_lora.py |    4 +-
 .../gpu_ci/test_pause_and_continue_generation.py   |  360 --
 .../gpu/gpu_ci/test_policy_local_engines_e2e.py    |    4 +-
 .../gpu/gpu_ci/test_save_weights_for_sampler.py    |    4 +-
 .../gpu/gpu_ci/test_skyrl_gym_generator.py         |   27 +-
 .../gpu/gpu_ci/test_skyrl_vlm_gym_generator.py     |    4 +-
 .../skyrl_train/gpu/gpu_ci/test_training_step.py   |  123 +
 .../gpu/gpu_ci/test_transfer_strategies_e2e.py     |  396 --
 .../gpu/gpu_ci/test_verifiers_generator.py         |   32 +-
 .../backends/skyrl_train/gpu/test_main_generate.py |    1 -
 .../skyrl_train/gpu/test_skyrl_gym_generator.py    |    1 -
 tests/backends/skyrl_train/gpu/utils.py            |  276 +-
 .../skyrl_train/inference_engines/__init__.py      |    0
 .../test_inference_engine_client.py                |  434 ---
 .../test_route_prompts_to_engines.py               |   89 -
 .../skyrl_train/inference_engines/vllm/__init__.py |    0
 .../skyrl_train/inference_engines/vllm/utils.py    |   24 -
 .../inference_servers/test_build_vllm_cli_args.py  |   19 +-
 .../test_remote_inference_client.py                |   30 +
 tests/backends/skyrl_train/test_fake_int4_qat.py   |  161 +
 .../skyrl_train/test_token_based_batching_utils.py |   12 +
 tests/backends/skyrl_train/test_train_batch.py     |   16 +-
 tests/backends/skyrl_train/util.py                 |    5 +-
 .../skyrl_train/utils/test_local_read_files.py     |   43 +
 tests/backends/skyrl_train/utils/test_profiler.py  |  358 ++
 .../skyrl_train/utils/test_replay_utils.py         |  206 ++
 .../weight_sync/test_remote_weight_loader.py       |  175 -
 .../weight_sync/test_transfer_strategies.py        |   17 +-
 .../workers/test_sft_loss_fn_outputs_gate.py       |  226 ++
 .../skyrl_train/workers/test_worker_utils.py       |   33 +
 .../tinker/skyrl_train/test_loss_normalization.py  |   45 +
 tests/tinker/test_api_validation.py                |   18 +
 tests/tinker/test_engine.py                        |    9 +
 tests/train/algorithms/test_losses.py              |  101 +
 tests/train/algorithms/test_skip_fwd_logprobs.py   |   30 +
 tests/train/dataset/test_preprocess.py             |   46 +-
 .../generators/test_generator_output_utils.py      |    1 +
 tests/train/generators/test_skyrl_gym_generator.py |   56 +-
 tests/train/gpu_e2e_test/gsm8k_colocate.sh         |    5 +-
 .../train/gpu_e2e_test/gsm8k_colocate_megatron.sh  |    5 +-
 tests/train/gpu_e2e_test/gsm8k_fully_async.sh      |    5 +-
 tests/train/gpu_e2e_test/gsm8k_tinker.sh           |   11 +-
 .../train/gpu_e2e_test/gsm8k_tinker_fully_async.sh |   11 +-
 .../gsm8k_official_tinker_qwen3-8b_baseline.sh     |    6 +-
 ...ker_colocated_llama-3.1-8b-instruct_baseline.sh |    6 +-
 ...r_fully_async_llama-3.1-8b-instruct_baseline.sh |    6 +-
 tests/train/gpu_e2e_test/sft_tulu3_megatron.sh     |    5 +-
 tests/train/test_async_batch_collation.py          |  300 ++
 .../test_collation_vectorization_equivalence.py    |  376 ++
 tests/train/test_config.py                         |  357 +-
 tests/train/test_packing_round_trip.py             |   57 +-
 tests/train/test_sft_callbacks.py                  |   94 +-
 tests/train/test_sft_config.py                     |   55 +
 tests/train/test_sft_dataloader.py                 |  557 +++
 tests/train/test_sft_packing_collate.py            |   70 +-
 tests/train/test_sft_tokenization.py               |   72 +-
 tests/train/test_trainer_utils.py                  |    9 +
 tests/train/test_vllm_metrics_scraper.py           |  243 ++
 tests/train/util.py                                |    6 +-
 tests/train/utils/test_logging_config.py           |  120 +
 tests/utils/test_token_metadata.py                 |   88 +
 upstream-sync/bedrock-ci-setup.md                  |  106 +
 upstream-sync/gen_develop_delta.sh                 |  104 +
 uv.lock                                            | 3768 +++++++++-----------
 395 files changed, 21501 insertions(+), 13116 deletions(-)
```
<!-- END AUTO:develop-delta -->

### Notes (LLM-enriched — optional)

<!--
  The block below is written by the OPTIONAL Claude enrichment step in
  .github/workflows/develop-delta.yaml (dormant until `vars.ENABLE_CLAUDE_DELTA == 'true'` and
  Lila Bedrock CI access is provisioned — see upstream-sync/bedrock-ci-setup.md). It paraphrases
  what each carried change does and suggests a category. The deterministic script NEVER touches
  this block, and Claude NEVER touches the tables block above — they own disjoint byte ranges.
-->

<!-- BEGIN AUTO:develop-delta-notes -->
_No LLM enrichment yet. Activates once `ENABLE_CLAUDE_DELTA` + Bedrock CI access are set up._
<!-- END AUTO:develop-delta-notes -->

## 3. Drop-list (carried work superseded by a merged upstream PR)

When an upstream equivalent merges, the carried copy is dropped at the next sync. Deterministic
input to the SOP's rebase step.

| internal commit / PR | superseded by upstream PR# | dropped on |
|---|---|---|
| #2 MoE aux loss (`c7345271`) | #1705 (`140b3f57`) | 2026-06-10 (catch-up) |
| #3 try/except + wandb (`7ab4fc15`) | #1706 (`8e1a7dab`) | 2026-06-10 |
| #8 tokens/sec ("MFU replacement", `8096c0fa`) | #1711 (`ddc1e556`) | 2026-06-10 |
| #10 GPU-util (`4b352b93`) | #1712 (`172bc17a`) | 2026-06-10 |
| #12 logprob-diff (`11016350`) | #1707 (`18f0ac7d`) | 2026-06-10 |
| #13 aux-loss-coeff (`252baa4f`) | #1708 (`1f02f866`) | 2026-06-10 |
| #11 vLLM engine-dead hang fix | #1691 (`de55355e`) | 2026-06-10 (PR closed; same author, round-tripped) |

> All 6 former fork-`main` commits were verified content-contained upstream (every added line
> present in `upstream/main`, modulo upstream's `int→float` refinement of `moe_aux_loss_coeff` in
> #1708) and dropped when `main` was reset to `f508739f`. #11 was closed as superseded by #1691.
> **#7 and #9 are deliberately NOT on this list** — their upstream equivalents are not in `main`
> (see §2).

## 4. Conflict hotspots (review with care every sync)

`skyrl/train/config/config.py`, `skyrl/train/config/sft_config.py`, `skyrl/train/trainer.py`,
`skyrl/train/sft_trainer.py`, `skyrl/train/fully_async_trainer.py` (+ `pyproject.toml` for dep
pins). Gated by `.github/CODEOWNERS`.
