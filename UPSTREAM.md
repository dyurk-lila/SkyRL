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
| `fd79ceec` | — | 2026-07-20 | `lila-sync-fd79ceec` |

> Seed value: the `upstream/main` SHA fork `main` was reset to in the one-time catch-up
> (2026-06-10). `main` was verified byte-identical to `upstream/main` (`f508739f`) — 0 ahead /
> 0 behind. Update this row every cycle (SOP step 7).

## 2. Carried patches (what diverges from a pure upstream mirror)

`main` carries **zero** commits. The delta lives entirely on topic branches that PR into
`develop`. Goal: keep this list short. `category` ∈ {`keep-fork-only`, `pending-upstream`,
`evaluate`}.

| branch | internal PR# | category | public PR# (if pending) | hotspot files touched |
|---|---|---|---|---|
| `vdinh/wandb-tags-tracking` | #14 | pending-upstream | _TODO: NovaSky PR#_ | `config.py`, `sft_config.py`, `tracking.py` |
| `feat/sync-weights-subtimers` | #15 | keep-fork-only | — | `broadcast_strategy.py`, `fully_async_trainer.py`, worker dispatch |
| `vdinh/fix-max-training-steps-and-hang` | #5 | keep-fork-only | — | `config.py`, `trainer.py` |
| `tito_example` | #6 | keep-fork-only | — | examples only |
| `achaloo/mfu` | #4 | evaluate | — | `trainer.py`, `flops_tracker.py` (true MFU %, distinct from upstream's tokens/sec) |
| `optimize-sft-tokenization` | #9 | evaluate | — | `sft_trainer.py`, `sft_config.py` |

Profiler driving (#16) and validation-sample logging (#7) were removed from the carry list after
their upstream implementations landed by `fd79ceec`.

> **`evaluate` = author decision pending.** These were *not* cleanly superseded by upstream
> (verified 2026-06-10): #4 adds a true MFU-% tracker (`flops_tracker.py`), distinct from
> upstream's `tokens_per_second_per_gpu` (#1711); #9 overlaps upstream #1695 in intent but uses a
> different chunk-worker implementation (~93/377 added lines absent
> from `main`). Authors decide: rebase onto `develop`, upstream, or drop.

## 2.5. Live `develop` ↔ `main` delta (AUTO-GENERATED)

<!--
  Everything between the BEGIN/END AUTO markers below is regenerated on every merge to `develop`
  by .github/workflows/develop-delta.yaml (script: upstream-sync/gen_develop_delta.sh).
  DO NOT edit inside the markers by hand — your changes will be overwritten on the next merge.
  §2 above (curated categories / public PR#s) is hand-maintained and is NEVER touched by the bot.
-->

<!-- BEGIN AUTO:develop-delta -->
_Last generated for `origin/main` (de1a58b4) ↔ `HEAD` (fe575101); merge-base `de1a58b4`. **Auto-generated — do not edit by hand.**_

`develop` is **93 commit(s)** ahead of `main`.

### Commits on `develop` not on `main`

| commit | subject | author | date |
|---|---|---|---|
| `239e2be4` | style: apply black 24.10.0 formatting to routing-replay guar | lila-sync-bot | 2026-07-28 |
| `538eff10` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-28 |
| `eba569bd` | fix(megatron): reject moe_router_fusion together with routin | lila-sync-bot | 2026-07-28 |
| `10156585` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-21 |
| `8cbfda48` | fix(train): filter fully masked reward groups (#70) | dyurk-lila | 2026-07-21 |
| `4417c94f` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-21 |
| `66b3d3f0` | chore(sync): merge upstream fd79ceec into develop (#69) | dyurk-lila | 2026-07-21 |
| `de3d8d61` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-20 |
| `8674de8c` | refactor(generators): assemble incremental routed expert tra | dyurk-lila | 2026-07-20 |
| `49f4f0b7` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-20 |
| `26b4faad` | refactor(inference): expose reusable remote generator (#63) | dyurk-lila | 2026-07-20 |
| `82e07b7c` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-20 |
| `112462c6` | perf(r3): limit route expansion to local layers (#58) | dyurk-lila | 2026-07-20 |
| `ddc6dc69` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-20 |
| `081166d6` | perf(r3): accelerate routed-expert JSON transport (#57) | dyurk-lila | 2026-07-20 |
| `f475788e` | docs(upstream-sync): regenerate develop↔main delta [skip ci] | lila-sync-bot | 2026-07-20 |
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

<details><summary>merge commits (13)</summary>

| commit | subject | author | date |
|---|---|---|---|
| `fe575101` | Merge pull request #72 from fl97inc/dyurk/reject-router-fu.. | dyurk-lila | 2026-07-28 |
| `dd3c1c24` | Merge branch 'develop' into dyurk/reject-router-fusion-wit.. | lila-sync-bot | 2026-07-28 |
| `8161b29a` | Merge pull request #71 from fl97inc/sync/develop-2026-07-27  | dyurk-lila | 2026-07-28 |
| `deb0438b` | chore(sync): merge upstream de1a58b4 into develop            | lila-sync-bot | 2026-07-27 |
| `2e8aa7ff` | chore(sync): record upstream fd79ceec ancestry (squash-mer.. | lila-sync-bot | 2026-07-27 |
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
 .github/CODEOWNERS                                 |  26 +
 .github/actionlint.yaml                            |   3 +
 .github/workflows/cpu_skyrl.yaml                   |  14 +-
 .github/workflows/cpu_skyrl_train.yaml             |   4 +-
 .github/workflows/develop-delta.yaml               |  91 +++
 .github/workflows/gpu_skyrl.yaml                   |  23 +-
 .github/workflows/sync-upstream.yaml               | 654 +++++++++++++++++++++
 NOTICE                                             |  21 +
 UPSTREAM.md                                        | 276 +++++++++
 examples/train/sft/data_mixing_sampler.py          | 118 ++++
 pyproject.toml                                     |   2 +
 .../distributed/megatron/fused_lm_head.py          |  95 +++
 .../distributed/megatron/model_utils.py            |  21 +-
 .../backends/skyrl_train/inference_servers/base.py |   5 +-
 .../inference_servers/remote_inference_client.py   | 318 ++++++----
 .../inference_servers/routed_experts_wire.py       |  45 ++
 .../inference_servers/vllm_server_actor.py         |  21 +-
 skyrl/backends/skyrl_train/training_batch.py       |  17 +-
 skyrl/backends/skyrl_train/utils/ppo_utils.py      |  53 +-
 skyrl/backends/skyrl_train/utils/replay_utils.py   | 333 +++++------
 .../workers/megatron/megatron_model_wrapper.py     | 225 +++----
 .../workers/megatron/megatron_worker.py            |  30 +-
 skyrl/backends/skyrl_train/workers/worker.py       | 148 +++--
 skyrl/backends/skyrl_train/workers/worker_utils.py |  26 +-
 skyrl/train/config/config.py                       |  28 +-
 skyrl/train/dataset/collators.py                   |  80 ++-
 skyrl/train/dataset/preprocess.py                  | 214 +++++--
 skyrl/train/dataset/replay_buffer.py               |   7 +-
 skyrl/train/evaluate.py                            |   4 +-
 skyrl/train/fully_async_trainer.py                 |   8 +-
 skyrl/train/generators/base.py                     |   3 +-
 skyrl/train/generators/skyrl_gym_generator.py      |  88 +--
 skyrl/train/generators/utils.py                    |  17 +-
 skyrl/train/sft_trainer.py                         |  53 +-
 skyrl/train/trainer.py                             |  29 +-
 skyrl/train/utils/trainer_utils.py                 |  42 +-
 skyrl/train/utils/utils.py                         |  17 +-
 skyrl/utils/routed_experts.py                      | 102 ++++
 skyrl/utils/token_metadata.py                      | 253 ++++++++
 .../test_chunked_logprob_backward_streaming.py     | 116 ++++
 .../skyrl_train/distributed/test_fused_lm_head.py  | 228 +++++++
 .../megatron/test_chunked_logprob_backward.py      |   2 +-
 .../gpu/gpu_ci/megatron/test_router_replay.py      |  19 +-
 .../skyrl_train/gpu/gpu_ci/test_training_step.py   | 123 ++++
 .../test_remote_inference_client.py                |  63 +-
 .../inference_servers/test_routed_experts_wire.py  |  92 +++
 .../skyrl_train/test_token_based_batching_utils.py |  12 +
 tests/backends/skyrl_train/test_train_batch.py     |  16 +-
 .../skyrl_train/utils/test_replay_utils.py         | 229 ++++++++
 .../workers/test_sft_loss_fn_outputs_gate.py       | 228 +++++++
 .../skyrl_train/workers/test_worker_utils.py       |  33 ++
 tests/train/algorithms/test_losses.py              | 101 ++++
 tests/train/algorithms/test_skip_fwd_logprobs.py   |  30 +
 tests/train/dataset/test_preprocess.py             | 130 +++-
 tests/train/generators/test_datatypes.py           |   1 -
 .../generators/test_generator_output_utils.py      |   2 +-
 tests/train/generators/test_skyrl_gym_generator.py |  76 ++-
 .../test_collation_vectorization_equivalence.py    | 376 ++++++++++++
 tests/train/test_config.py                         |   8 +
 tests/train/test_sft_callbacks.py                  |  94 ++-
 tests/train/test_trainer_utils.py                  |  21 +
 tests/train/utils/test_logging_config.py           | 120 ++++
 tests/utils/test_token_metadata.py                 | 151 +++++
 upstream-sync/bedrock-ci-setup.md                  | 106 ++++
 upstream-sync/gen_develop_delta.sh                 | 104 ++++
 uv.lock                                            |  10 +
 66 files changed, 5191 insertions(+), 814 deletions(-)
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
