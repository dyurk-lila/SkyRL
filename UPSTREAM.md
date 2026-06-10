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
