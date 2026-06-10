# Bedrock CI setup — activating Claude Code enrichment in `develop-delta.yaml`

> **Status:** runbook for an admin / infra. The deterministic half of `develop-delta.yaml` works
> today with **no** AWS access. This doc is only needed to switch on the **optional Claude
> enrichment** step (the prose-summary / auto-categorization in the `develop-delta-notes` block).
>
> **Verified 2026-06-10** by reading sibling Lila repos (`ml.train`, `ml.agent-harness`,
> `lila-actions`) and Lila Notion/Slack. Values may drift — see "Freshness caveat" at the end.

## TL;DR — use the blessed Lila path, not a hand-rolled one

Lila does **not** wire `anthropics/claude-code-action@v1` + `aws-actions/configure-aws-credentials`
per-repo. Every Lila repo that runs Claude Code in CI uses the internal composite action
**`fl97inc/lila-actions/claude-code@v2`**, which wraps the Anthropic action and configures Bedrock
(SigV4, region `us-east-1`) internally. Canonical reference workflow: `pr_review_claude_code.yml`
(shipped by `fl97inc/python-library-template`); a known-good example is `fl97inc/lila-backends#526`.

GitHub-hosted runners (`ubuntu-latest`) **cannot reach Lila's private Bedrock.** Claude-Code CI
runs on **self-hosted ARC runners** in-VPC; the IAM role name reflects this
(`solo-github-arc-gha-claude-code-bedrock`). So enabling enrichment is mostly an *access-grant*
task, not a write-some-YAML task.

## Facts (copy these into the workflow)

| Thing | Value |
|---|---|
| Reusable action | `fl97inc/lila-actions/claude-code@v2` (floating `@v2`; **do not** pin `@v2.9.3` — broke with a 403 SigV4 error) |
| AWS account | `535002886782` (`infra-dev` / "Administrating-Lila"; where internal Bedrock lives) |
| IAM role ARN | `arn:aws:iam::535002886782:role/solo-github-arc-gha-claude-code-bedrock` |
| Region | `us-east-1` (all Anthropic models) |
| Model IDs | `us.anthropic.claude-sonnet-4-6` (default reviews), `us.anthropic.claude-opus-4-7` (heavy), `us.anthropic.claude-haiku-4-5-20251001-v1:0` (cheap) — `us.` cross-region inference profiles |
| Runner | self-hosted Bedrock-capable ARC runner (NOT `ubuntu-latest`); the role is `solo-github-arc-*` |
| GitHub App token | mint via `actions/create-github-app-token` with `vars.LILA_BOT_APP_ID` + `secrets.LILA_BOT_NON_BASE64_PRIVATE_KEY` (already used by `ml.train`, `ml.agent-harness`) |
| Env toggles | `CLAUDE_CODE_USE_BEDROCK=1`, `AWS_REGION=us-east-1` |

## Activation steps (admin / infra)

1. **Grant `fl97inc/SkyRL` access to the private `fl97inc/lila-actions` action.** Without this the
   job fails with `Unable to resolve action 'fl97inc/lila-actions, not found'`. Fix: email
   **devsecops@lila.ai** asking to enable `lila-actions` access for `fl97inc/SkyRL` (this is the
   documented process — Rory Schadler, #sw-team-all 2026-05-21).
2. **Confirm Bedrock role/runner authorization.** Ping **#sw-team-infra** (owner: **Ritchie
   Lincoln, `rlincoln@lila.ai`**, author of RFC 87) to confirm `fl97inc/SkyRL`'s Actions may assume
   `solo-github-arc-gha-claude-code-bedrock` on the solo ARC runners. The OIDC trust is managed
   centrally in `fl97inc/infra-bootstrap` (terragrunt `aws/infra-dev/base`) by infra (Sebastian
   Morawiec / William Da Palma) — you do **not** author a per-repo `sub` condition yourself. (If
   one is ever required, it'd be `repo:fl97inc/SkyRL:*` — but confirm, don't assume.)
3. **Provide the bot App credentials** to the repo if not already inherited from the org:
   `vars.LILA_BOT_APP_ID` and `secrets.LILA_BOT_NON_BASE64_PRIVATE_KEY`.
4. **Set the feature flag:** repo variable `ENABLE_CLAUDE_DELTA=true`
   (`gh variable set ENABLE_CLAUDE_DELTA --repo fl97inc/SkyRL --body true`).
5. **Uncomment** the `Claude enrichment` step in `.github/workflows/develop-delta.yaml` and change
   the job's `runs-on` to the self-hosted Bedrock runner label infra gives you (e.g.
   `solo-unprivileged-1cpu-4g`, as `ml.agent-harness` uses). Add `id-token: write` to the job
   `permissions` if infra confirms OIDC role assumption (vs instance-role) is in play.

## Freshness caveat

RFC 87 (approved, tracked in DEX-168) plans to migrate internal Bedrock off `infra-dev`/
`535002886782` to dedicated `bedrock-internal-prod` accounts using `global.anthropic.*` inference
profiles. If that has landed by setup time, the **account ID, role ARN, and model-id prefix may
change** — confirm current values with Ritchie Lincoln / #sw-team-infra before wiring.

## Why this is optional

The deterministic `gen_develop_delta.sh` already produces a complete, always-correct list of every
commit and file on `develop` not on `main`. Claude only adds prose + suggested categories on top,
in a separate block. If Bedrock CI is never provisioned, the feature still fully satisfies the
original ask ("a clear list of everything on develop that's not on main") — you just don't get the
LLM-written summaries.
