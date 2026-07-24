# Keural SFT Pipeline — Qwen3.6-35B-A3B → Keural Nova (LoRA, ms-swift, 2× H200)

Production pipeline implementing `keural-sft-training-guide.md` (the runbook) with every
architecture/data/tooling claim re-verified hands-on — see **`VERIFIED-FACTS.md`** for the
ground truth, including corrections where the guide was truncated or outdated.

## Project outcome (2026-07-16)

The full pipeline ran end-to-end and shipped two public models:

- **[mkd-hossain/Keural-Nova-v1.0](https://huggingface.co/mkd-hossain/Keural-Nova-v1.0)** — checkpoint-7500 (~1 epoch, best held-out eval loss)
- **[mkd-hossain/Keural-Nova-v1.1](https://huggingface.co/mkd-hossain/Keural-Nova-v1.1)** — checkpoint-15846 (2 full epochs, final)

| Benchmark | Base | v1.0 | v1.1 |
|---|---:|---:|---:|
| KoBEST (ko) | 62.88 | **68.82** | 67.29 |
| HAE-RAE (ko) | **75.53** | 73.42 | 74.52 |
| KMMLU (ko) | **64.46** | 62.52 | 62.38 |
| MMLU | **83.83** | 82.07 | 82.37 |
| GSM8K (strict) | 32.15¹ | 87.11 | 86.88 |
| HumanEval | **62.20** | 20.73 | 11.59² |

¹ base GSM8K depressed by answer-format mismatch; delta is mostly format compliance.
² known regression: a large share of code rows in upstream `keural-SFT` have corrupted
indentation at the source (85.5% of raw magicoder rows) — the model learned broken
formatting. Fix planned for v2 (filter/repair code rows, retrain). Full details in the model
cards and `VERIFIED-FACTS.md`.

Training: LoRA r32/α64 on all 40 layers — attention, Gated-DeltaNet linear attention,
shared experts, and all 256 routed experts/layer via PEFT `target_parameters`; router and
gates verified frozen pre/post training. 253,528 examples (~85% ko / 15% en replay),
2 epochs, 68 h on 2× H200 (DeepSpeed ZeRO-2, bf16).

Notable bugs found & fixed along the way (documented in `VERIFIED-FACTS.md`): peft
`ParamWrapper` crashes with `lora_dropout != 0`; transformers drops `mtp.*` weights on load
(merge re-injects them); flashinfer's CUDA JIT breaks on paths containing spaces (venv
physically relocated to `/home/work/.keural_venv`); ms-swift 4.4.0 renamed
`train_type`→`tuner_type` and defaults to ModelScope hub (`use_hf: true` required); vLLM
cannot serve `target_parameters` adapters (merge before serving).

## v2 — in development

v2 targets the known v1 weaknesses (chiefly the **HumanEval code regression** — a large share
of upstream `keural-SFT` code rows have corrupted indentation at the source) and expands/cleans
the Korean mix. It reuses the exact same pipeline; only the **data** and a few hyperparameters
change.

- **Dataset prep:** [`V2-DATASET-PREP.md`](V2-DATASET-PREP.md) — the plan; built by
  [`scripts/build_dataset_v2.py`](scripts/build_dataset_v2.py) → `data/processed_v2/`
  (146,274 rows). Key upgrades over v1: **code AST-validated** (drop/repair broken-indent
  rows), **benchmark test-split excluded** (no contamination), **identity de-contaminated**,
  balanced ko-conversation / clean-code / replay / math / MCQ / long-form / identity.
- **Training plan:** [`V2-TRAINING-PLAN.md`](V2-TRAINING-PLAN.md) — 1 epoch, lower LR to
  protect base capability, same LoRA targeting + safety gate as v1.
- **Client brief & writeups:** [`CLIENT-BRIEF-Keural.md`](CLIENT-BRIEF-Keural.md),
  and the launch articles under [`blog/`](blog/) (v1.0 and v1.1, with benchmark charts).

## Serving (gateway)

[`gateway/`](gateway/) is the OpenAI-compatible serving layer used to expose Keural Nova to
coworkers: a small FastAPI gateway (`gateway.py`) in front of vLLM, plus launch scripts
(`serve_v1_hct.sh` matches the production HCT config: TP=2, YaRN 1M context,
Image-Text-to-Text). See [`gateway/GATEWAY.md`](gateway/GATEWAY.md). The gateway API key lives
in `gateway/.env` (**never committed**). [`test model/`](test%20model/) holds simple chat/vision
test clients.

> **Read this first**
> - The GPUs on this box are shared. Every GPU stage refuses to start while another job
>   occupies them (do **not** kill that job; wait or coordinate).
> - The runbook file is truncated at line 146; hyperparameters marked `[RECONSTRUCTED]` in
>   `configs/train_*.yaml` are starting points to validate with the pilot run, per the
>   guide's own advice.
> - Everything runs from the project venv with `PYTHONPATH` cleared — the box exports a
>   global `PYTHONPATH` that leaks a *shared* site-packages into venvs (see VERIFIED-FACTS §1).

## Setup (every shell)

```bash
cd "/home/work/keural MoE Pro"        # NB: path contains a space — always quote
source venv/bin/activate
export PYTHONPATH=
```

The venv is already fully provisioned (torch 2.11.0+cu129, ms-swift 4.4.0, transformers
5.12.1, peft 0.19.1, vllm 0.24.0+cu129, lm_eval 0.4.12, deepspeed 0.19.2 — cu129 builds are
**required** on this driver; PyPI defaults are CUDA-13 and will not run. See VERIFIED-FACTS §1).

One-time: `hf auth login` with a token that has write access to the publish repo
(`mkd-hossain/…`, see `configs/publish.yaml`).

## Stages — run in order

| # | Stage | Command | GPU? |
|---|-------|---------|------|
| 0 | Environment check | `python scripts/env_check.py` (add `--gpu` once GPUs are free) | opt |
| 1 | Build dataset | `python scripts/build_dataset.py --config configs/dataset.yaml` | no |
| 2 | Baseline eval (stock model) | `python scripts/run_eval.py --model base --tag base` | yes |
| 3 | Verify LoRA targeting (gate) | `python scripts/verify_lora_targets.py --pre --train-config configs/train_pilot.yaml` | no |
| 4 | Pilot train (~5k) | `python scripts/train.py --config configs/train_pilot.yaml` | yes |
| 5 | Pilot eval + compare | `run_eval.py --model <pilot merged or adapter> --tag pilot` → `compare_evals.py --baseline … --candidate …` | yes |
| 6 | Full train (~157k) | `python scripts/train.py --config configs/train_full.yaml` | yes |
| 7 | Post-training eval | `python scripts/run_eval.py --model outputs/merged/Keural-35B-A3B --tag keural` | yes |
| 8 | Merge LoRA → full weights | `python scripts/merge_lora.py --checkpoint <ckpt dir>` | no (CPU, ~160 GB RAM) |
| 9 | Compare vs baseline | `python scripts/compare_evals.py --tag base --tag2 keural` | no |
| 10 | Model card + publish | `python scripts/make_model_card.py` → `python scripts/publish.py` | no |

Notes:
- **Stage 1** downloads ~1–2 GB of datasets on first run (keural-SFT is already cached);
  writes `data/processed/{train,eval,pilot}.jsonl` + a manifest with pinned revisions,
  license table, and content hashes. Idempotent: rerun = no-op unless config changed
  (`--force` to rebuild).
- **Stage 3 is also run automatically by `train.py` before every launch** and the same
  assertions re-run against the first saved checkpoint. It fails loudly if the router
  (`mlp.gate`) or `shared_expert_gate` would be trained, or the routed experts
  (`mlp.experts.gate_up_proj/down_proj`, fused `nn.Parameter`s) would NOT be — the silent
  failure this whole pipeline is designed to prevent.
- **Training** is resumable: `python scripts/train.py --config … --resume` picks up the
  newest valid checkpoint. Run it inside `tmux`/`nohup` for long jobs. Logs under `logs/`,
  tensorboard under the run dir.
- **Eval** results are versioned under `results/<tag>/<timestamp>/`; nothing is overwritten.
  `--smoke` runs a capped-size sanity pass. HumanEval executes generated code — the harness
  sets the sandbox flags for you.
- **Merge (stage 8) is CPU-only by design** (works while GPUs are busy) and re-injects the
  MTP tensors that transformers drops on load (VERIFIED-FACTS §0.4), so the published
  model keeps the full base tensor set (vision tower + MTP) with LoRA deltas applied.
- **Publish** uploads to `mkd-hossain/Keural-35B-A3B` (private by default) with a model card
  auto-populated from the dataset manifest (incl. license table and AI Hub exclusions) and
  the eval comparison.

## Domain eval (action item)

`evals/domain/` is a placeholder — the guide (§1) calls for 50–100 real examples from the
actual use case. Drop `*.jsonl` files there (schema in `evals/domain/README.md`) and rerun
stages 2/7; until then the domain suite reports "skipped".

## Licensing (from the guide §2, enforced in stage 1)

Included sources are commercially clean (CC BY 4.0 / MIT / Apache-2.0 / CC BY-SA 4.0).
Excluded always: AI Hub-derived rows (`aihub_multisession_sci`, `aihub_multisession_social`,
`aihub_expert_qa`, `mkd-chanwoo/aihub_conversation`) — non-commercial research terms —
plus a belt-and-braces license-string filter. **Open action item:** confirm with chanwoo
whether the org's AI Hub registration permits commercial use before relaxing this.

## Layout

```
configs/            pipeline.yaml (shared), dataset.yaml, train_pilot.yaml,
                    train_full.yaml, eval.yaml, publish.yaml
scripts/            common.py + one script per stage; build_dataset_v2.py for v2
gateway/            OpenAI-compatible serving layer (FastAPI + vLLM launch scripts)
test model/         simple chat / vision test clients
blog/               launch writeups (v1.0, v1.1) + benchmark charts
evals/domain/       your domain eval set (placeholder)
results/            versioned eval results + comparisons (v1 numbers included)
data/               raw + processed datasets (created at runtime — gitignored)
outputs/            checkpoints + merged model (created at runtime — gitignored)
logs/               per-stage logs (gitignored)
VERIFIED-FACTS.md   hands-on-verified ground truth + guide corrections  ← start here
V2-DATASET-PREP.md  V2-TRAINING-PLAN.md   the v2 plan
```

> **Note:** *Keural Orbit* (a separate Gemma-4 fine-tune project) lives in its own repo and is
> intentionally excluded here — this repository is **Keural Nova (v1 + v2)** only.
