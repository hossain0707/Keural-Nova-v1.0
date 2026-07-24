# Keural MoE Pro — v2 Training Plan

Planning doc for the next fine-tune, written from the **measured results of v1.0** (not
theory). Goal: keep everything Qwen 3.6 base can do, fix v1.0's regressions, and improve the
things SFT can actually improve. Pairs with `VERIFIED-FACTS.md` (ground truth) and
`DATASET-REQUIREMENTS.md` (v1 data spec).

---

## 1. What v1.0 actually did (the evidence this plan is built on)

Measured, same harness for base and fine-tune (lm-eval 0.4.12, vLLM, seed 42):

| Benchmark | Base | v1.0 | Δ | Read |
|---|---:|---:|---:|---|
| KoBEST (ko) | 62.88 | 68.82 | **+5.94** | real Korean gain ✅ |
| HAE-RAE (ko) | 75.53 | 73.42 | −2.11 | knowledge regression ⚠️ |
| KMMLU (ko) | 64.46 | 62.52 | −1.94 | knowledge regression ⚠️ |
| MMLU (en) | 83.83 | 82.07 | −1.76 | English/general regression ⚠️ |
| GSM8K | 32.15¹ | 87.11 | +54.96¹ | mostly answer-format ✅ |
| HumanEval | 62.20 | 20.73 | **−41.47** | code collapse 🔴 |

¹ base GSM8K depressed by answer-format mismatch under strict-match; the gain is largely
format compliance, not new reasoning.

Plus three qualitative findings from live testing:
- **Too-short answers** — v1.0 defaults to terse replies (learned from short training targets;
  it *can* write long when explicitly asked — proven — so it's a style bias, not inability).
- **Identity bleed** — says "made by OpenAI / GPT-4" (contamination from GPT-generated English
  SFT data).
- **Language skew** — answers Korean even to English prompts (heavy Korean data mix).

Root cause of the code collapse: **85.5% of raw magicoder rows in upstream
`mkd-chanwoo/keural-SFT` have corrupted indentation** — the model learned broken Python. Not a
pipeline/merge/eval bug.

---

## 2. Honest goal for v2 (what SFT can and cannot do)

SFT teaches **format, style, task-following** — it does **not** add knowledge or raw
intelligence. Knowledge tested by KMMLU/MMLU/HAE-RAE already lives in the Qwen base.

| Axis | Realistic v2 target |
|---|---|
| Korean fluency / naturalness / format / persona | **Beat base** (real, defensible edge) |
| KoBEST-style conversational Korean | **Beat base** (v1.0 already did) |
| Answer length / detail | **Beat base's default** via data balance (but "long ≠ better" — aim *appropriate*, not padded) |
| GSM8K / answer formatting | **Beat base** (format compliance) |
| Code (HumanEval) | **Recover to base level** (fix the data) |
| KMMLU / MMLU / HAE-RAE knowledge | **Match base (±noise)** — SFT can't exceed; format-matched MCQ data can nudge, not leap |
| English / general | **Hold flat (±noise)** — minimize forgetting + **measure it** |
| Context length (262,144 tokens) | **Unchanged, guaranteed** (architectural; SFT doesn't touch RoPE) |

> The realistic win is **"a Qwen-level brain that speaks Korean better, formats better, writes
> appropriately long, and behaves like your product"** — not "a smarter model in absolute
> terms." Aiming beyond that from SFT alone sets up disappointment.

**Two things that are NOT guaranteed and must be verified, not assumed:**
1. English/general "loses nothing" — SFT always risks some forgetting; v1.0 proved it can
   (MMLU −1.8, HumanEval collapse). v2 is *designed* to hold it flat, but you confirm with the
   eval gate before shipping.
2. Long-context reliability — the token *limit* stays 262,144, but training is on 8,192-token
   sequences, so behavior at very long context is untested unless you test it.

---

## 3. Base model decision

**Fine-tune v2 fresh from `Qwen/Qwen3.6-35B-A3B` (the original base) — NOT from v1.0 or v1.1.**

Re-fine-tuning v1.0/v1.1 compounds their baked-in damage (crippled code, dropped knowledge,
length/identity/language bias). The base is the clean, maximum-capability starting point; every
SFT run should start there so you only add, never fight prior scars. (Exception: a future
DPO/preference stage *would* build on the best SFT checkpoint — but that's a different stage,
not another round of SFT.)

---

## 4. Dataset mix (~150k examples, 1 epoch)

Quality and balance over raw volume. ~150k *clean, balanced, format-matched* beats v1.0's
~250k noisy for this goal.

| Category | Purpose | Amount | Sources |
|---|---|---:|---|
| Clean Korean conversation/instruction | Korean fluency, KoBEST (your strength) | ~50k | `mkd-chanwoo/keural-conversation-chatml-ko`, `keural-rag-chatml-ko`; `Mkd-Yonas/keural-SFT-chatml-ko-v1` (kullm_v2, kocommercial, korean_safe_conversation); `nlpai-lab/kullm-v2`, `beomi/KoAlpaca-v1.1a` |
| **Korean MCQ / knowledge** (KMMLU answer format) | **Lift** Korean knowledge scores — the key lever | ~15k | `HAERAE-HUB/KMMLU` **train/dev split ONLY** + Korean knowledge sets |
| English MCQ / knowledge (MMLU format) | Preserve MMLU | ~10k | `cais/mmlu` **`auxiliary_train` split ONLY**, `allenai/ai2_arc`, `openbookqa` |
| Math CoT (ko+en) | GSM8K + reasoning, `#### answer` format | ~20k | `openai/gsm8k` train, `meta-math/MetaMathQA`, `TIGER-Lab/MathInstruct`, `orca_math_korean` |
| **Code — indentation-verified clean** | Fix HumanEval | ~25k | `ise-uiuc/Magicoder-Evol-Instruct-110K`, `ise-uiuc/Magicoder-OSS-Instruct-75K`, `m-a-p/CodeFeedback-Filtered-Instruction` — **from ORIGINAL upstream, not chanwoo's copy** |
| General English replay (diverse) | Anti-forgetting insurance | ~25k | `HuggingFaceH4/ultrachat_200k`, `Open-Orca/OpenOrca`, `teknium/OpenHermes-2.5` |
| Long-form Korean answers | Fix v1.0's short-answer bias | ~5k | Filter existing Korean data for long, high-quality targets (no new source) |
| Identity / persona | Fix identity bleed | ~1–2k | **Hand-authored** — ~300 seed Q&A, paraphrased up |

Language balance stays Korean-majority (~55–60%) but with real format/knowledge/code coverage.

---

## 5. Three non-negotiable data rules

1. **🔴 Test-split hygiene.** For KMMLU / MMLU / GSM8K, train **only** on `train` /
   `auxiliary_train` / `dev` splits — **never** the `test` split the eval harness scores on.
   Training on test = fake benchmark numbers. `build_dataset.py` must hard-exclude test splits.
2. **🔴 Clean code from upstream + validate.** Pull code from the original `ise-uiuc/Magicoder-*`
   repos (correctly formatted), then programmatically reject any Python block that doesn't parse
   / has collapsed indentation, before training.
3. **License discipline.** Keep the commercial-clean policy; verify each source (most are
   MIT/Apache/CC BY, but confirm — exclude any non-commercial sets).

Plus two data-shaping rules:
4. **Balance response length** — include plenty of long, structured targets so the default
   isn't one-liners (but keep them genuinely detailed, not padded — long ≠ better).
5. **Strip GPT-identity contamination** — remove "I'm made by OpenAI / I'm GPT-4" responses
   from English data.

---

## 6. Recipe changes (cheap, high-impact — matter as much as the data)

| Setting | v1.0 | v2 | Why |
|---|---|---|---|
| Base | Qwen 3.6 | **Qwen 3.6 (fresh)** | never re-train on v1.0/v1.1 |
| Epochs | 2 | **1** | epoch 2 gave zero eval-loss gain, only added forgetting risk |
| Learning rate | 1e-4 | **~5e-5** | directly reduces catastrophic forgetting |
| Dataset | ~250k noisy | **~150k clean/balanced** | quality > volume for SFT on a strong base |
| LoRA r/α, dropout, targeting | r32/α64/0.0, all 40 layers incl. routed experts | **keep** | verification gate + `use_hf: true` already correct |

Everything else in the pipeline stays: MoE-aware LoRA targeting, router/expert verification
gates, `use_hf: true`, CPU merge with MTP re-injection, merge-before-serving.

---

## 7. Verification before shipping v2 (don't assume — measure)

1. Full baseline already exists (`results/base/20260710T045822Z`).
2. After v2 train + merge → run `run_eval.py --tag keural-v2` → `compare_evals.py`.
3. Ship only if: **no general-suite regression >1pt** (the gate enforces this), KMMLU/MMLU/
   HAE-RAE within ±noise of base, KoBEST up, HumanEval recovered near base, GSM8K up.
4. Live-test length, identity, and English-prompt behavior in the `test model/` chat tools.

---

## 8. Division of labor

- **You / `build_dataset.py`:** English MCQ (auxiliary_train), math, replay, clean-code pulls —
  standard HF downloads; add as new sources with license filter + **test-split exclusion** +
  **indentation validation**.
- **chanwoo (data teammate):** (a) regenerate Korean code data *without* the indentation bug;
  (b) build the **Korean MCQ-format set** from KMMLU-train + Korean knowledge (highest-value
  new asset, needs Korean-language care).
- **You (small, high-impact):** hand-author the ~300 identity seed examples — only you can
  define Keural's persona.

---

## 9. One-line summary

Fresh SFT from Qwen 3.6 base · ~150k clean/balanced/format-matched examples · 1 epoch · LR 5e-5
· clean code from upstream · MCQ-format data as the knowledge lever · test-split excluded ·
length balanced · identity fixed · then **measure** English/knowledge held flat before shipping.
Context length stays 262,144 tokens, guaranteed.
