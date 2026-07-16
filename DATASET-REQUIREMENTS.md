# Keural SFT — Dataset Collection & Processing Spec

For: whoever is sourcing/curating training data for the Keural fine-tune (Qwen3.6-35B-A3B →
Keural, LoRA SFT). This document is self-contained — you don't need to read the training code
to use it. It covers: what data we're using today, what's still needed from you, the exact
format required, and the licensing rules that decide what's in/out.

---

## 1. Summary

We're building a ~157,000-example SFT dataset, bilingual (Korean-majority, English replay),
already mostly sourced from existing `mkd-chanwoo` datasets on Hugging Face. What's still
open:

| # | What we need | Status |
|---|---|---|
| 1 | 15,000–20,000 company/domain prompt+response pairs | **Not yet provided — see §5** |
| 2 | 50–100 graded domain examples for evaluation | **Not yet provided — see §6** |
| 3 | Confirmation on AI Hub license terms | **Open question — see §4** |

Everything else (the four public/company `mkd-chanwoo` sources) is already pinned, filtered,
and pipeline-ready.

---

## 2. Target composition

| Language | Share | Target count |
|---|---:|---:|
| Korean | ~80% | ~145,000–150,000 |
| English / general replay | ~20% | ~27,000 |
| **Total** | | **~157,000** |

"Replay" data (English general/math/code) is included on purpose — it's a regression
tripwire. Without it, fine-tuning on Korean data risks degrading the model's existing English
and general-reasoning ability (catastrophic forgetting). Don't reduce this share without
re-running the general benchmarks (MMLU, GSM8K, HumanEval) to confirm no regression.

---

## 3. Sources already in use

All four are already-published HF datasets under `mkd-chanwoo/`. Row schema for **all** of
them is identical:

```json
{"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>",
 "source_name": "openorca",
 "license": "mit",
 "n_tokens": 238}
```

`text` is a fully-rendered ChatML string (may contain multiple user/assistant turns). Our
pipeline parses it back into structured `{"role", "content"}` messages — you don't need to
convert anything if you're adding to one of these existing repos.

| Source repo | Rows we take | Rows available | License | Notes |
|---|---:|---:|---|---|
| `mkd-chanwoo/keural-conversation-chatml-ko` | 40,000 | 133,339 | CC BY 4.0 | Korean multi-turn chit-chat |
| `mkd-chanwoo/keural-rag-chatml-ko` | 40,000 | 303,355 | CC BY 4.0 | Korean RAG-style Q&A (grounded in a passage) |
| `mkd-chanwoo/keural-SFT` (Korean subset) | ~50,000 | ~294,000 | MIT / Apache-2.0 / CC BY-SA 4.0 | Filtered to `source_name` ∈ `{orca_math_korean, koinstruct_qa, koinstruct_base, koalpaca}` only |
| `mkd-chanwoo/keural-SFT` (English/replay subset) | 12,000 | large | MIT / CC BY 4.0 | Filtered to `source_name` ∈ `{ultrachat, openorca, alpaca, magicoder, mathinstruct, gsm8k}` only |
| `mkd-chanwoo/keural-QA-chatml-en` | 15,000 | 1,580,359 | CC BY 4.0 | English QA pairs |

### Added 2026-07-09: `Mkd-Yonas/keural-SFT-chatml-ko-v1`

Same row schema, 709,095 total rows across 8 sources, all commercially-clean licenses, no AI
Hub content. Adds real source diversity beyond `mkd-chanwoo`'s own pipeline.

| source_name | rows we take | rows available | license | notes |
|---|---:|---:|---|---|
| `kocommercial` | 20,000 | 171,462 | MIT | |
| `openorca_gugugo_ko` | 20,000 | 315,323 | MIT | machine-translated OpenOrca |
| `kullm_v2` | 15,000 | 127,932 | Apache-2.0 | GPT-translated alpaca/dolly/vicuna |
| `korean_safe_conversation` | 15,000 | 26,803 | Apache-2.0 | safety/refusal behavior |
| `medical_reasoning_ko` | 10,000 | 24,208 | Apache-2.0 | medical QA, CoT traces excluded |
| `kopen_platypus` | 10,000 | 17,464 | CC BY 4.0 | translated Open-Platypus (STEM/logic) |
| `openorca_ko` | 5,000 | 14,853 | MIT | small sample — overlaps `keural-SFT`'s own `openorca`; cross-source dedup handles exact duplicates |
| `claude_reasoning_ko` | 11,050 (all) | 11,050 | apache-2.0 (repackager label) | derived from Claude-model outputs (`Jongsim/claude-opus-4.6-reasoning-12k-ko-filtered-v2`) — **commercial use explicitly confirmed/cleared internally 2026-07-09** |

New running total: **~263,000 rows pre-dedup** (up from ~157,000), before the eval/pilot split
and exact-duplicate removal.

**Note on `keural-SFT`**: it's a big bag of 14 sources tagged by `source_name` — we only pull
specific tags, not the whole dataset (see exclusions below). If you add new data to
`keural-SFT`, **please tag it with a `source_name`** so it can be selectively included, and
record its actual license per-row (not a blanket dataset-level license) — that's what lets us
filter safely.

**A note on repo layout**: `keural-rag-chatml-ko` used to also contain the conversation data;
that's since moved to its own repo (`keural-conversation-chatml-ko`). If you regenerate either
repo's `manifest.json`, please make sure it reflects the actual files present — we found the
manifest inside `keural-rag-chatml-ko` was stale (still listing files that had moved out).

---

## 4. Licensing rules — what's excluded and why

We can only use data that's clearly clean for **commercial** use. The following are excluded
**by default, worst-case assumption**, from `keural-SFT`:

- Rows tagged `source_name` ∈ `{aihub_multisession_sci, aihub_multisession_social, aihub_expert_qa}`
- The standalone dataset `mkd-chanwoo/aihub_conversation`
- Anything derived from `KoCommercial-NoSSL` (there's a CC-BY-NC-4.0 variant and a clean MIT
  variant of this out in the wild — if any of this is in our pipeline, we need to know which
  variant)

**Reason**: AI Hub (Korean government data portal) content is typically licensed for
non-commercial research only, unless your organization has a specific commercial-use
registration with them.

**Action needed from you**: please confirm whether MKD's AI Hub registration permits
commercial use. If yes, tell us explicitly (and which specific datasets/terms apply) — we'll
relax the exclusion. Until then we assume "no" and drop everything AI Hub-tagged.

Beyond the explicit tag list, we also do a belt-and-braces scan: any row whose `license`
field contains "nc", "non-commercial", or "aihub" (case-insensitive) is dropped even if not
in the source_name list above. **So please make sure the `license` field on every row is
accurate** — mislabeling something as `cc-by-4.0` when it's actually NC-restricted is the
one mistake that would slip past our filters.

---

## 5. What we need from you: company/domain data (15,000–20,000 examples)

This is the one part of the dataset that's genuinely ours — not adaptable from public data.

**What it should be**: real prompt + response pairs representative of how people actually use
(or will use) Keural in our product/use case. Ideally sourced from:
- Real user queries + good responses (support logs, chat transcripts, internal Q&A — with any
  PII scrubbed)
- Hand-written examples covering our specific domain terminology, tone, and task types
- If synthetic/LLM-generated, please note that in the license/provenance field — synthetic
  data is fine but needs disclosure so we can decide inclusion

**Format** — plain JSONL, one of these two shapes (we'll auto-detect):

```json
{"messages": [
  {"role": "system", "content": "optional system prompt"},
  {"role": "user", "content": "user message"},
  {"role": "assistant", "content": "the response"}
]}
```
or the same ChatML-string shape as the other sources:
```json
{"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>",
 "source_name": "keural_company", "license": "internal", "n_tokens": 123}
```

Multi-turn conversations are fine (more than one user/assistant pair per row) — just keep the
turns in order, alternating roles, ending on an assistant turn.

**A few concrete rules**:
- No `<think>` blocks needed on your end — our pipeline adds the empty-think formatting
  automatically to every assistant turn.
- Max ~8,192 tokens per example (matches the rest of the dataset's truncation limit) — very
  long transcripts should be split or trimmed.
- Don't worry about deduplication or shuffling — the pipeline does that across all sources.
- License field: use `"internal"` (or your best description) if it's proprietary company
  data — this dataset will not be published as-is, so it doesn't need a public-compatible
  license, but flag anything that came from a third party under its own terms.

**Where to send it**: a single JSONL file (or a few, we'll merge them) — hand it to us or drop
it in the shared dataset location and tell us the path.

---

## 6. What we need from you: domain evaluation set (50–100 examples)

Separate from training data — this is a **held-out** eval set used to score whether Keural
actually improved at real tasks, before/after training. Do not reuse training examples here;
overlap would make the score meaningless.

**Format** — JSONL, one example per line:

```json
{"id": "eval-001",
 "prompt": "the exact user-facing question/task",
 "reference": "an ideal/gold-standard answer",
 "rubric": "optional: what to check for when grading (e.g. 'must mention X, must not do Y')"}
```

Aim for a spread across the real task types Keural will handle — not just easy cases. 50 is
the minimum useful size; 100 is better if you have the time.

---

## 7. Format quick-reference (for anyone hand-authoring examples)

- One JSON object per line (JSONL), UTF-8, no trailing commas.
- Conversations must **end on an assistant turn** (that's what gets trained on).
- Every row needs *some* assistant content — a row with only a user turn is useless and will
  be dropped.
- Keep `source_name` and `license` accurate per row if using the ChatML-string schema — these
  drive both filtering and the published model card's data-provenance table.

---

## 8. Questions?

If anything above is ambiguous for a specific batch of data (e.g. "is this AI Hub-derived?",
"what license does this internal export count as?"), flag it explicitly rather than guessing
— it's much cheaper to ask now than to discover a licensing problem after training.
