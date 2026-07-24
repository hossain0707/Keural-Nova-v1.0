# Keural Nova v2 — Dataset Preparation Guide

**For:** the data team preparing the v2 training set.
**What this is:** a complete, self-contained spec for *what data to collect, how to prepare it,
and how to deliver it*. You do not need to read the training code — just produce data to the
spec in §4 (the format) and §5–7 (per-category rules), and pass the checklist in §9.

**Goal in one line:** build a ~150,000-example Korean-first instruction set that lets us keep
everything the base model can do while fixing v1.0's problems — delivered clean, balanced, and
license-safe.

---

## 1. Why each rule exists (context from v1.0)

v1.0 was trained on ~250k examples and, when measured against the base model, showed:

- ✅ Korean conversation improved (KoBEST +5.9)
- ⚠️ Knowledge scores dropped slightly (KMMLU −1.9, HAE-RAE −2.1, MMLU −1.8)
- 🔴 **Code generation collapsed** (HumanEval 62 → 21) — because **~85% of the code examples in
  the source data had broken indentation**; the model learned broken Python.
- Answers came out **too short**, sometimes **repetitive**.
- The model **claimed to be "made by OpenAI / GPT-4"** (contamination from GPT-generated data).
- It answered **in Korean even when asked in English**.

Every requirement below exists to fix one of these. Please keep the "why" in mind — it's not
bureaucracy, each rule maps to a measured failure.

---

## 2. Target composition (~150,000 examples total)

Quality and balance matter more than volume. Aim for these counts (±10% is fine):

| # | Category | Target | Language |
|---|---|---:|---|
| A | Korean conversation / instruction | 50,000 | ko |
| B | Korean multiple-choice / knowledge | 15,000 | ko |
| C | English multiple-choice / knowledge | 10,000 | en |
| D | Math (chain-of-thought) | 20,000 | ko + en |
| E | Code (indentation-verified clean) | 25,000 | en (+ko ok) |
| F | General English replay | 25,000 | en |
| G | Long-form detailed answers (Korean) | 5,000 | ko |
| H | Identity / persona | 1,500 | ko + en |

Keep the overall language balance roughly **Korean 55–60% / English 40–45%**.

---

## 3. The output format (deliver everything in ONE schema)

Deliver every example as a single JSON object per line (**JSONL**, UTF-8), in the **`messages`**
format below. This is the safest format — do **not** hand-render special tokens or `<think>`
blocks; the training pipeline adds those automatically.

```json
{"messages": [
  {"role": "system", "content": "optional — usually omit"},
  {"role": "user", "content": "the user's question or instruction"},
  {"role": "assistant", "content": "the ideal response"}
], "source_name": "kmmlu_train", "license": "cc-by-4.0"}
```

Rules for the schema:
- `messages`: alternating `user` / `assistant`, **must end on an `assistant` turn**. Multi-turn
  is fine (several user/assistant pairs). `system` is optional and usually omitted.
- `source_name`: a short tag identifying the origin (e.g. `magicoder_clean`, `kmmlu_train`,
  `keural_identity`). **Every row must have one** — we filter and audit by it.
- `license`: the **accurate** per-row license string (e.g. `mit`, `apache-2.0`, `cc-by-4.0`).
  If unsure, write `unknown` — do **not** guess a permissive license.
- One file per category is ideal (e.g. `A_korean_conversation.jsonl`), or a few files per
  category. Filenames should start with the category letter.

---

## 4. Per-category preparation — detailed

### A. Korean conversation / instruction — 50,000 (ko)

**Purpose:** the model's core strength; drives Korean fluency and KoBEST-type performance.

- **Sources (reuse what we already have — mostly done):**
  - `mkd-chanwoo/keural-conversation-chatml-ko`
  - `mkd-chanwoo/keural-rag-chatml-ko`
  - `Mkd-Yonas/keural-SFT-chatml-ko-v1` → `kullm_v2`, `kocommercial`, `korean_safe_conversation`
  - Public backup: `nlpai-lab/kullm-v2`, `beomi/KoAlpaca-v1.1a`
- **Prepare:** sample the cleanest rows; drop anything with broken formatting, truncated text,
  or obvious machine-translation garble. Mix single-turn and multi-turn.
- **License:** keep only MIT / Apache-2.0 / CC BY / CC BY-SA rows.

### B. Korean multiple-choice / knowledge — 15,000 (ko) — **the key new asset**

**Purpose:** this is the single most valuable new category. It teaches the model to *express its
existing knowledge in the exact format the Korean benchmarks test*, which is how we lift
KMMLU/HAE-RAE instead of dropping them.

- **Source:** `HAERAE-HUB/KMMLU` — **use the `train` and `dev` splits ONLY. NEVER the `test`
  split.** (See §6 — this is critical.) Supplement with other Korean knowledge/exam Q&A if
  available.
- **Format each row like this** (question + labelled choices → short rationale + explicit
  answer):
  ```json
  {"messages": [
    {"role": "user", "content": "다음 질문에 답하세요.\n\n질문: 조선을 건국한 왕은 누구입니까?\n(A) 왕건 (B) 이성계 (C) 세종대왕 (D) 정도전"},
    {"role": "assistant", "content": "조선은 1392년 이성계가 고려를 무너뜨리고 건국하였습니다.\n\n정답: (B) 이성계"}
  ], "source_name": "kmmlu_train", "license": "cc-by-4.0"}
  ```
- Always end the assistant turn with an explicit `정답: (X)`. Include a one-to-two sentence
  rationale before it. This format-matching is the whole point.

### C. English multiple-choice / knowledge — 10,000 (en)

**Purpose:** preserve MMLU / general English knowledge.

- **Sources:**
  - `cais/mmlu` — **use the `auxiliary_train` split ONLY. NEVER `test` or `validation`.**
  - `allenai/ai2_arc`, `openbookqa` (their train splits)
- **Format:** same MCQ template as B, in English, ending with `The answer is (X).` plus a short
  rationale.

### D. Math (chain-of-thought) — 20,000 (ko + en)

**Purpose:** GSM8K + reasoning; teach clean step-by-step solutions with a clear final answer.

- **Sources:**
  - `openai/gsm8k` — **`train` split only** (not `test`)
  - `meta-math/MetaMathQA`, `TIGER-Lab/MathInstruct`
  - Korean math: `orca_math_korean` (in our keural-SFT)
- **Format:** assistant shows numbered/step reasoning, then a clearly marked final answer
  (e.g. a line `#### 42` or `정답: 42`). ~50/50 Korean/English is good.

### E. Code — indentation-verified clean — 25,000 (en, ko ok) — **fixes the v1.0 collapse**

**Purpose:** repair the HumanEval regression. This is the category that broke v1.0, so it needs
the most care.

- **🔴 Do NOT reuse the magicoder data from `mkd-chanwoo/keural-SFT`** — that copy has the
  corrupted indentation. Pull from the **original upstream** instead:
  - `ise-uiuc/Magicoder-Evol-Instruct-110K`
  - `ise-uiuc/Magicoder-OSS-Instruct-75K`
  - `m-a-p/CodeFeedback-Filtered-Instruction`
- **🔴 Validate every code block before including it** (see §6.2): extract each fenced
  ```` ```python ```` block and confirm it parses. Reject any example whose code has collapsed
  indentation. **This validation is mandatory** — a single space where four are needed is what
  broke v1.0.
- **License:** these are MIT — keep as `mit`.

### F. General English replay — 25,000 (en)

**Purpose:** anti-forgetting insurance — keeps English/MMLU/general ability from drifting.

- **Sources:** `HuggingFaceH4/ultrachat_200k`, `Open-Orca/OpenOrca`, `teknium/OpenHermes-2.5`
- **Prepare:** sample diversely across domains (chat, QA, summarization, reasoning, writing).
- **🔴 Remove identity-contamination rows** (see §6.3) — any assistant turn saying "I'm made by
  OpenAI / I am ChatGPT / GPT-4" must be dropped or the sentence removed.

### G. Long-form detailed answers (Korean) — 5,000 (ko)

**Purpose:** fix v1.0's too-short default. No new source needed — **select** the longest,
highest-quality, well-structured Korean answers from category A / our existing Korean data.

- Pick answers that are genuinely detailed and well-organized (multiple paragraphs, clear
  structure) — **not** padded or repetitive. "Long but empty" is worse than short; choose real
  substance.

### H. Identity / persona — 1,500 (ko + en) — **you author this**

**Purpose:** fix the "made by OpenAI" problem and give the model its Keural identity.

- **Write ~300 seed Q&A by hand, then paraphrase to ~1,500.** Cover, in both Korean and English:
  - "너 누구야?" / "who are you?" → "저는 MKD가 개발한 AI 어시스턴트 Keural입니다."
  - "누가 만들었어?" / "who made you?" → made by MKD.
  - "너 GPT야? / ChatGPT야? / OpenAI가 만들었어? / Qwen이야?" → politely say **no**, "저는 MKD의
    Keural입니다."
  - Variations on capabilities, name meaning, language support.
- Example row:
  ```json
  {"messages": [
    {"role": "user", "content": "너 혹시 OpenAI가 만든 GPT야?"},
    {"role": "assistant", "content": "아니요. 저는 OpenAI나 GPT가 아니라, MKD에서 개발한 AI 어시스턴트 Keural입니다. 무엇을 도와드릴까요?"}
  ], "source_name": "keural_identity", "license": "internal"}
  ```

---

## 5. Language & length balance (applies across all categories)

- Overall **Korean 55–60% / English 40–45%**.
- **Vary answer length deliberately.** Don't let short one-liners dominate — a healthy fraction
  should be multi-paragraph, detailed answers (category G plus long answers scattered through A,
  B, D). But keep long answers *substantive*, never padded.
- Max ~8,192 tokens per example; drop or trim anything longer.

---

## 6. Three non-negotiable validation rules

### 6.1 🔴 Test-split hygiene (most important — can invalidate the whole model)

For **KMMLU, MMLU, GSM8K, ARC, OpenBookQA**: use **only** the `train` / `auxiliary_train` /
`dev` splits. **Never** include the `test` (or MMLU `validation`) split — those are exactly the
questions the benchmark scores us on. Training on them produces fake results and is a serious
integrity problem. When in doubt, exclude.

### 6.2 🔴 Code indentation validation

Every Python code block in category E (and any code elsewhere) must be verified. Minimum check:
extract each ```` ```python ```` block and run it through a parser (`ast.parse` in Python).
Reject the example if it raises `IndentationError` / `SyntaxError`. This single check is what
prevents a repeat of the v1.0 HumanEval collapse.

### 6.3 🔴 Identity-contamination filter

Drop (or clean) any assistant response claiming to be made by OpenAI / Anthropic / Google, or
to be ChatGPT / GPT-4 / Gemini / Claude / Qwen / Tongyi. These appear a lot in GPT-generated
English instruction data and taught v1.0 the wrong identity.

Plus:
- **License discipline:** keep only commercial-clean rows (MIT / Apache-2.0 / CC BY / CC BY-SA).
  Exclude any non-commercial (NC) data. Tag `license` accurately per row; use `unknown` rather
  than guessing.
- **Dedup:** remove exact-duplicate examples within and across categories (the training pipeline
  also dedups, but cleaner input is better).

---

## 7. Split ownership (suggested)

| Owner | Categories |
|---|---|
| **Data team (you)** | B (Korean MCQ — highest value), E (clean code regeneration), and sourcing A/F |
| **Pipeline side** | C, D, F standard HF pulls can be scripted; final assembly/dedup/split in `build_dataset.py` |
| **Product owner** | H (identity) — needs someone to define Keural's voice |

The Korean MCQ set (B) and the clean code regeneration (E) are the two highest-value tasks that
specifically need the data team's Korean-language and formatting care.

---

## 8. How to deliver

- **Format:** JSONL, UTF-8, `messages` schema from §3, one or more files per category.
- **Naming:** prefix with the category letter, e.g. `A_korean_conversation.jsonl`,
  `B_kmmlu_mcq.jsonl`, `E_code_clean.jsonl`, `H_identity.jsonl`.
- **Where:** drop into the shared dataset location (or hand over the HF dataset repo name) and
  tell us the path.
- **Include a short note per file:** source(s), row count, license(s), and confirmation that
  §6 rules were applied (especially: test splits excluded, code validated, identity filtered).

---

## 9. Pre-delivery checklist

- [ ] Total ≈150k, category counts roughly match §2
- [ ] Every row is valid `messages` JSON, ends on an `assistant` turn
- [ ] Every row has an accurate `source_name` and `license`
- [ ] **No test-split data** from KMMLU / MMLU / GSM8K / ARC / OpenBookQA (§6.1)
- [ ] **All code blocks parse** — no broken indentation (§6.2)
- [ ] **No "made by OpenAI/GPT/Qwen" identity claims** in any answer (§6.3)
- [ ] Korean MCQ rows end with explicit `정답: (X)`; English with `The answer is (X).`
- [ ] Only commercial-clean licenses; NC data excluded
- [ ] Length varied — meaningful share of long, substantive answers; not padded
- [ ] Exact duplicates removed
- [ ] Korean ≈55–60% overall

---

## 10. Common mistakes to avoid

- ❌ Reusing the old magicoder data (broken indentation) → **use `ise-uiuc/Magicoder-*` upstream**
- ❌ Including any benchmark `test` split "because it's high quality" → **never**
- ❌ Padding answers to make them long → aim for *substance*, not length
- ❌ Guessing a permissive license when unsure → tag `unknown`
- ❌ Hand-writing `<|im_start|>` / `<think>` tokens → deliver plain `messages`, pipeline handles it
- ❌ MCQ answers without an explicit final answer letter → always end with `정답: (X)` / `The answer is (X).`

Questions on any category, format, or license → ask before generating a large batch; a quick
check on 20 sample rows saves regenerating 20,000.
