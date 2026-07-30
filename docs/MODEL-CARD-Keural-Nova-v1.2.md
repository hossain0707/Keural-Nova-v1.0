---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
language:
- ko
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- keural
- keural-nova
- mkd
- korean
- bilingual
- tool-calling
- function-calling
- agent
- moe
- qwen3
---

# Keural Nova v1.2 (experimental)

**Keural Nova v1.2** is a bilingual (Korean/English) large language model developed by
**[MKD](https://mkd.kr)** — an AI company based in Busan, South Korea. It is a LoRA
supervised fine-tune of **Qwen3.6-35B-A3B** (35B parameters, ~3B active MoE), tuned to improve
**Korean fluency, tool-calling (including long context), and code** while preserving the base
model's general capability.

> ⚠️ **Experimental release.** This is a v1.2 *experimental run*. It is the strongest Keural
> Nova to date on code and tool-calling, with honest trade-offs documented below.

---

## Highlights

- 🥇 **Best-in-line tool-calling** — 12/12 on our function-calling suite (base Qwen: 11/12),
  robust multi-turn (call → result → answer), and verified tool-calling in **very long context
  (~237k tokens)**.
- 🥇 **Code recovered and beats base** — HumanEval **68.3** (base 62.2; earlier v1.0/v1.1 had
  regressed to ~12–21 due to a data issue, now fixed with indentation-validated code).
- 🥇 **Best Korean of the Nova line** — KoBEST **70.0** and KMMLU **63.5**, the highest of all
  Keural Nova versions.
- 🌏 **256K native context** (Qwen3.6 architecture; up to 1M via YaRN — see *Serving*).

---

## Evaluation

Same harness (lm-eval, vLLM backend, seed 42) across base and every Keural Nova version.

| Benchmark | Base | Nova v1.0 | Nova v1.1 | **Nova v1.2** |
|---|---:|---:|---:|---:|
| **KoBEST** (ko, conversational) | 62.88 | 68.82 | 67.29 | **70.03** |
| **KMMLU** (ko, knowledge) | 64.46 | 62.52 | 62.38 | **63.50** |
| HAE-RAE (ko, knowledge) | **75.53** | 73.42 | 74.52 | 73.88 |
| **MMLU** (en, knowledge) | **83.83** | 82.07 | 82.37 | 82.88 |
| GSM8K (math, strict) | 32.15¹ | **87.11** | 86.88 | 72.25 |
| **HumanEval** (code) | 62.20 | 20.73 | 11.59 | **68.29** |

¹ Base GSM8K is depressed by answer-format mismatch under strict-match; the fine-tuned gain is
largely format compliance.

**Tool-calling** (our XML function-calling suite, non-thinking, temperature 0):

| Metric | Base Qwen | **Nova v1.2** |
|---|---:|---:|
| Single-turn (12 cases) | 11/12 | **12/12** |
| Multi-turn (call → tool result → final answer) | ✅ | ✅ |
| Long-context tool call (~237k-token context) | — | ✅ (correct call + argument) |

### How v1.2 compares

- **Better than v1.0 and v1.1 overall** — best Korean MCQ (KMMLU), best KoBEST, best MMLU of the
  fine-tunes, dramatically better code, and the strongest tool-calling (incl. long context).
- **Trade-offs (honest):** GSM8K (**72.3**) is lower than v1.0/v1.1 (~87) though still far above
  base; HAE-RAE (73.9) is ~flat vs v1.1 (74.5). Small knowledge dips vs base on MMLU (−0.9) and
  HAE-RAE (−1.7) are the expected SFT trade-off (SFT tunes style/skill, not stored knowledge).

---

## Intended use

General assistant, **Korean/English chat and RAG**, **agentic / tool-calling** workloads
(web search, document QA, function calling), and **coding**. Especially suited to Korean
enterprise assistants and agent frameworks.

## Serving (vLLM)

Recommended config for the tool/agent workload — **native 256K context**:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mkd-hossain/Keural-Nova-v1.2-experimental \
  --served-model-name Keural-Nova-v1.2 \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len 262144 \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice
```

- **Non-thinking is the default** (the chat template is set so the model answers/tool-calls
  directly). `enable_thinking=true` remains available per request.
- **Tool calls use Qwen XML** (`<tool_call><function=NAME><parameter=P>VAL</parameter></function></tool_call>`)
  — serve with `--tool-call-parser qwen3_xml --enable-auto-tool-choice`.
- **1M context** is available via YaRN (`rope_scaling`, `factor` 4.0), but static YaRN degrades
  short-prompt tool-calling on Qwen models generally; **serve native 256K for agent/tool use**
  and enable YaRN only when a request truly needs >256K.

## Training

- **Base:** `Qwen/Qwen3.6-35B-A3B` (fine-tuned fresh from base, not from v1.0/v1.1).
- **Method:** LoRA (rank 16, α 32, dropout 0.0) via `ms-swift`, targeting attention +
  Gated-DeltaNet linear-attention + shared experts + all routed MoE experts
  (PEFT `target_parameters`); **router and gates frozen**. 1 epoch, LR 5e-5, bf16,
  DeepSpeed ZeRO-2, 2× H200.
- **Data:** ~162k examples — a cleaned, balanced Korean/English/code/replay mix
  (AST-validated code, benchmark test-splits excluded, identity de-contaminated) plus a
  **tool-calling slice** (~17k: single-turn, multi-turn, negative, and long-context up to 32k;
  sources: Glaive-function-calling-v2 and Hermes-function-calling, both Apache-2.0, converted to
  Qwen XML) and a small Korean tool-use set.
- **Design goal:** improve Korean + tool-calling + code while holding general capability flat
  (verified against a base-model eval gate before release).

## Limitations

- **Experimental**: intended for evaluation; validate on your workload before production.
- **GSM8K** math word-problems are weaker than Nova v1.0/v1.1.
- Small knowledge dips vs base on MMLU / HAE-RAE (expected SFT trade-off).
- Tool-calling at the full **1M**-YaRN context is not guaranteed (a serving-side YaRN property,
  not a model defect); native 256K is recommended for agents.
- Identity is a fine-tuned behavior; it is not adversarially hardened.

## License & attribution

Released under **Apache-2.0**, inheriting the license of the base model
**Qwen/Qwen3.6-35B-A3B** (© Alibaba Cloud / the Qwen team), whose license and attribution are
retained. "Keural" and "Keural Nova" are model names by **MKD**.

## Citation

```
@misc{keural-nova-v1_2,
  title  = {Keural Nova v1.2 (experimental)},
  author = {MKD},
  year   = {2026},
  url    = {https://huggingface.co/mkd-hossain/Keural-Nova-v1.2-experimental}
}
```
