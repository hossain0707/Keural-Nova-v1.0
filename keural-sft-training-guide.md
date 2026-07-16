# Keural SFT Training Guide (Qwen3.6-35B-A3B → Keural)

Path: **SFT-only** (no continued pretraining). Base model: `Qwen/Qwen3.6-35B-A3B` (MoE, ~3B active params/token, Apache 2.0). Hardware: 2× H200. Method: LoRA.

Note: I don't have access to your actual GPU cluster — this is the runbook to execute there. Treat every hyperparameter below as a starting point to sanity-check against your own pilot run, not a guarantee.

## ⚠️ Architecture reality check (read this before Section 5)

Confirmed directly against the real `config.json` and the real `modeling_qwen3_5_moe.py` source in `transformers` (fetched and read line-by-line, not inferred) — this is a more unusual architecture than a "standard" MoE:

- **Hybrid attention:** only 1 in 4 layers is full self-attention (`self_attn.q/k/v/o_proj`). The other 3 in 4 use a different "linear attention" mechanism (Gated DeltaNet, module names `linear_attn.in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj`). If you only target the standard attention module names, you're adapting a quarter of the network's sequence-mixing layers.
- **It's multimodal** (vision-language) — has a vision encoder/processor, not text-only. Doesn't change your Korean text SFT data, but does mean you need to preserve/ship the vision processor files when publishing (Section 8).
- **Routed experts are NOT `nn.Linear` submodules.** They're fused into raw `nn.Parameter` tensors (`gate_up_proj`, `down_proj`) inside the experts container — standard `--target_modules` name-matching cannot attach LoRA to them at all. Only the small per-layer `shared_expert` (a real `nn.Linear`) would get adapted if you use plain module-name targeting. This fails silently: training runs, loss goes down, checkpoints save, and the actual routed experts — most of the model's capacity — never get touched.

**Before your pilot run, hands-on verify this yourself** — don't trust this guide or any automated source-reading blindly on an architecture this nonstandard:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", torch_dtype="bfloat16")
for name, module in model.named_modules():
    if "layers.0." in name or "layers.1." in name:
        print(name, type(module).__name__)
```

Confirm the actual module/parameter names in your downloaded copy match what Section 5 assumes before trusting the training command.

---

## 0. Environment setup

- [ ] Confirm both H200s visible: `nvidia-smi`
- [ ] Install training framework: **ms-swift** (ModelScope) is the recommended choice — built by the same ecosystem as Qwen, with native MoE-aware LoRA support (explicit control over which modules get adapted, including the router). Unsloth/Axolotl/LLaMA-Factory are alternatives, but whichever you use, verify actual trainable modules before scaling up (see Section 5) — don't trust an "auto-detect all linear layers" flag on an MoE model.
- [ ] Install `transformers`, `peft`, `trl`, `huggingface_hub`, `vllm` (for fast eval/serving)
- [ ] Sanity-check 2-GPU communication: run a trivial `torchrun --nproc_per_node=2` job before touching real data
- [ ] Log into Hugging Face CLI (`huggingface-cli login`) with a token that has write access to wherever you'll publish

---

## 1. Baseline evaluation (do this before anything else)

Without this, you cannot prove Keural is actually better — do not skip.

- [ ] Korean eval: KMMLU, HAE-RAE, KoBEST (public benchmark sets)
- [ ] General eval: MMLU, GSM8K, HumanEval (or your preferred general set) — this is your regression tripwire
- [ ] Your own domain eval: 50-100 real examples from your actual use case, graded consistently (rubric or LLM-judge)
- [ ] Run stock `Qwen/Qwen3.6-35B-A3B` against all three, save every score and raw output to a file. This is your only source of truth for "did we actually improve."

---

## 2. Dataset collection

All sources below are already commercially clean (verified against each dataset's own license table) and already ChatML-formatted — no external data collection needed beyond your own company data.

### Korean (~85%, target ~145,000-150,000 examples)

| Source | Samples to use | Available | License |
|---|---:|---:|---|
| `mkd-chanwoo/keural-conversation-chatml-ko` | 40,000 | 133,339 | CC BY 4.0 |
| `mkd-chanwoo/keural-rag-chatml-ko` | 40,000 | 303,355 | CC BY 4.0 |
| `mkd-chanwoo/keural-SFT` — rows where `source_name` is `orca_math_korean`, `koinstruct_qa`, `koinstruct_base`, or `koalpaca` only | ~50,000 | ~294,000 clean | MIT / Apache-2.0 / CC BY-SA 4.0 |
| Your own company/domain prompts + responses | 15,000-20,000 | whatever you have | Yours |

### General / replay (~15%, target ~27,000 examples)

| Source | Samples to use | Available | License |
|---|---:|---:|---|
| `mkd-chanwoo/keural-QA-chatml-en` | 15,000 | 1,580,359 | CC BY 4.0 |
| `mkd-chanwoo/keural-SFT` — rows where `source_name` is `ultrachat`, `openorca`, `alpaca`, `magicoder`, `mathinstruct`, or `gsm8k` | 12,000 | large | MIT / CC BY 4.0 |

### Explicitly exclude (license-restricted for commercial use)

- `mkd-chanwoo/keural-SFT` rows tagged `aihub_multisession_sci`, `aihub_multisession_social`, `aihub_expert_qa` — AI Hub Terms, non-commercial research only
- `mkd-chanwoo/aihub_conversation` (standalone dataset) — same AI Hub restriction
- Any `KoCommercial-NoSSL`-derived data if it entered your pipeline — CC-BY-NC-4.0 variant exists alongside a clean MIT variant; confirm which one was used

**Action item:** confirm with chanwoo which exact AI Hub license terms apply to your org (registered commercial use vs. research-only) before deciding whether to keep the exclusions above — the guide currently assumes worst-case (excluded).

---

## 3. Data formatting

Qwen3.6 uses native ChatML special tokens (`<|im_start|>`, `<|im_end|>` — verified against its own tokenizer config, already in vocab, no need to add them). Format every sample as:

```
<|im_start|>system
{system_prompt, if present}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

{response}<|im_end|>
```

Key details:

- **Empty `<think>\n\n</think>\n\n` block is required** before every assistant response. This matches Qwen3.6's own `enable_thinking=false` template path — it keeps the model's reasoning-mode behavior intact (untouched by this fine-tune) while your SFT data teaches the non-thinking response path consistently. Do not mix empty-think and no-think formatting — pick one (empty-think) and apply it everywhere.
- `eos_token` = `<|im_end|>` — make sure loss masking includes this token so the model learns to stop generating.
- `pad_token` = `<|endoftext|>`, `bos_token` = none (do not prepend a BOS token).
- Mask loss on system/user turns — only train on assistant tokens.
- **Don't manually compute loss over the empty `<think></think>` scaffold as a hard rule.** ms-swift has a purpose-built flag for exactly this (see Section 5: `--loss_scale ignore_empty_think`) — it skips loss on the empty think block specifically, rather than teaching the model "always emit empty think" as something to memorize. Use that instead of blanket-training the whole assistant span.
- Max sequence length: keep consistent with source data truncation (8,192 tokens is what `keural-SFT` used).
- Shuffle the final merged dataset — don't leave it grouped by source.

**Dataset schema:** ms-swift expects structured JSON, not a pre-rendered ChatML string — it applies the chat template itself. Convert `keural-SFT`'s stored `text` field back into role/content pairs:

```json
{"messages": [
    {"role": "system", "content": "<system prompt, if present>"},
    {"role": "user", "content": "<user message>"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\n<response>"}
]}
```

Do not feed it the raw literal `<|im_start|>...<|im_end|>` text — that's for inspecting/verifying output, not the training input format.

---

## 4. Pilot run (do not skip)

With only 2 GPUs, a broken pipeline discovered after a multi-hour run is expensive. Before the full run:

- [ ] Build a ~5,000-example slice proportional to the mix above
- [ ] Run 1 short training pass end-to-end
- [ ] Verify: loss decreases, no tokenizer/special-token errors, checkpoint saves and reloads correctly
- [ ] Run your Section 1 eval sets against this pilot checkpoint — confirm general scores haven't collapsed (a collapse here means a formatting or LR bug, not a real result — fix before scaling up)

---

## 5. Full SFT training

**Method:** LoRA, rank 32 — not the higher rank you'd use for continued pretraining. For SFT-only (teaching format/style/task-following on an already-capable base, no new knowledge injection), rank 32 is standard practice; rank 256 on ~170k examples with no CPT underneath it invites overfitting and wastes VRAM for no real benefit.

**Critical: the router is easy to accidentally leave untouched (good) but the routed experts are easy to accidentally leave untouched too (bad — this is the failure mode to design around).**

Per the architecture reality check above: standard `--target_modules` name-matching (PEFT does exact dotted-suffix matching, so `gate_proj` genuinely cannot collide with the router's `mlp.gate`) will correctly skip the router, but it will *also* skip the routed experts entirely, since they're fused `nn.Parameter` tensors, not named submodules. Use `--target_parameters` for those instead — a newer PEFT/ms-swift feature built for exactly this MoE pattern.

**Suggested starting hyperparameters** (validate against your pilot run and the hands-on module check above — don't take these as final):

| Parameter | Value |
|---|---|
| LoRA rank | 32 |
| LoRA alpha | 64 (2× rank) |
| Target modules (submodules) | `q_p