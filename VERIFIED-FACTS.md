# VERIFIED-FACTS — Keural SFT pipeline ground truth

Everything below was verified **hands-on on this machine on 2026-07-08** (not taken from the
guide). Where the guide (`keural-sft-training-guide.md`) is wrong, missing, or outdated, it is
called out explicitly. Builder/review agents and future maintainers: treat THIS file as the
source of truth; the guide is the source of *intent*.

## 0. Critical guide corrections

1. **The guide file is TRUNCATED at line 146** (9,807 bytes, ends mid-word in the Section 5
   hyperparameter table: `| Target modules (submodules) | \`q_p`). Missing: the rest of the
   Section 5 hyperparameter table and ms-swift command, and all of Sections 6–8 (referenced
   elsewhere: Section 8 = publishing incl. "preserve/ship the vision processor files").
   Values that had to be reconstructed are marked **[RECONSTRUCTED]** in configs and below.
2. **Guide's Section 2 dataset table is correct**, but the repo layout changed after its
   snapshot: `keural_conversation_shard_0000.jsonl` was split OUT of
   `mkd-chanwoo/keural-rag-chatml-ko` into the standalone repo
   `mkd-chanwoo/keural-conversation-chatml-ko`. The `manifest.json` inside `keural-rag-chatml-ko`
   is **stale** (still lists both sub-datasets). Do not trust that manifest; list repo files.
3. **ms-swift package layout changed** vs. any docs the guide era knew: in ms-swift 4.4.0 the
   modules live under `swift/model/`, `swift/arguments/`, `swift/loss_scale/` (not
   `swift/llm/...`). CLI arg surface (`--target_modules`, `--target_parameters`,
   `--loss_scale ignore_empty_think`) exists as the guide describes.
4. **The model has an MTP (multi-token-prediction) block the guide never mentions**
   (`text_config.mtp_num_hidden_layers = 1`). transformers' modeling class explicitly ignores
   `mtp.*` weights on load (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`,
   modeling_qwen3_5_moe.py:904,1805). Consequence: a naive
   `PeftModel.merge_and_unload()` → `save_pretrained()` **silently drops the MTP weights (and
   on the ForCausalLM path, the vision tower)** from the published model. The merge step MUST
   be a weight-level shard-streaming merge that copies non-targeted tensors byte-identical.

## 1. Machine / environment (verified 2026-07-08)

- KT Cloud NIPA session, NGC PyTorch 25.03-style container, Ubuntu, python 3.12.3.
- 2× NVIDIA H200 (143771 MiB each), driver **570.86.10**, CUDA **12.8** max.
  - **CUDA 13 wheels DO NOT run here** (need r580+ driver). CUDA 12.9 wheels DO run
    (CUDA 12.x minor-version compatibility, r525+).
- **GPUs are currently 100% busy** with another team job (`train_mid.py`, PIDs 2426759/2426760,
  ~139/143 GiB used per GPU). DO NOT start GPU work or call `torch.cuda.is_available()` /
  init a CUDA context while that job runs — even a small context allocation may OOM it.
- Global pip config: `PIP_CONSTRAINT=/etc/pip/constraint.txt` (NVIDIA pins, e.g.
  `torch==2.7.0a0+...nv25.3`) and `PYTHONPATH=/home/work/.local/lib/python3.12/site-packages`
  are exported in the shell. `~/.local/...` is the **shared** package dir used by the running
  GPU job — NEVER pip-install/upgrade there.
- Therefore ALL pipeline commands must run with the project venv AND
  `PYTHONPATH=` (empty) and `PIP_CONSTRAINT=` (empty) for any pip operation:
  `source "/home/work/keural MoE Pro/venv/bin/activate"; export PYTHONPATH=`
- Disk: `/home/work` fs has ~346 GB free (model = ~70 GB, checkpoints & merge need room;
  the merge stage needs ~140 GB free — check before running).
- RAM 2 TiB, 32 cores.

### Pinned venv stack (self-contained in `venv/`, installed 2026-07-08)

| package | version | note |
|---|---|---|
| torch | 2.11.0+cu129 | cu129 = CUDA 12.9 wheels, driver-570 compatible; **cu130 default from PyPI is NOT** |
| torchvision / torchaudio | 0.26.0+cu129 / 2.11.0+cu129 | |
| vllm | 0.24.0+cu129 | from GitHub release wheel `vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` (PyPI wheel is CUDA-13-era) |
| transformers | 5.12.1 | has native `qwen3_5_moe`; ms-swift requires >=5.2.0 for this model |
| ms-swift | 4.4.0 | |
| peft | 0.19.1 | `target_parameters` needs >=0.17.0 ✓ |
| trl | 0.29.1 | |
| accelerate | 1.14.0 | |
| deepspeed | 0.19.2 | built from sdist OK |
| lm_eval | 0.4.12 | eval harness |
| datasets | 4.8.4 | (ms-swift pinned it down from 5.0.0) |
| qwen_vl_utils, decord | installed | required by ms-swift for this model type |

GPU runtime (torch CUDA init, vllm load, NCCL) is **not yet validated** — GPUs were busy.
`scripts/env_check.py --gpu` must be run once the GPUs are free, before any training.

## 2. Model architecture (verified against downloaded config.json + installed transformers 5.12.1 source)

- Repo: `Qwen/Qwen3.6-35B-A3B`, architecture class `Qwen3_5MoeForConditionalGeneration`,
  `model_type: qwen3_5_moe`. 26 safetensors shards. Multimodal (vision+video preprocessors) ✓.
- Text: 40 layers, hidden 2048, vocab 248320, 256 experts (`num_experts_per_tok` 8),
  `moe_intermediate_size` 512, shared_expert_intermediate_size 512, max_position 262144.
- **Hybrid attention ✓ (guide correct):** `full_attention_interval: 4`; layers 3,7,11,…,39
  (0-indexed; 10 of 40) are `full_attention`, the other 30 are `linear_attention`
  (Gated DeltaNet).
- Module tree (dotted paths under `Qwen3_5MoeForConditionalGeneration`):
  - `model.language_model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}` — nn.Linear,
    ONLY on full-attention layers (3,7,...,39); plus non-target `q_norm`/`k_norm`.
  - `model.language_model.layers.{i}.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`
    — nn.Linear, ONLY on linear-attention layers; plus non-target `conv1d` (nn.Conv1d),
    `dt_bias`/`A_log` (nn.Parameter), `norm` (RMSNormGated).
  - `model.language_model.layers.{i}.mlp.gate` — **the ROUTER** (`Qwen3_5MoeTopKRouter`),
    single param `mlp.gate.weight` [256, 2048]. MUST NOT be trainable.
  - `model.language_model.layers.{i}.mlp.experts.gate_up_proj` [256, 1024, 2048] and
    `...mlp.experts.down_proj` [256, 2048, 512] — **raw nn.Parameter ✓ (guide correct)**;
    reachable only via peft `target_parameters`.
  - `model.language_model.layers.{i}.mlp.shared_expert.{gate_proj,up_proj,down_proj}` —
    real nn.Linear ✓.
  - **Name-collision hazard:** `experts.down_proj` (Parameter) vs `shared_expert.down_proj`
    (Linear module) share the suffix `down_proj`. `target_modules=["down_proj"]` matches only
    the Linear; `target_parameters=["down_proj"]`... the verification gate asserts the exact
    resolved set rather than trusting suffix semantics.
  - `model.visual.*` — vision tower (blocks use `attn.qkv`, `attn.proj`,
    `mlp.linear_fc1/linear_fc2`, patch_embed `proj`): **no name collision** with any text-side
    target name above. Must stay frozen (ms-swift `freeze_vit: true` default) and be excluded
    from LoRA.
  - `lm_head`, `model.language_model.embed_tokens` — frozen.
- Tokenizer/template (verified from repo files): `eos = <|im_end|>`, `pad = <|endoftext|>`,
  `bos = null`, `add_bos_token = false` ✓; chat_template's `enable_thinking=false` path emits
  exactly `<think>\n\n</think>\n\n` ✓. `<|im_start|>`/`<|im_end|>` already in vocab ✓.
- generation_config: eos_token_id [248046, 248044], do_sample, temp 1.0, top_p 0.95, top_k 20.

## 3. ms-swift 4.4.0 facts (verified from installed source)

- `Qwen/Qwen3.6-35B-A3B` registered under `MLLMModelType.qwen3_5_moe`, template
  `qwen3_5`, tags vision+video (`swift/model/models/qwen.py:1396-1421`);
  requires `transformers>=5.2.0`, `qwen_vl_utils>=0.0.14`, `decord`.
- `--target_modules` (list), `--target_regex` (single regex; mutually exclusive with
  target_modules), `--target_parameters` (list; peft>=0.17) — `swift/arguments/tuner_args.py`.
- `--loss_scale ignore_empty_think` → zero-loss on `^<think>\s*</think>\s*` prefix of
  assistant content (`swift/loss_scale/config/ignore_empty_think.json`) ✓ guide correct.
- Dataset input: `messages` JSON (`{"messages": [{"role": ..., "content": ...}]}`) ✓.
- `freeze_vit` and `freeze_aligner` default True.

## 4a-i. Open finding (2026-07-10): `koinstruct_base`/`koinstruct_qa` license mismatch

Real dataset build (`build_dataset.py --force`, 2026-07-10) surfaced this via the
`license_summary` output — NOT caught by config/code review, only visible once the actual
data was scanned. The guide's Section 2 table claims `koinstruct_qa`/`koinstruct_base`
(within `mkd-chanwoo/keural-SFT`) are Apache-2.0. **The raw per-row `license` field for
BOTH sources is literally the string `"unknown"`** (verified: 42,276/42,276 `koinstruct_base`
rows and 45,299/45,299 `koinstruct_qa` rows, 100% of each, no exceptions) — not a data
error, not mixed with other values, just uniformly `"unknown"`. These two sources
contributed 13,660 of the 253,528 rows (~5.4%) in the current `train.jsonl` build.

Our belt-and-braces license regex (`(nc|non[-_ ]?commercial|aihub)`) does NOT match
`"unknown"`, so these rows passed through the exclusion filter untouched — this is not a
pipeline bug, the pipeline is accurately reporting what's actually in the source data.

**Unresolved — flagged to user, not decided unilaterally:** either (a) chanwoo's pipeline
truly didn't tag row-level license for these two sources even though the true license is
Apache-2.0 as the guide states, or (b) the license is genuinely unverified for this content.
Action needed: confirm with chanwoo, then either accept-as-is, exclude these two
`source_name`s, or backfill the correct license before publishing the model card's
data-provenance table.

## 4a. Dataset addition (2026-07-09): `Mkd-Yonas/keural-SFT-chatml-ko-v1`

Verified same row schema as §4 (`{"text","source_name","license","n_tokens"}`). Pinned
revision `c56d7885efb37799deb1ecba471e4f6bf263852e`, 709,095 rows across 8 single-shard
sources (`kocommercial`, `kullm_v2`, `openorca_gugugo_ko`, `openorca_ko`, `kopen_platypus`,
`korean_safe_conversation`, `medical_reasoning_ko`, `claude_reasoning_ko`), all
MIT/Apache-2.0/CC-BY-4.0, no AI Hub content per the repo's own README. Smoke-tested through
`build_dataset.py` with zero code changes required (schema-compatible).

`claude_reasoning_ko` (11,050 rows) is derived from Claude-model outputs
(`Jongsim/claude-opus-4.6-reasoning-12k-ko-filtered-v2`); the row-level `license: apache-2.0`
is the repackager's label, not a statement about the underlying generation terms. **User
confirmed 2026-07-09 that commercial use of this specific source has been cleared** — sampled
in full (all 11,050 rows) per `configs/dataset.yaml`.

Wired into `configs/dataset.yaml` as 8 new `sources` entries (`yonas_*`). New pre-dedup total
target: ~263k rows (was ~157k). See `DATASET-REQUIREMENTS.md` for the per-source count table.

## 4. Datasets (verified on HF Hub + local cache 2026-07-08)

Row schema of ALL four sources: `{"text": "<ChatML string>", "source_name": str,
"license": str, "n_tokens": int}` — pre-rendered ChatML that must be parsed back into
`messages` (guide Section 3 ✓). Multi-turn conversations exist (e.g. keural-conversation);
parser must handle N turns, optional system turn.

Pinned revisions (full SHAs, fetched 2026-07-08 — pass as `revision=` for reproducibility):

| repo | revision | rows | use |
|---|---|---|---|
| mkd-chanwoo/keural-conversation-chatml-ko | `64a8c45cf5d777b77fe8e1351e5a52da9dd6868b` | 133,339 | sample 40,000 |
| mkd-chanwoo/keural-rag-chatml-ko | `aa9023d231a3451d25135d5aa84b12b48ca06450` | 303,355 | sample 40,000 |
| mkd-chanwoo/keural-SFT | `ce73cdc04d4b0e2a1cc4d69deb94749691d6ab2c` | 1,144,119 (115 shards, fully in local HF cache) | see filters |
| mkd-chanwoo/keural-QA-chatml-en | `29088c640d47e1cbc380230595d4316e70272efe` | 1,580,359 | sample 15,000 |

keural-SFT filters (guide Section 2, verified against dataset README source table):
- Korean keep: `source_name ∈ {orca_math_korean, koinstruct_qa, koinstruct_base, koalpaca}`
  (294,028 available) → sample ~50,000.
- English/replay keep: `source_name ∈ {ultrachat, openorca, alpaca, magicoder, mathinstruct, gsm8k}`
  → sample 12,000.
- **EXCLUDE always:** `aihub_multisession_sci`, `aihub_multisession_social`, `aihub_expert_qa`
  (AI Hub Terms, non-commercial), and the standalone `mkd-chanwoo/aihub_conversation`.
  Belt-and-braces: also drop any row whose `license` field contains `nc` / `aihub` regardless
  of source_name. (`competition_math` [MIT, 12,040 rows] exists in keural-SFT but is not in the
  guide's include list — excluded by default, config-toggleable.)
- Company/domain data (15–20k): NOT available yet — pipeline accepts an optional local JSONL
  path; skips with a logged warning if unset.
- Note: rows in these datasets have NO empty `<think>` blocks in assistant text — the converter
  adds `<think>\n\n</think>\n\n` prefix to every **final** assistant turn (see guide §3;
  ms-swift's template adds it to non-final turns as needed at render time... the converter adds
  it to ALL assistant turns so rendered output matches the enable_thinking=false path
  exactly; `ignore_empty_think` zero-masks it at loss time).

### Extra architecture finds (not in the guide)

- **`mlp.shared_expert_gate`** (nn.Linear) exists per layer — a gating module like the router.
  No suffix collision with our target names (verified); the gate script asserts it stays frozen.
- **Linear-attention fast path** needs `flash-linear-attention` (triton) + `causal-conv1d`
  (CUDA ext). Without them transformers falls back to a slow torch implementation for 30/40
  layers. Installed into the venv on 2026-07-08 (see pip-fla.log; causal-conv1d compiled from
  source with the container's CUDA 12.8 nvcc).
- **peft ParamWrapper meta-device attach test PASSED** (2026-07-08, peft 0.19.1,
  transformers 5.12.1): with the exact target lists below on an 8-layer shrunk config —
  self_attn adapted only on full-attn layers, linear_attn only on the rest, routed experts on
  all layers, router/shared_expert_gate/vision/embed/lm_head untouched, no violations.
- Expert adapters double-wrap in naming (`experts.base_layer.lora_A` + `experts.lora_A`) —
  one ParamWrapper per targeted parameter of the same module; expected, not a bug.
- **Trainable-size estimate at rank 32 on the full 40-layer model: ~1.9B LoRA params**
  (fused-expert lora_B spans all 256 experts: gate_up B = 256×1024×32, down B = 256×2048×32
  per layer). With bf16 params + fp32 Adam moments, expect roughly 20–25 GB of
  adapter+optimizer state per replica before ZeRO sharding — budget for it; zero2 across
  2 GPUs halves optimizer/grad state.

## 5. Training config decisions ([RECONSTRUCTED] = guide truncated before stating it)

- LoRA rank 32, alpha 64, **dropout 0.0 — REQUIRED, not a choice**: peft 0.19.1
  `lora.ParamWrapper` raises `ValueError` when `lora_dropout != 0` and `target_parameters` is
  used (verified by direct attach test). ms-swift's default is 0.05 → anyone using defaults
  crashes at model prep. Config pins `lora_dropout: 0.0`.
- `target_modules`: `q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z in_proj_b in_proj_a
  out_proj gate_proj up_proj down_proj` — hits full-attn, linear-attn (DeltaNet), and
  shared_expert Linears; guide's surviving text implies all three groups were intended.
- `target_parameters`: `gate_up_proj down_proj` → routed experts. (`down_proj` appears in both
  lists deliberately; the gate asserts the resolved outcome.)
- Router `mlp.gate` NOT targeted; `embed_tokens`, `lm_head`, `model.visual.*` frozen.
- **[RECONSTRUCTED]** LR 1e-4 cosine, warmup_ratio 0.05, 2 epochs, per-device batch 1,
  grad-accum 16 (effective 32), max_length 8192, bf16, gradient checkpointing, deepspeed zero2,
  attn_impl flash_attn if available else sdpa, save every 200 steps + eval loss every 200.
  These are starting points per the guide's own advice; pilot run exists to validate them.
- Pilot: 5,000 examples, same source mix proportions, 1 epoch.
- Pre-training gate (`verify_lora_targets.py`) HARD-FAILS unless (on a meta/tiny instantiation
  of the real class + peft attach): router params frozen; expert Parameters adapted on all 40
  layers; self_attn adapted on exactly the 10 full-attn layers; linear_attn on exactly the 30
  others; shared_expert adapted; vision/embed/lm_head untouched; and the same check re-runs
  against the ms-swift-produced adapter_config.json in the first saved checkpoint.

## 5b. Facts discovered during the build (2026-07-08/09)

- **ms-swift 4.4.0 renamed `train_type` → `tuner_type`** (SftArguments has no `train_type`
  field; `tuner_type` default `lora`). Configs updated. train.py validates EVERY swift_args
  key against the installed SftArguments dataclass before launch, so the next rename aborts
  loudly instead of burning GPU hours.
- **transformers 5.12.1 save trap:** `qwen3_5_moe_text` inherits a qwen2_moe WeightConverter;
  `save_pretrained`'s default `save_original_format=True` would re-export fused expert
  tensors as legacy per-expert `experts.{i}.gate/up/down_proj.weight` names — a DIFFERENT
  layout from the base repo's fused `experts.gate_up_proj [256,1024,2048]`. merge_lora.py
  passes `save_original_format=False`. Anyone saving this model with other tooling must too.
- **peft 0.19.1 `ParamWrapper.merge()` is correct for the fused 3-D expert params**
  (per-expert einsum scaled alpha/r; the nested double-wrap merges recursively via
  `unload_and_optionally_merge_module`). Verified by the merge self-test: forward-pass
  |peft − merged| = 3.6e-07 on a tiny random fixture, and mtp.* re-injection proven.
- **CPU loads of this model can still init CUDA:** with flash-linear-attention installed,
  `FusedRMSNormGated.__init__` calls `torch.cuda.current_device()` at CONSTRUCTION time.
  merge_lora.py sets `CUDA_VISIBLE_DEVICES=""` before importing torch for CPU-side work.
  Any future CPU tooling that instantiates the model must do the same (GPU-busy hazard).
- **lm_eval 0.4.12 CLI**: subcommand form `lm-eval run …`, `--tasks` accepts space- or
  comma-separated lists (verified from --help); run_eval.py's generated commands match.
- **HF token mismatch (open action item for publishing):** this box is logged in as
  `mkd-hika` (orgs: mkd-ai) but the publish target is `mkd-hossain/Keural-35B-A3B`.
  `publish.py` now warns explicitly; before stage 10 run `hf auth login` with a write
  token for `mkd-hossain` (or change `hf.publish_repo`).
- **causal-conv1d prebuilt wheel is ABI-incompatible with torch 2.11.0+cu129** — rebuilt
  from source (`--no-binary causal-conv1d --no-build-isolation --no-cache-dir` with
  `CUDA_HOME=/usr/local/cuda-12.8`); imports clean now. If torch is ever re-pinned,
  rebuild it again.

## 6. Eval facts

- lm_eval 0.4.12 task registry (verified installed): `kmmlu` (+`kmmlu_direct`), `haerae`,
  `kobest`, `mmlu`, `gsm8k`, `humaneval` all present.
- vllm 0.24.0 supports qwen3_5_moe (registered in its model registry) — but GPU-runtime
  untested on this driver until `env_check.py --gpu` passes.
- HumanEval requires `--confirm_run_unsafe_code` and `HF_ALLOW_CODE_EVAL=1`.
- Domain eval: placeholder harness at `evals/domain/` (guide: 50–100 real examples; not yet
  provided by the team).
