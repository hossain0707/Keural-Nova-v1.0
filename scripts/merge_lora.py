#!/usr/bin/env python
"""Stage 7a — CPU-only LoRA merge for Keural (Qwen3.6-35B-A3B).

WHY this script exists (instead of a three-line merge_and_unload snippet):

* transformers' ``Qwen3_5MoeForConditionalGeneration`` declares
  ``_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`` — the multi-token-prediction
  block present in the base checkpoint is silently DROPPED by
  ``from_pretrained`` (VERIFIED-FACTS §0.4).  A naive
  load → merge_and_unload → save_pretrained therefore publishes a model whose
  tensor set is smaller than the base repo's.  This script re-injects the
  ``mtp.*`` tensors byte-identically from the base shards and patches the
  safetensors index, then PROVES (tensor-set / shape / dtype / byte-level
  verification) that nothing else was lost or corrupted.

* MoE LoRA fails silently when misconfigured: routed experts are fused
  ``nn.Parameter`` tensors handled by peft's ``lora.ParamWrapper``.  The merge
  verifier asserts that targeted tensors actually CHANGED vs the base and that
  untargeted tensors (router, embeddings, vision tower, norms, mtp) are
  byte-identical.

peft 0.19.1 ParamWrapper findings (verified by reading the installed
site-packages/peft/tuners/lora/layer.py on 2026-07-08 — re-checked dynamically
in :func:`paramwrapper_findings`):

* ``ParamWrapper.merge()`` IS correctly implemented for fused 3-D parameters:
  ``get_delta_weight`` reshapes lora_A/lora_B into per-expert slices and applies
  ``einsum("o r e, e r i -> e i o")`` (or ``-> e o i`` when the base parameter is
  stored transposed, tracked via ``_did_swap_in_out_features``), scaled by
  alpha/r, then adds in-place to ``param.data`` (layer.py ~2380).
* Nested double-wrapping (one ParamWrapper per targeted parameter of the same
  module, e.g. ``experts.gate_up_proj`` + ``experts.down_proj``) is handled by
  ``ParamWrapper.unload_and_optionally_merge_module`` which merges recursively
  through inner wrappers (layer.py:2430) — ``merge_and_unload()`` uses that
  hook, so BOTH adapters of a double-wrapped module are merged.
* Constraints enforced by ParamWrapper: lora_dropout must be 0.0 (raises
  otherwise), no fan_in_fan_out / DoRA / lora_bias.

=> Path A (load + merge_and_unload + save, then MTP re-injection) is safe and
is what this script implements.

CPU-only by construction: ``CUDA_VISIBLE_DEVICES`` is cleared before torch is
imported (another team's job owns the GPUs).  This has a second, load-bearing
effect: transformers' ``is_causal_conv1d_available()`` /
``is_flash_linear_attention_available()`` both AND with
``is_torch_cuda_available()``, so with CUDA hidden the modeling module falls
back to pure-torch kernels.  That sidesteps two verified hazards:
(1) the venv's causal_conv1d_cuda .so fails to import (ABI mismatch with
torch 2.11.0+cu129), which would crash the modeling import; and
(2) fla's ``FusedRMSNormGated.__init__`` calls ``torch.cuda.current_device()``,
which would initialize a CUDA context on the busy GPUs at model construction.
"""

from __future__ import annotations

import os

# MUST run before any torch/transformers import — see module docstring.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import common
from common import PipelineError

DEFAULT_CONFIG = "configs/publish.yaml"
MTP_SHARD_NAME = "model-mtp.safetensors"
MTP_RE = re.compile(r"^mtp\.")

# Files shipped alongside the merged weights (guide §8 intent: the model is
# multimodal — the vision/video processor files must ship).  config.json and
# generation_config.json are written by save_pretrained and are NOT clobbered;
# generation_config.json is only copied if save_pretrained did not produce one.
AUX_PATTERNS = [
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
]


# ------------------------------------------------------------------ peft audit


def paramwrapper_findings() -> str:
    """Dynamically re-verify the installed peft ParamWrapper merge contract.

    WHY: the merge path relies on non-obvious peft internals (recursive nested
    merge).  If peft is ever re-pinned, this fails loudly instead of producing
    a silently half-merged model.
    """
    import inspect

    import peft
    from peft.tuners.lora.layer import ParamWrapper

    problems = []
    for attr in ("merge", "get_delta_weight", "unload_and_optionally_merge_module"):
        if not hasattr(ParamWrapper, attr):
            problems.append(f"ParamWrapper.{attr} missing")
    src = inspect.getsource(ParamWrapper.unload_and_optionally_merge_module) if not problems else ""
    if src and "while isinstance(base_layer, ParamWrapper)" not in src:
        problems.append(
            "ParamWrapper.unload_and_optionally_merge_module no longer merges nested "
            "wrappers recursively — double-wrapped expert adapters would be dropped"
        )
    if problems:
        raise PipelineError(
            f"peft {peft.__version__} ParamWrapper contract changed: {problems}. "
            "Re-audit peft/tuners/lora/layer.py before merging (a wrong merge is "
            "exactly the silent failure this pipeline exists to prevent)."
        )
    return (
        f"peft {peft.__version__}: ParamWrapper.merge applies per-expert einsum delta "
        "(B@A * alpha/r, transposed-layout aware) in-place; nested double-wrap merged "
        "recursively via unload_and_optionally_merge_module — Path A "
        "(merge_and_unload + MTP re-injection) verified safe."
    )


# ------------------------------------------------------- safetensors utilities


def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Parse a safetensors file header without loading tensor data."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
    header.pop("__metadata__", None)
    out = {}
    for name, meta in header.items():
        out[name] = {
            "dtype": meta["dtype"],
            "shape": tuple(meta["shape"]),
            "offsets": tuple(meta["data_offsets"]),
            "file": path,
            "data_start": 8 + header_len,
        }
    return out


def model_tensor_map(model_dir: Path) -> dict[str, dict]:
    """name -> {dtype, shape, offsets, file, data_start} for a saved model dir."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = common.read_json(index_path)["weight_map"]
        tmap: dict[str, dict] = {}
        for shard in sorted(set(weight_map.values())):
            shard_path = model_dir / shard
            if not shard_path.exists():
                raise PipelineError(f"Shard listed in index but missing on disk: {shard_path}")
            tmap.update(read_safetensors_header(shard_path))
        missing = set(weight_map) - set(tmap)
        if missing:
            raise PipelineError(
                f"{len(missing)} tensors in index but absent from shards of {model_dir}: "
                f"{sorted(missing)[:5]}..."
            )
        return tmap
    single = model_dir / "model.safetensors"
    if single.exists():
        return read_safetensors_header(single)
    raise PipelineError(
        f"No model.safetensors[.index.json] under {model_dir} — not a saved model dir."
    )


def read_tensor_bytes(meta: dict) -> bytes:
    with open(meta["file"], "rb") as f:
        f.seek(meta["data_start"] + meta["offsets"][0])
        return f.read(meta["offsets"][1] - meta["offsets"][0])


# ------------------------------------------------------------- target analysis


def load_adapter_targets(checkpoint: Path) -> tuple[list | str, list]:
    """Read (target_modules, target_parameters) from adapter_config.json."""
    cfg_path = checkpoint / "adapter_config.json"
    if not cfg_path.exists():
        raise PipelineError(
            f"{cfg_path} not found — --checkpoint must point at a LoRA checkpoint dir "
            "(e.g. outputs/full/<run>/checkpoint-XXXX) containing adapter_config.json."
        )
    acfg = common.read_json(cfg_path)
    modules = acfg.get("target_modules") or []
    params = acfg.get("target_parameters") or []
    return modules, params


def make_target_predicate(target_modules, target_parameters):
    """Predicate: is a base-checkpoint tensor name expected to change on merge?

    Used only to SELECT tensors for the changed/unchanged byte checks — the
    ground truth remains the byte comparison itself.  Conservative direction:
    misclassifying an adapted tensor as untargeted makes the check FAIL LOUDLY
    (never a silent pass).
    """
    params = set(target_parameters or [])
    if isinstance(target_modules, str):
        pattern = re.compile(target_modules)

        def module_hit(mod_path: str) -> bool:
            return bool(pattern.fullmatch(mod_path))
    else:
        modules = set(target_modules or [])

        def module_hit(mod_path: str) -> bool:
            return mod_path.rsplit(".", 1)[-1] in modules

    def is_targeted(name: str) -> bool:
        if MTP_RE.match(name):
            return False  # mtp is never instantiated, so never adapted
        last = name.rsplit(".", 1)[-1]
        if last in params and not name.endswith(".weight"):
            return True  # fused expert nn.Parameter (e.g. experts.gate_up_proj)
        if name.endswith(".weight") and module_hit(name[: -len(".weight")]):
            return True
        return False

    return is_targeted


# ------------------------------------------------------------------ mtp + aux


def inject_mtp(base_dir: Path, out_dir: Path, logger) -> list[str]:
    """Copy every ^mtp\\. tensor from the base shards into out_dir, patching the index.

    WHY: from_pretrained drops mtp.* (VERIFIED-FACTS §0.4); without this the
    published model silently loses its MTP block.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_map = model_tensor_map(base_dir)
    mtp_names = sorted(n for n in base_map if MTP_RE.match(n))
    if not mtp_names:
        raise PipelineError(
            f"Base model at {base_dir} contains no mtp.* tensors — expected 19 per "
            "VERIFIED-FACTS §0.4. Wrong base snapshot?"
        )

    by_file: dict[Path, list[str]] = {}
    for n in mtp_names:
        by_file.setdefault(base_map[n]["file"], []).append(n)
    tensors = {}
    for fpath, names in sorted(by_file.items()):
        with safe_open(str(fpath), framework="pt", device="cpu") as sf:
            for n in names:
                tensors[n] = sf.get_tensor(n)
    save_file(tensors, str(out_dir / MTP_SHARD_NAME), metadata={"format": "pt"})
    mtp_bytes = sum(base_map[n]["offsets"][1] - base_map[n]["offsets"][0] for n in mtp_names)
    logger.info("re-injected %d mtp.* tensors (%.1f MB) into %s",
                len(mtp_names), mtp_bytes / 1e6, MTP_SHARD_NAME)

    index_path = out_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = common.read_json(index_path)
    else:
        # Unsharded save (small max_shard_size not hit, e.g. tiny fixture):
        # normalize to indexed layout so a single unambiguous loading path exists
        # (leaving both model.safetensors AND an index would be ambiguous).
        single = out_dir / "model.safetensors"
        if not single.exists():
            raise PipelineError(f"Neither index nor model.safetensors in {out_dir}")
        hdr = read_safetensors_header(single)
        shard_name = "model-00001-of-00002.safetensors"
        single.rename(out_dir / shard_name)
        index = {
            "metadata": {"total_size": sum(
                m["offsets"][1] - m["offsets"][0] for m in hdr.values())},
            "weight_map": {n: shard_name for n in hdr},
        }
    for n in mtp_names:
        index["weight_map"][n] = MTP_SHARD_NAME
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + mtp_bytes
    common.write_json_atomic(index_path, index)
    return mtp_names


def copy_aux_files(base_dir: Path, out_dir: Path, logger) -> list[str]:
    """Ship tokenizer/processor files from the base snapshot (never clobbering
    files save_pretrained already wrote, notably config.json/generation_config.json)."""
    copied = []
    for pattern in AUX_PATTERNS:
        for src in sorted(base_dir.glob(pattern)):
            if not src.is_file():
                continue
            dst = out_dir / src.name
            if dst.exists():
                logger.info("aux: %s already written by save_pretrained — kept", src.name)
                continue
            shutil.copy2(src, dst)
            copied.append(src.name)
    logger.info("aux files copied from base snapshot: %s", copied or "none")
    return copied


def diff_config_json(base_dir: Path, out_dir: Path, logger) -> dict:
    """Diff base vs merged config.json (merged one comes from save_pretrained —
    do NOT blindly replace it; just record what changed)."""

    def flatten(d, prefix=""):
        out = {}
        for k, v in sorted(d.items()):
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flatten(v, key + "."))
            else:
                out[key] = v
        return out

    base = flatten(common.read_json(base_dir / "config.json"))
    merged = flatten(common.read_json(out_dir / "config.json"))
    diff = {}
    for key in sorted(set(base) | set(merged)):
        b, m = base.get(key, "<absent>"), merged.get(key, "<absent>")
        if b != m:
            diff[key] = {"base": b, "merged": m}
    for key, d in diff.items():
        logger.info("config.json diff: %s: base=%r merged=%r", key, d["base"], d["merged"])
    return diff


# ---------------------------------------------------------------- verification


def verify_merged(base_dir: Path, out_dir: Path, target_modules, target_parameters,
                  logger, seed: int = 42, n_untargeted: int = 5, n_targeted: int = 3) -> dict:
    """Post-merge verification — ALWAYS runs; raises PipelineError on any failure.

    (1) tensor-name-set equality vs base (mtp included) + shape/dtype equality;
    (2) N random UNTARGETED tensors byte-identical to base (plus the router,
        one vision tensor and one mtp tensor, forced into the sample);
    (3) N TARGETED tensors must NOT be identical (a merge that changed nothing
        is the silent failure this pipeline hunts);
    (4) tokenizer/processor aux files present.
    """
    rng = random.Random(seed)
    base_map = model_tensor_map(base_dir)
    out_map = model_tensor_map(out_dir)
    report: dict = {}

    missing = sorted(set(base_map) - set(out_map))
    extra = sorted(set(out_map) - set(base_map))
    if missing or extra:
        raise PipelineError(
            f"Tensor set mismatch vs base: {len(missing)} missing "
            f"(e.g. {missing[:5]}), {len(extra)} extra (e.g. {extra[:5]}). "
            "The merged model would silently differ from the base architecture."
        )
    bad = [n for n in base_map
           if base_map[n]["shape"] != out_map[n]["shape"]
           or base_map[n]["dtype"] != out_map[n]["dtype"]]
    if bad:
        details = {n: (base_map[n]["shape"], base_map[n]["dtype"],
                       out_map[n]["shape"], out_map[n]["dtype"]) for n in bad[:5]}
        raise PipelineError(f"{len(bad)} tensors changed shape/dtype vs base: {details}")
    report["tensor_set"] = {"count": len(base_map), "match": True}
    logger.info("check 1 OK: %d tensor names, shapes and dtypes identical to base", len(base_map))

    is_targeted = make_target_predicate(target_modules, target_parameters)
    targeted = sorted(n for n in base_map if is_targeted(n))
    untargeted = sorted(n for n in base_map if not is_targeted(n))
    if len(targeted) < n_targeted:
        raise PipelineError(
            f"Only {len(targeted)} tensors resolve as LoRA-targeted "
            f"(target_modules={target_modules!r}, target_parameters={target_parameters!r}) "
            "— the adapter targeted (almost) nothing. This is the silent MoE-LoRA "
            "failure mode; refusing to publish."
        )

    forced = [n for n in untargeted if n.endswith(".mlp.gate.weight")][:1]
    forced += [n for n in untargeted if n.startswith("model.visual.")][:1]
    forced += [n for n in untargeted if MTP_RE.match(n)][:1]
    pool = [n for n in untargeted if n not in forced]
    sample_untargeted = forced + rng.sample(pool, min(n_untargeted, len(pool)))
    for n in sample_untargeted:
        if read_tensor_bytes(base_map[n]) != read_tensor_bytes(out_map[n]):
            raise PipelineError(
                f"UNTARGETED tensor {n!r} differs from base after merge — the merge "
                "touched weights it must not touch (router/vision/mtp/frozen params)."
            )
    report["untargeted_byte_identical"] = sample_untargeted
    logger.info("check 2 OK: %d untargeted tensors byte-identical (incl. %s)",
                len(sample_untargeted), forced)

    sample_targeted = rng.sample(targeted, n_targeted)
    unchanged = [n for n in sample_targeted
                 if read_tensor_bytes(base_map[n]) == read_tensor_bytes(out_map[n])]
    if unchanged:
        raise PipelineError(
            f"TARGETED tensors identical to base after merge: {unchanged}. "
            "The LoRA delta was not applied (zero adapter? wrong checkpoint? "
            "peft merge regression). A no-op merge is the silent failure this "
            "pipeline exists to catch — refusing to publish."
        )
    report["targeted_changed"] = sample_targeted
    logger.info("check 3 OK: %d targeted tensors differ from base: %s",
                n_targeted, sample_targeted)

    present, absent = [], []
    for pattern in AUX_PATTERNS:
        base_hits = [p.name for p in base_dir.glob(pattern) if p.is_file()]
        for name in base_hits:
            (present if (out_dir / name).exists() else absent).append(name)
    if absent:
        raise PipelineError(
            f"Aux files present in base snapshot but missing from merged dir: {absent}. "
            "The multimodal processor/tokenizer files must ship (guide §8)."
        )
    if not (out_dir / "config.json").exists():
        raise PipelineError(f"config.json missing from {out_dir}")
    report["aux_files_present"] = sorted(set(present))
    logger.info("check 4 OK: aux files present: %s", sorted(set(present)))
    return report


# ------------------------------------------------------------------ merge core


def ram_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6  # kB -> GB
    raise PipelineError("MemAvailable not found in /proc/meminfo")


def run_merge(base_dir: Path, checkpoint: Path, out_dir: Path, logger,
              dtype_name: str = "bfloat16", max_shard_size: str = "4GB",
              preserve_mtp: bool = True, copy_aux: bool = True,
              allow_no_target_parameters: bool = False) -> dict:
    """Load base -> attach adapter -> merge_and_unload -> save -> re-inject mtp
    -> copy aux -> verify.  Pure function of explicit paths so the tiny-fixture
    self-test exercises the EXACT production code path."""
    import torch

    if torch.cuda.is_available():  # CVD="" makes this False without touching GPUs
        raise PipelineError("CUDA visible despite CUDA_VISIBLE_DEVICES='' — refusing "
                            "(GPUs belong to another team's job).")

    target_modules, target_parameters = load_adapter_targets(checkpoint)
    if not target_parameters and not allow_no_target_parameters:
        raise PipelineError(
            f"adapter_config.json at {checkpoint} has empty target_parameters — the "
            "routed experts were NOT adapted during training (the classic silent MoE "
            "failure). If this is genuinely intended, re-run with "
            "--allow-no-target-parameters."
        )

    timings: dict[str, float] = {}
    t0 = time.time()
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
    )
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as _mod
    if _mod.causal_conv1d_fn is not None or _mod.FusedRMSNormGated is not None:
        raise PipelineError(
            "GPU fast-path kernels active in a CPU-only merge (causal_conv1d/fla) — "
            "the CUDA_VISIBLE_DEVICES guard did not take effect; aborting to avoid "
            "touching the busy GPUs."
        )

    logger.info("loading base model from %s (dtype=%s, cpu)...", base_dir, dtype_name)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        str(base_dir), dtype=getattr(torch, dtype_name), low_cpu_mem_usage=True,
    )
    model.eval()
    timings["load_base_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    from peft import PeftModel

    logger.info("attaching adapter from %s ...", checkpoint)
    peft_model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    logger.info("merging (safe_merge=True: NaN detection on every merged tensor)...")
    merged = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
    timings["merge_s"] = round(time.time() - t0, 1)

    leftover = [n for n, _ in merged.named_modules() if "lora_" in n or "base_layer" in n]
    if leftover:
        raise PipelineError(f"peft wrappers survived merge_and_unload: {leftover[:5]} — "
                            "merged model is not clean.")

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("saving merged model to %s (max_shard_size=%s)...", out_dir, max_shard_size)
    # save_original_format=False is LOAD-BEARING: transformers 5.x registers a
    # qwen2_moe-style WeightConverter for qwen3_5_moe_text (conversion_mapping.py)
    # and by default REVERTS fused expert tensors to legacy per-expert
    # `experts.{i}.gate/up/down_proj.weight` names on save.  The real base repo
    # stores the FUSED format (`experts.gate_up_proj` [256,1024,2048]); reverting
    # would silently publish a different tensor layout than the base and fail the
    # tensor-set-equality verification below.  False == save runtime names as-is.
    merged.save_pretrained(str(out_dir), max_shard_size=max_shard_size,
                           safe_serialization=True, save_original_format=False)
    timings["save_s"] = round(time.time() - t0, 1)
    del merged, peft_model, model

    mtp_names: list[str] = []
    if preserve_mtp:
        mtp_names = inject_mtp(base_dir, out_dir, logger)
    copied = copy_aux_files(base_dir, out_dir, logger) if copy_aux else []
    cfg_diff = diff_config_json(base_dir, out_dir, logger)

    t0 = time.time()
    checks = verify_merged(base_dir, out_dir, target_modules, target_parameters, logger)
    timings["verify_s"] = round(time.time() - t0, 1)

    return {
        "checks": checks,
        "mtp_injected": mtp_names,
        "aux_copied": copied,
        "config_json_diff": cfg_diff,
        "timings_sec": timings,
        "target_modules": target_modules,
        "target_parameters": target_parameters,
    }


# -------------------------------------------------------------------- fixture


def build_tiny_fixture(work: Path, logger, seed: int = 42):
    """Shrunken Qwen3_5Moe (hidden 64, 4 layers [lin,lin,lin,full], 8 experts,
    moe_intermediate 32, tiny vision tower) + fake mtp.* tensors injected into
    the tiny base index — so the production mtp-reinjection path is exercised.

    Returns (base_dir, ckpt_dir, ref_logits, input_ids).
    """
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
    )

    torch.manual_seed(seed)
    text = dict(
        hidden_size=64, num_hidden_layers=4, full_attention_interval=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, intermediate_size=128,
        vocab_size=512, max_position_embeddings=512,
        mtp_num_hidden_layers=1, attn_output_gate=True, partial_rotary_factor=0.25,
        rope_parameters=dict(rope_type="default", rope_theta=10000000,
                             mrope_section=[3, 3, 2], mrope_interleaved=True,
                             partial_rotary_factor=0.25),
        tie_word_embeddings=False,
    )
    vision = dict(depth=2, hidden_size=64, num_heads=2, intermediate_size=128,
                  out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                  temporal_patch_size=2, num_position_embeddings=64, in_channels=3,
                  deepstack_visual_indexes=[])
    cfg = Qwen3_5MoeConfig(text_config=text, vision_config=vision,
                           image_token_id=500, video_token_id=501,
                           vision_start_token_id=502, vision_end_token_id=503)
    model = Qwen3_5MoeForConditionalGeneration(cfg).eval().float()

    base_dir = work / "tiny_base"
    base_dir.mkdir(parents=True, exist_ok=True)
    # small shard size => sharded layout, like the real 26-shard base;
    # save_original_format=False => fused expert tensor names, matching the real
    # base repo format (see the same flag in run_merge for the full rationale).
    model.save_pretrained(str(base_dir), max_shard_size="500KB",
                          safe_serialization=True, save_original_format=False)

    # Fake aux files so the aux-copy path is exercised end-to-end.
    for name in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
                 "preprocessor_config.json", "video_preprocessor_config.json"):
        (base_dir / name).write_text('{"fixture": true}\n', encoding="utf-8")
    (base_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

    # Inject fake mtp.* tensors into the tiny BASE index (mimics the real base
    # repo, whose shards 25/26 carry 19 mtp.* tensors that from_pretrained drops).
    mtp_tensors = {
        "mtp.fc.weight": torch.randn(64, 128),
        "mtp.norm.weight": torch.randn(64),
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.randn(8, 64, 64),
    }
    mtp_file = "model-mtp-base.safetensors"
    save_file(mtp_tensors, str(base_dir / mtp_file), metadata={"format": "pt"})
    index_path = base_dir / "model.safetensors.index.json"
    index = common.read_json(index_path)
    for n, t in mtp_tensors.items():
        index["weight_map"][n] = mtp_file
        index["metadata"]["total_size"] += t.numel() * t.element_size()
    common.write_json_atomic(index_path, index)
    logger.info("tiny base written: %d tensors (+%d fake mtp) at %s",
                len(index["weight_map"]) - len(mtp_tensors), len(mtp_tensors), base_dir)

    # Attach LoRA with the REAL target lists from the frozen train config.
    train_cfg = common.load_yaml("configs/train_full.yaml")["swift_args"]
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=list(train_cfg["target_modules"]),
        target_parameters=list(train_cfg["target_parameters"]),
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)

    # Randomize adapter weights (lora_B is zero-init => merge would be a no-op
    # and the "targeted tensors changed" check would rightly fail).
    n_rand = 0
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                torch.nn.init.normal_(p, std=0.05)
                n_rand += 1
    if n_rand == 0:
        raise PipelineError("fixture: no lora_A/lora_B parameters found after attach")
    logger.info("fixture: randomized %d adapter tensors", n_rand)

    # sanity: expert Parameters really are adapted (ParamWrapper double-wrap)
    wrapped = [n for n, m in peft_model.named_modules()
               if type(m).__name__ == "ParamWrapper"]
    if len(wrapped) < 8:  # 4 layers x 2 targeted parameters
        raise PipelineError(
            f"fixture: expected >=8 ParamWrapper modules (4 layers x 2 fused params), "
            f"got {len(wrapped)}: {wrapped}")

    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, 500, (2, 24))
    with torch.no_grad():
        ref_logits = peft_model(input_ids=input_ids).logits.clone()

    ckpt_dir = work / "tiny_ckpt"
    peft_model.save_pretrained(str(ckpt_dir))
    # peft>=0.17 may save into <dir>/ directly for the default adapter
    if not (ckpt_dir / "adapter_config.json").exists():
        sub = ckpt_dir / "default"
        if (sub / "adapter_config.json").exists():
            ckpt_dir = sub
        else:
            raise PipelineError(f"fixture: adapter_config.json not found under {ckpt_dir}")
    return base_dir, ckpt_dir, ref_logits, input_ids


def self_test(logger) -> None:
    """End-to-end CPU validation of the production merge path on a tiny fixture.

    Proves: mtp re-injection works, untargeted tensors stay byte-identical,
    targeted tensors change, and — the actual LoRA math — forward-pass outputs
    of (base+adapter) vs (merged) agree to atol 1e-4 in fp32.
    """
    import tempfile

    import torch
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
    )

    logger.info(paramwrapper_findings())
    with tempfile.TemporaryDirectory(prefix="keural-merge-selftest-") as tmp:
        work = Path(tmp)
        base_dir, ckpt_dir, ref_logits, input_ids = build_tiny_fixture(work, logger)
        out_dir = work / "tiny_merged"

        result = run_merge(base_dir, ckpt_dir, out_dir, logger,
                           dtype_name="float32", max_shard_size="500KB")
        if len(result["mtp_injected"]) != 3:
            raise PipelineError(f"self-test: expected 3 injected mtp tensors, "
                                f"got {result['mtp_injected']}")

        merged = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            str(out_dir), dtype=torch.float32).eval()
        with torch.no_grad():
            merged_logits = merged(input_ids=input_ids).logits

        diff = (merged_logits - ref_logits).abs().max().item()
        scale = ref_logits.abs().max().item()
        logger.info("self-test: forward-pass max|peft - merged| = %.3e "
                    "(logit scale max|ref| = %.3f, atol 1e-4, fp32)", diff, scale)
        if diff > 1e-4:
            raise PipelineError(f"self-test FAILED: merged forward deviates from peft "
                                f"forward by {diff:.3e} > 1e-4")

        base = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            str(base_dir), dtype=torch.float32).eval()
        with torch.no_grad():
            base_logits = base(input_ids=input_ids).logits
        base_diff = (merged_logits - base_logits).abs().max().item()
        logger.info("self-test: max|base - merged| = %.3e (must be >1e-3: the "
                    "adapter must actually change the model)", base_diff)
        if base_diff < 1e-3:
            raise PipelineError("self-test FAILED: merged model is numerically "
                                "identical to base — no-op merge")
        logger.info("SELF-TEST PASSED: merge path verified end-to-end on tiny fixture "
                    "(equivalence %.3e <= 1e-4, base deviation %.3e, mtp preserved)",
                    diff, base_diff)


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None,
                        help="LoRA checkpoint dir (overrides cfg merge.checkpoint)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print plan + peft ParamWrapper findings, do nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-merge over an existing merged_dir")
    parser.add_argument("--self-test", action="store_true",
                        help="run the tiny-fixture CPU validation and exit")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--min-free-ram-gb", type=float, default=160.0,
                        help="required MemAvailable before a real merge (~70GB model "
                             "held twice transiently + shard buffers)")
    parser.add_argument("--allow-no-target-parameters", action="store_true")
    args = parser.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("merge", cfg)
    common.stage_banner(logger, "merge_lora", args.config)
    common.guard_environment(logger)

    try:
        if args.self_test:
            self_test(logger)
            return

        mcfg = cfg.get("merge") or {}
        checkpoint = args.checkpoint or mcfg.get("checkpoint")
        merged_dir = common.resolve_path(cfg, mcfg.get("merged_dir", "outputs/merged"))
        model_id = cfg["pipeline"]["model"]["id"]
        revision = cfg["pipeline"]["model"]["revision"]

        if not checkpoint:
            if args.dry_run:
                # A plan preview must work before any training checkpoint exists.
                logger.info("[dry-run] peft findings: %s", paramwrapper_findings())
                logger.info("[dry-run] plan: base=%s@%s checkpoint=<UNSET — set "
                            "merge.checkpoint or pass --checkpoint> -> %s",
                            model_id, revision, merged_dir)
                logger.info("[dry-run] disk free: %.0f GB (need %s GB); RAM available: "
                            "%.0f GB (need %.0f GB)",
                            shutil.disk_usage(common.project_root()).free / 1e9,
                            cfg["pipeline"]["runtime"]["min_free_disk_gb"].get("merge"),
                            ram_available_gb(), args.min_free_ram_gb)
                return
            raise PipelineError(
                "No checkpoint: set merge.checkpoint in configs/publish.yaml or pass "
                "--checkpoint outputs/full/<run>/checkpoint-XXXX")
        checkpoint = common.resolve_path(cfg, checkpoint)
        adapter_file = checkpoint / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise PipelineError(f"{adapter_file} not found — not a LoRA checkpoint dir.")

        fingerprint = common.config_fingerprint({
            "model_id": model_id, "revision": revision,
            "checkpoint": str(checkpoint),
            "adapter_sha256": common.sha256_file(adapter_file),
            "preserve_mtp": bool(mcfg.get("preserve_mtp", True)),
            "copy_aux_files": bool(mcfg.get("copy_aux_files", True)),
            "max_shard_size": args.max_shard_size, "dtype": "bfloat16",
        })

        report_path = merged_dir / "merge_report.json"
        if report_path.exists() and not args.force:
            prev = common.read_json(report_path)
            if prev.get("fingerprint") == fingerprint and prev.get("status") == "ok":
                logger.info("merge already completed for this exact checkpoint+config "
                            "(fingerprint %s) — no-op. Use --force to re-merge.", fingerprint)
                return
            raise PipelineError(
                f"{merged_dir} holds a DIFFERENT merge (fingerprint "
                f"{prev.get('fingerprint')} != {fingerprint}). Re-run with --force to "
                "replace it, or point merge.merged_dir elsewhere.")
        if merged_dir.exists() and any(merged_dir.iterdir()) and not report_path.exists() \
                and not args.force:
            raise PipelineError(
                f"{merged_dir} exists without a merge_report.json (interrupted or foreign "
                "contents). Re-run with --force to replace it.")

        if args.dry_run:
            logger.info("[dry-run] peft findings: %s", paramwrapper_findings())
            logger.info("[dry-run] plan: base=%s@%s checkpoint=%s -> %s "
                        "(dtype=bfloat16, max_shard_size=%s, preserve_mtp=%s, "
                        "copy_aux=%s, fingerprint=%s)",
                        model_id, revision, checkpoint, merged_dir,
                        args.max_shard_size, mcfg.get("preserve_mtp", True),
                        mcfg.get("copy_aux_files", True), fingerprint)
            logger.info("[dry-run] disk free: %.0f GB (need %s GB); RAM available: %.0f GB "
                        "(need %.0f GB)", shutil.disk_usage(common.project_root()).free / 1e9,
                        cfg["pipeline"]["runtime"]["min_free_disk_gb"].get("merge"),
                        ram_available_gb(), args.min_free_ram_gb)
            return

        common.assert_free_disk(cfg, "merge", logger)
        avail = ram_available_gb()
        if avail < args.min_free_ram_gb:
            raise PipelineError(
                f"Only {avail:.0f} GB RAM available, CPU merge needs "
                f"~{args.min_free_ram_gb:.0f} GB (70 GB bf16 model + save buffers). "
                "Free RAM or lower --min-free-ram-gb at your own risk.")
        logger.info("RAM available: %.0f GB (>= %.0f GB required)", avail, args.min_free_ram_gb)
        logger.info(paramwrapper_findings())

        from huggingface_hub import snapshot_download

        logger.info("resolving base snapshot %s@%s (downloads ~70 GB if not cached)...",
                    model_id, revision)
        base_dir = Path(snapshot_download(model_id, revision=revision))

        # Work in a .partial dir: a killed run never leaves a half-written
        # merged_dir that publish.py could mistake for a finished merge.
        work_dir = merged_dir.parent / (merged_dir.name + ".partial")
        if work_dir.exists():
            logger.warning("removing stale partial merge at %s", work_dir)
            shutil.rmtree(work_dir)
        if merged_dir.exists() and args.force:
            logger.warning("--force: removing existing %s", merged_dir)
            shutil.rmtree(merged_dir)

        result = run_merge(
            base_dir, checkpoint, work_dir, logger,
            dtype_name="bfloat16", max_shard_size=args.max_shard_size,
            preserve_mtp=bool(mcfg.get("preserve_mtp", True)),
            copy_aux=bool(mcfg.get("copy_aux_files", True)),
            allow_no_target_parameters=args.allow_no_target_parameters,
        )
        report = {
            "status": "ok",
            "fingerprint": fingerprint,
            "created_utc": common.utc_stamp(),
            "base_model": model_id,
            "base_revision": revision,
            "base_path": str(base_dir),
            "checkpoint": str(checkpoint),
            "adapter_sha256": common.sha256_file(adapter_file),
            "dtype": "bfloat16",
            "peft_paramwrapper_findings": paramwrapper_findings(),
            **result,
        }
        common.write_json_atomic(work_dir / "merge_report.json", report)
        merged_dir.parent.mkdir(parents=True, exist_ok=True)
        work_dir.rename(merged_dir)
        logger.info("MERGE OK -> %s (report: %s)", merged_dir, merged_dir / "merge_report.json")
    except PipelineError as exc:
        common.die(logger, str(exc))


if __name__ == "__main__":
    main()
