#!/usr/bin/env python
"""Router/expert LoRA verification gate — the raison d'être of this pipeline.

MoE LoRA on Qwen3.6-35B-A3B fails SILENTLY when misconfigured: the routed experts are
fused nn.Parameter tensors (not nn.Linear), so plain --target_modules never touches
them — training runs, loss decreases, checkpoints save, and most of the model's
capacity is never adapted. Conversely, accidentally adapting the router
(`mlp.gate`) or `shared_expert_gate` corrupts routing. This script hard-verifies
the *resolved* adapter placement instead of trusting name-suffix semantics
(VERIFIED-FACTS.md §2 name-collision hazard, §5 gate spec).

Two modes, both CPU/meta-device ONLY — never loads weights, never touches GPUs:

  --pre --train-config configs/train_pilot.yaml
      Reads the LoRA targeting fields from the YAML (never hardcoded, so any config
      edit is what actually gets verified), instantiates the REAL 40-layer
      Qwen3_5MoeForConditionalGeneration on torch.device('meta') (free: meta tensors
      have no storage), attaches peft LoRA exactly as configured, and asserts the
      resulting trainable-parameter set.

  --checkpoint <dir>
      Re-runs the same structural assertions using the checkpoint's own
      adapter_config.json (catches ms-swift translating our YAML into something
      else), then additionally opens adapter_model.safetensors (safe_open — key
      metadata only, no tensor loads) and asserts the actually-saved key set.

All checks are evaluated before exiting so a broken config reports every violation
at once. Exit 0 only if ALL checks pass. Human summary table + JSON report go to
stdout; the JSON report is also written to logs/verify/<stamp>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Belt and braces: this process must never see a GPU even if a caller leaks
# CUDA_VISIBLE_DEVICES (another team's job owns the GPUs — VERIFIED-FACTS §1).
# Must happen before torch is imported anywhere below.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

STAGE = "verify"
DEFAULT_TRAIN_CONFIG = "configs/train_pilot.yaml"

# The three shared-expert projections that must be adapted on every layer (spec f).
SHARED_EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")

# --- name classification -----------------------------------------------------
# Order matters: violations are matched first so e.g. an adapted router would
# never be mis-binned. `mlp.gate.` (trailing dot) cannot match `gate_proj`;
# `.mlp.shared_expert.` (trailing dot) cannot match `shared_expert_gate`.
ROUTER_RE = re.compile(r"mlp\.gate\.")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
EXPERT_LORA_RE = re.compile(r"mlp\.experts\..*lora")

VIOLATION_BUCKETS = (
    "VIOLATION_router",
    "VIOLATION_shared_expert_gate",
    "VIOLATION_visual",
    "VIOLATION_embed_tokens",
    "VIOLATION_lm_head",
)


def classify(name: str) -> str:
    """Bucket a trainable-parameter name / saved-adapter key.

    Works on both live peft names (…lora_A.default.weight) and safetensors keys
    (…lora_A.weight) because it only relies on stable substrings.
    """
    if ROUTER_RE.search(name):
        return "VIOLATION_router"
    if "shared_expert_gate" in name:
        return "VIOLATION_shared_expert_gate"
    if "visual" in name:
        return "VIOLATION_visual"
    if "embed_tokens" in name:
        return "VIOLATION_embed_tokens"
    if "lm_head" in name:
        return "VIOLATION_lm_head"
    if EXPERT_LORA_RE.search(name):
        return "experts"
    if ".self_attn." in name and "lora" in name:
        return "self_attn"
    if ".linear_attn." in name and "lora" in name:
        return "linear_attn"
    if ".mlp.shared_expert." in name and "lora" in name:
        return "shared_expert"
    return "UNEXPECTED"


def layer_of(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


# --- model/topology helpers ---------------------------------------------------


def load_pinned_text_topology(cfg: dict, logger) -> tuple[object, set[int], set[int], int]:
    """AutoConfig for the pinned model revision -> (config, full_attn, linear_attn, n_layers).

    The expected attention split is DERIVED from config.text_config.layer_types
    (never hardcoded) so a model-revision bump is what actually gets verified.
    Tries the local HF cache first: the snapshot is pinned and network may be
    unavailable/slow; falls back to the Hub with the same pinned revision.
    """
    from transformers import AutoConfig

    model_id = cfg["pipeline"]["model"]["id"]
    revision = cfg["pipeline"]["model"]["revision"]
    try:
        conf = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
        logger.info("model config loaded from local HF cache (%s @ %s)", model_id, revision)
    except Exception as e:  # noqa: BLE001 — any cache miss falls through to network
        logger.warning("local HF cache miss (%s); fetching from Hub with pinned revision", e)
        conf = AutoConfig.from_pretrained(model_id, revision=revision)

    tc = conf.text_config
    layer_types = getattr(tc, "layer_types", None)
    if not layer_types:
        raise common.PipelineError(
            f"config.text_config.layer_types missing for {model_id}@{revision} — cannot "
            "derive the full/linear attention split. Model config changed; update the gate."
        )
    n_layers = tc.num_hidden_layers
    if len(layer_types) != n_layers:
        raise common.PipelineError(
            f"layer_types has {len(layer_types)} entries but num_hidden_layers={n_layers}"
        )
    full_attn = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    linear_attn = {i for i, t in enumerate(layer_types) if t == "linear_attention"}
    unknown = set(range(n_layers)) - full_attn - linear_attn
    if unknown:
        raise common.PipelineError(
            f"layer_types contains unrecognized types at layers {sorted(unknown)}: "
            f"{sorted({layer_types[i] for i in unknown})} — update the gate for this model."
        )
    return conf, full_attn, linear_attn, n_layers


def attach_lora_on_meta(conf, lora: dict, logger) -> list[tuple[str, int]]:
    """Instantiate the REAL model class on the meta device and attach peft LoRA.

    Meta tensors have no storage, so the full 40-layer 35B model costs ~nothing;
    this is the only way to verify the true resolved adapter placement without
    loading 70 GB of weights or touching the (busy) GPUs.

    Returns [(trainable_param_name, numel), ...].
    """
    import torch
    from peft import LoraConfig, get_peft_model

    try:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    except ImportError as e:
        raise common.PipelineError(
            f"transformers has no qwen3_5_moe/Qwen3_5MoeForConditionalGeneration ({e}) — "
            "wrong transformers version? Pipeline pins 5.12.1 (VERIFIED-FACTS §1)."
        ) from e

    logger.info("instantiating %s on meta device (full config, no weights)...",
                "Qwen3_5MoeForConditionalGeneration")
    with torch.device("meta"):
        model = Qwen3_5MoeForConditionalGeneration._from_config(conf)

    peft_kwargs = dict(
        r=int(lora["lora_rank"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        target_modules=lora["target_modules"],
        target_parameters=lora["target_parameters"] or None,
    )
    if lora.get("modules_to_save"):
        peft_kwargs["modules_to_save"] = lora["modules_to_save"]
    logger.info("attaching peft LoRA: %s", json.dumps(peft_kwargs, default=str))
    peft_model = get_peft_model(model, LoraConfig(**peft_kwargs))
    return [(n, p.numel()) for n, p in peft_model.named_parameters() if p.requires_grad]


# --- check engine --------------------------------------------------------------


def run_checks(names: list[str], numels: dict[str, int] | None, lora: dict,
               full_attn: set[int], linear_attn: set[int], n_layers: int,
               kind: str) -> dict:
    """Evaluate ALL gate assertions over a set of adapter names.

    `names` is either the trainable-parameter names of a live peft attach
    (kind='structural') or the key set of a saved adapter_model.safetensors
    (kind='weights'). Every check is evaluated — no early exit — so one run
    reports every violation.
    """
    all_layers = set(range(n_layers))
    buckets: dict[str, dict] = {}
    for n in names:
        b = buckets.setdefault(classify(n), {"tensors": 0, "params": 0, "layers": set(),
                                             "examples": []})
        b["tensors"] += 1
        b["params"] += (numels or {}).get(n, 0)
        ly = layer_of(n)
        if ly is not None:
            b["layers"].add(ly)
        if len(b["examples"]) < 3:
            b["examples"].append(n)

    def bucket_layers(name: str) -> set[int]:
        return buckets.get(name, {}).get("layers", set())

    checks: list[dict] = []

    def check(cid: str, desc: str, passed: bool, detail: str) -> None:
        checks.append({"id": cid, "desc": desc, "passed": bool(passed), "detail": detail})

    # a. router must be untouched — THE #1 silent failure mode.
    bad = buckets.get("VIOLATION_router", {}).get("examples", [])
    check("a_router_frozen", "no adapted/trainable param matches r'mlp\\.gate\\.'",
          "VIOLATION_router" not in buckets,
          "clean" if not bad else f"ROUTER ADAPTED: e.g. {bad}")

    # b. shared_expert_gate must be untouched.
    bad = buckets.get("VIOLATION_shared_expert_gate", {}).get("examples", [])
    check("b_shared_expert_gate_frozen", "no name contains 'shared_expert_gate'",
          "VIOLATION_shared_expert_gate" not in buckets,
          "clean" if not bad else f"shared_expert_gate adapted: e.g. {bad}")

    # c. routed experts adapted on ALL layers (the silent-failure victim), and one
    #    adapter (ParamWrapper) per targeted parameter per layer. peft double-wraps
    #    naming: experts.lora_* AND experts.base_layer.lora_* both exist
    #    (VERIFIED-FACTS 'extra finds') — both count toward the same layer.
    exp_layers = bucket_layers("experts")
    missing = sorted(all_layers - exp_layers)
    check("c_experts_all_layers",
          f"routed-expert LoRA (r'mlp\\.experts\\..*lora') present on all {n_layers} layers",
          exp_layers == all_layers,
          "all layers covered" if exp_layers == all_layers
          else f"MISSING on layers {missing} (experts NOT adapted — the silent failure)")
    n_tp = len(lora["target_parameters"] or [])
    if n_tp:
        per_layer_a: dict[int, int] = {i: 0 for i in all_layers}
        for n in names:
            if EXPERT_LORA_RE.search(n) and "lora_A" in n:
                ly = layer_of(n)
                if ly is not None:
                    per_layer_a[ly] += 1
        wrong = {i: c for i, c in per_layer_a.items() if c != n_tp}
        check("c2_experts_wrapper_count",
              f"exactly {n_tp} expert lora_A wrapper(s)/layer (= len(target_parameters))",
              not wrong,
              "counts match" if not wrong else f"per-layer lora_A count wrong: {wrong}")

    # d. self_attn adapted on EXACTLY the full-attention layers (derived from
    #    config.text_config.layer_types — the guide's 1-in-4 hybrid split).
    sa = bucket_layers("self_attn")
    check("d_self_attn_layers",
          f"self_attn LoRA on exactly the {len(full_attn)} full-attention layers",
          sa == full_attn,
          f"exact match {sorted(sa)}" if sa == full_attn
          else f"expected {sorted(full_attn)}, got {sorted(sa)} "
               f"(missing {sorted(full_attn - sa)}, extra {sorted(sa - full_attn)})")

    # e. linear_attn (Gated DeltaNet) adapted on EXACTLY the complement.
    la = bucket_layers("linear_attn")
    check("e_linear_attn_layers",
          f"linear_attn LoRA on exactly the {len(linear_attn)} linear-attention layers",
          la == linear_attn,
          "exact match" if la == linear_attn
          else f"expected {sorted(linear_attn)}, got {sorted(la)} "
               f"(missing {sorted(linear_attn - la)}, extra {sorted(la - linear_attn)})")

    # f. shared_expert gate/up/down adapted on all layers.
    se_ok = True
    se_details = []
    for proj in SHARED_EXPERT_PROJS:
        lys = {layer_of(n) for n in names
               if f".mlp.shared_expert.{proj}." in n and "lora" in n} - {None}
        if lys != all_layers:
            se_ok = False
            se_details.append(f"{proj}: missing layers {sorted(all_layers - lys)}")
    check("f_shared_expert_all_layers",
          f"shared_expert {'/'.join(SHARED_EXPERT_PROJS)} LoRA on all {n_layers} layers",
          se_ok, "all covered" if se_ok else "; ".join(se_details))

    # g. vision tower / embeddings / lm_head untouched.
    for cid, bucket, label in (("g1_visual_frozen", "VIOLATION_visual", "model.visual"),
                               ("g2_embed_frozen", "VIOLATION_embed_tokens", "embed_tokens"),
                               ("g3_lm_head_frozen", "VIOLATION_lm_head", "lm_head")):
        bad = buckets.get(bucket, {}).get("examples", [])
        check(cid, f"nothing trainable under {label}", bucket not in buckets,
              "clean" if not bad else f"{label} adapted: e.g. {bad}")

    # h. peft ParamWrapper raises ValueError when lora_dropout != 0 and
    #    target_parameters is used (VERIFIED-FACTS §5; ms-swift default 0.05 crashes).
    dropout = float(lora["lora_dropout"])
    h_ok = (not n_tp) or dropout == 0.0
    check("h_dropout_zero_with_target_parameters",
          "lora_dropout == 0.0 whenever target_parameters is non-empty",
          h_ok,
          f"dropout={dropout}, target_parameters={lora['target_parameters']}"
          + ("" if h_ok else " — peft ParamWrapper WILL crash at model prep; set lora_dropout: 0.0"))

    # j. catch-all: no trainable name outside the expected buckets (catches e.g.
    #    modules_to_save surprises that the named checks would miss).
    bad = buckets.get("UNEXPECTED", {}).get("examples", [])
    check("j_no_unexpected_trainables", "no adapted names outside the expected buckets",
          "UNEXPECTED" not in buckets,
          "clean" if not bad else f"unexpected trainables: e.g. {bad} "
                                  f"({buckets['UNEXPECTED']['tensors']} tensors)")

    return {
        "kind": kind,
        "n_names": len(names),
        "trainable_params_total": sum((numels or {}).values()) if numels else None,
        "buckets": {k: {"tensors": v["tensors"], "params": v["params"],
                        "n_layers": len(v["layers"]), "examples": v["examples"]}
                    for k, v in sorted(buckets.items())},
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }


# --- lora-field extraction ------------------------------------------------------


def lora_fields_from_yaml(cfg: dict) -> dict:
    """Read the LoRA targeting fields from the train YAML — never hardcoded.

    Missing keys are fatal: ms-swift defaults are dangerous here (target_modules
    defaults to all-linear, lora_dropout defaults to 0.05 which crashes with
    target_parameters), so the config must be explicit.
    """
    sa = cfg.get("swift_args")
    if not isinstance(sa, dict):
        raise common.PipelineError("train config has no swift_args mapping")
    required = ("target_modules", "lora_rank", "lora_alpha", "lora_dropout")
    missing = [k for k in required if k not in sa]
    if missing:
        raise common.PipelineError(
            f"swift_args missing {missing} — the gate refuses to verify implicit "
            "ms-swift defaults (all-linear targeting / dropout 0.05 are both hazards); "
            "set them explicitly in the train YAML."
        )
    return {
        "target_modules": list(sa["target_modules"]),
        "target_parameters": list(sa.get("target_parameters") or []),
        "lora_rank": int(sa["lora_rank"]),
        "lora_alpha": int(sa["lora_alpha"]),
        "lora_dropout": float(sa["lora_dropout"]),
        "modules_to_save": list(sa.get("modules_to_save") or []),
    }


def lora_fields_from_adapter_config(path: Path, cfg: dict, logger) -> dict:
    """Read the peft adapter_config.json ms-swift saved into the checkpoint.

    This is what ms-swift ACTUALLY handed to peft — verifying it (rather than our
    YAML) catches ms-swift translating/augmenting our arguments.
    """
    if not path.is_file():
        raise common.PipelineError(
            f"{path} not found — not a LoRA checkpoint dir? Expected the peft "
            "adapter_config.json that ms-swift writes next to adapter_model.safetensors."
        )
    try:
        ac = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise common.PipelineError(f"{path} is corrupt/not JSON: {e}") from e

    for k in ("r", "lora_alpha", "target_modules"):
        if k not in ac:
            raise common.PipelineError(f"{path} missing required peft field {k!r}")

    base = ac.get("base_model_name_or_path", "")
    model_id = cfg["pipeline"]["model"]["id"]
    if base and model_id not in base:
        logger.warning("adapter_config base_model_name_or_path=%r does not mention "
                       "pipeline model id %r (local snapshot path?) — continuing, but check.",
                       base, model_id)

    tm = ac["target_modules"]
    return {
        # peft may store target_modules as list OR a single regex string; both are
        # valid LoraConfig inputs, pass through unchanged.
        "target_modules": tm if isinstance(tm, str) else list(tm),
        "target_parameters": list(ac.get("target_parameters") or []),
        "lora_rank": int(ac["r"]),
        "lora_alpha": int(ac["lora_alpha"]),
        "lora_dropout": float(ac.get("lora_dropout", 0.0)),
        "modules_to_save": list(ac.get("modules_to_save") or []),
    }


def saved_adapter_keys(ckpt: Path) -> list[str]:
    """Key set of adapter_model.safetensors — metadata only, no tensor loads."""
    st = ckpt / "adapter_model.safetensors"
    if not st.is_file():
        hint = ""
        if (ckpt / "adapter_model.bin").is_file():
            hint = (" (found legacy adapter_model.bin instead — ms-swift should save "
                    "safetensors; was safe_serialization disabled?)")
        raise common.PipelineError(f"{st} not found{hint}")
    from safetensors import safe_open

    try:
        with safe_open(str(st), framework="pt", device="cpu") as f:
            return list(f.keys())
    except Exception as e:  # noqa: BLE001 — safetensors raises bare Exception subtypes
        raise common.PipelineError(f"{st} unreadable/corrupt: {e}") from e


# --- reporting -------------------------------------------------------------------


def print_summary(report: dict) -> None:
    """Human summary table (stdout) — buckets then checks, then verdict."""
    for section in report["sections"]:
        print(f"\n--- {section['kind']} ({section['n_names']} adapter names"
              + (f", {section['trainable_params_total']:,} trainable params"
                 if section["trainable_params_total"] else "") + ") ---")
        print(f"{'bucket':34} {'tensors':>8} {'params':>14} {'layers':>7}")
        for name, b in section["buckets"].items():
            print(f"{name:34} {b['tensors']:>8} {b['params']:>14,} {b['n_layers']:>7}")
        print(f"{'check':44} {'result':7} detail")
        for c in section["checks"]:
            mark = "PASS" if c["passed"] else "FAIL"
            print(f"{c['id']:44} {mark:7} {c['detail']}")
    print(f"\nVERDICT: {'PASS' if report['passed'] else 'FAIL'} "
          f"({report['mode']} mode, model {report['model']['id']}@{report['model']['revision'][:12]})")


def finalize(report: dict, cfg: dict, logger, json_out: str | None) -> int:
    stamp = common.utc_stamp()
    report["timestamp_utc"] = stamp
    logs_dir = common.resolve_path(cfg, cfg["pipeline"]["paths"]["logs_dir"]) / "verify"
    report_path = logs_dir / f"{stamp}.json"
    common.write_json_atomic(report_path, report)
    if json_out:
        common.write_json_atomic(common.resolve_path(cfg, json_out), report)
    print_summary(report)
    print("\n=== JSON REPORT ===")
    print(json.dumps(report, indent=2, default=str))
    logger.info("report written: %s", report_path)
    if not report["passed"]:
        failed = [c["id"] for s in report["sections"] for c in s["checks"] if not c["passed"]]
        logger.error("VERIFICATION FAILED — failed checks: %s. Do NOT train on this "
                     "configuration/checkpoint.", failed)
        return 1
    logger.info("all checks passed")
    return 0


# --- modes -----------------------------------------------------------------------


def mode_pre(train_config: str, json_out: str | None) -> int:
    cfg = common.load_config(train_config)
    logger = common.setup_logging(STAGE, cfg)
    common.stage_banner(logger, "verify_lora_targets --pre", train_config)
    common.guard_environment(logger)

    lora = lora_fields_from_yaml(cfg)
    conf, full_attn, linear_attn, n_layers = load_pinned_text_topology(cfg, logger)

    # Check (h) BEFORE attach so a bad dropout is OUR actionable error, not a peft
    # traceback; then attach with dropout forced to 0 so the structural checks
    # still all run (spec: every failure listed in one pass).
    attach_lora = dict(lora)
    dropout_hazard = lora["target_parameters"] and lora["lora_dropout"] != 0.0
    if dropout_hazard:
        logger.error("lora_dropout=%s with non-empty target_parameters — peft ParamWrapper "
                     "would crash; attaching with dropout=0.0 to complete the remaining "
                     "checks (check h will FAIL).", lora["lora_dropout"])
        attach_lora["lora_dropout"] = 0.0

    trainable = attach_lora_on_meta(conf, attach_lora, logger)
    names = [n for n, _ in trainable]
    numels = dict(trainable)
    section = run_checks(names, numels, lora, full_attn, linear_attn, n_layers,
                         kind="structural (meta-device peft attach)")

    report = {
        "mode": "pre",
        "train_config": str(train_config),
        "model": dict(cfg["pipeline"]["model"]),
        "lora": lora,
        "expected_layers": {"total": n_layers,
                            "full_attention": sorted(full_attn),
                            "linear_attention": sorted(linear_attn)},
        "sections": [section],
        "passed": section["passed"],
    }
    return finalize(report, cfg, logger, json_out)


def mode_checkpoint(checkpoint: str, train_config: str, json_out: str | None) -> int:
    cfg = common.load_config(train_config)
    logger = common.setup_logging(STAGE, cfg)
    common.stage_banner(logger, "verify_lora_targets --checkpoint", checkpoint)
    common.guard_environment(logger)

    ckpt = common.resolve_path(cfg, checkpoint)
    if not ckpt.is_dir():
        raise common.PipelineError(f"checkpoint dir not found: {ckpt}")

    lora = lora_fields_from_adapter_config(ckpt / "adapter_config.json", cfg, logger)
    conf, full_attn, linear_attn, n_layers = load_pinned_text_topology(cfg, logger)

    # Structural re-attach from the CHECKPOINT's own adapter_config (what ms-swift
    # actually gave peft), with the same dropout-hazard handling as --pre.
    attach_lora = dict(lora)
    if lora["target_parameters"] and lora["lora_dropout"] != 0.0:
        logger.error("checkpoint adapter_config has lora_dropout=%s with target_parameters "
                     "— check h will FAIL; attaching with 0.0 to complete the other checks.",
                     lora["lora_dropout"])
        attach_lora["lora_dropout"] = 0.0
    trainable = attach_lora_on_meta(conf, attach_lora, logger)
    structural = run_checks([n for n, _ in trainable], dict(trainable), lora,
                            full_attn, linear_attn, n_layers,
                            kind="structural (re-attach from adapter_config.json)")

    # Saved-weights key check: what is ACTUALLY in the checkpoint file. Handles the
    # ParamWrapper double-wrap naming (experts.base_layer.lora_* AND experts.lora_*).
    keys = saved_adapter_keys(ckpt)
    weights = run_checks(keys, None, lora, full_attn, linear_attn, n_layers,
                         kind="weights (adapter_model.safetensors key set)")

    report = {
        "mode": "checkpoint",
        "checkpoint": str(ckpt),
        "model": dict(cfg["pipeline"]["model"]),
        "lora": lora,
        "expected_layers": {"total": n_layers,
                            "full_attention": sorted(full_attn),
                            "linear_attention": sorted(linear_attn)},
        "sections": [structural, weights],
        "passed": structural["passed"] and weights["passed"],
    }
    return finalize(report, cfg, logger, json_out)


# --- cli -------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Router/expert LoRA verification gate (CPU/meta-device only; "
                    "never loads weights, never touches GPUs).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pre", action="store_true",
                      help="verify the LoRA targeting declared in --train-config")
    mode.add_argument("--checkpoint", metavar="DIR",
                      help="verify a saved ms-swift/peft LoRA checkpoint dir")
    ap.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG,
                    help=f"train stage YAML (default: {DEFAULT_TRAIN_CONFIG}); in "
                         "--checkpoint mode only used for pipeline model pin + paths")
    ap.add_argument("--json-out", default=None,
                    help="additional path to write the JSON report to (for callers)")
    args = ap.parse_args()

    try:
        if args.pre:
            return mode_pre(args.train_config, args.json_out)
        return mode_checkpoint(args.checkpoint, args.train_config, args.json_out)
    except common.PipelineError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
