#!/usr/bin/env python3
"""Stage 1 — build the Keural SFT dataset (train/eval/pilot JSONL in ms-swift schema).

What this stage does (guide §2-§4, VERIFIED-FACTS §4):
  1. For every source in configs/dataset.yaml, fetch its JSONL files from the HF Hub
     at the PINNED revision (reproducibility: revisions are full commit SHAs).
  2. Stream-parse rows {"text","source_name","license","n_tokens"}, apply license /
     source_name / length filters, parse the pre-rendered ChatML "text" back into
     structured {"messages": [...]} (ms-swift applies the chat template itself —
     feeding it raw ChatML strings would double-wrap the template, silently).
  3. Prepend the empty-think scaffold to EVERY assistant turn (VERIFIED-FACTS §4:
     source rows carry no <think> blocks; prefixing all turns makes rendered output
     match the tokenizer's enable_thinking=false path exactly, and ms-swift's
     `--loss_scale ignore_empty_think` zero-masks the scaffold at loss time).
  4. Deterministic per-source reservoir sampling, cross-source dedup, seeded
     shuffle, stratified eval split, and a proportional pilot slice of TRAIN.
  5. Write a manifest (fingerprint, per-source accounting, per-reason drop counts,
     output sha256s, license summary for the model card) + a stats file.

Why the paranoid accounting: MoE LoRA fails SILENTLY when misconfigured — the same
discipline applies to data. Every dropped row is counted by reason; nothing is
swallowed. Any structural surprise raises PipelineError with an actionable message.

Idempotency / resume model: outputs are written to <name>.tmp then os.replace()d;
the manifest is written LAST. A killed run leaves no manifest and restarts cleanly;
a completed run is a cheap no-op (fingerprint + output sha256 check) unless --force.

Smoke mode (--smoke): caps every source at SMOKE_CAP post-filter rows and reads only
what is needed — cached files are read locally and abandoned early; non-cached files
are STREAMED via HfFileSystem range-reads at the pinned revision (no full downloads).
Outputs go to data/processed/smoke/ with their own manifest, never touching the real
outputs.

GPU note: this stage is CPU-only by design. It never imports torch and never touches
CUDA (another team's job owns the GPUs — VERIFIED-FACTS §1).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# The interactive shell exports HF_HUB_ENABLE_HF_TRANSFER, which makes hub 1.x emit
# a FutureWarning at import time. Dropping it (this process only) keeps logs clean;
# transfer behavior is unchanged (hf_transfer is unused in hub 1.x either way).
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import PipelineError

try:
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
except ImportError as exc:  # pragma: no cover - env misconfiguration path
    sys.stderr.write(
        f"FATAL: cannot import huggingface_hub ({exc}). Run inside the project venv:\n"
        '  source "/home/work/keural MoE Pro/venv/bin/activate" && export PYTHONPATH=\n'
    )
    sys.exit(1)

STAGE = "dataset"
SMOKE_CAP = 300  # per-source post-filter row cap in --smoke mode
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VALID_ROLES = ("system", "user", "assistant")
# n_tokens histogram bucket edges (upper-open); rows above max_tokens are dropped
# upstream, so the top bucket only catches boundary rows / company data.
TOKEN_BUCKET_EDGES = (0, 256, 512, 1024, 2048, 4096, 8192)
MAX_DEBUG_EXAMPLES = 3  # per (source, drop-reason) logged at DEBUG


class RowDrop(Exception):
    """A row failed validation/filtering. .reason is the per-source counter key."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class Record:
    """One accepted training example plus the metadata the manifest/stats need."""

    messages: list[dict]
    source: str          # pipeline source name (cfg sources[i].name), used to stratify
    source_name: str     # row-level provenance field from the dataset itself
    license: str
    n_tokens: int | None


@dataclass
class SourceState:
    """Accounting + deterministic reservoir for one configured source."""

    name: str
    requested: int       # cfg sample size (pre smoke cap)
    k: int               # actual reservoir size (min(requested, smoke cap))
    rng: random.Random
    reservoir: list[Record] = field(default_factory=list)
    available: int = 0   # rows that passed ALL filters (reservoir candidates)
    raw_rows: int = 0    # lines seen (before any filtering)
    drops: dict[str, int] = field(default_factory=dict)
    _drop_logged: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, detail: str, logger) -> None:
        """Count a dropped row; log the first few examples of each reason at DEBUG."""
        self.drops[reason] = self.drops.get(reason, 0) + 1
        n = self._drop_logged.get(reason, 0)
        if n < MAX_DEBUG_EXAMPLES:
            self._drop_logged[reason] = n + 1
            logger.debug("[%s] drop #%d (%s): %s", self.name, n + 1, reason, detail[:400])

    def offer(self, rec: Record) -> None:
        """Deterministic reservoir sampling (Algorithm R, seeded per source).

        The rng is consumed once per eligible row past the fill phase, so the
        selected set is stable across reruns as long as the row stream order is
        (it is: files are fetched at a pinned revision and iterated sorted).
        """
        self.available += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(rec)
            return
        j = self.rng.randrange(self.available)
        if j < self.k:
            self.reservoir[j] = rec


# ------------------------------------------------------------------ ChatML parsing


def parse_chatml(text: str) -> list[dict]:
    """Parse a pre-rendered ChatML string into [{"role","content"}, ...].

    Format (verified against all four source repos): repeated blocks of
        <|im_start|>{role}\\n{content}<|im_end|>
    separated by nothing or whitespace. The datasets render content with ONE
    trailing newline before <|im_end|>; we strip exactly that one newline and
    nothing else — over-stripping would eat meaningful trailing whitespace
    (e.g. code blocks ending in blank lines).

    Raises RowDrop with one of: unbalanced_markers, unknown_role, empty_content.
    (no_assistant_turn / not_ending_with_assistant are checked by the caller so
    company data in `messages` form shares the same validation.)
    """
    messages: list[dict] = []
    pos = 0
    while True:
        start = text.find(IM_START, pos)
        if start == -1:
            if text[pos:].strip():
                raise RowDrop("unbalanced_markers",
                              f"trailing junk after last block: {text[pos:pos + 80]!r}")
            break
        if text[pos:start].strip():
            raise RowDrop("unbalanced_markers",
                          f"junk between blocks: {text[pos:start][:80]!r}")
        end = text.find(IM_END, start)
        if end == -1:
            raise RowDrop("unbalanced_markers", "im_start without matching im_end")
        if text.find(IM_START, start + len(IM_START), end) != -1:
            raise RowDrop("unbalanced_markers", "nested im_start before im_end")
        block = text[start + len(IM_START):end]
        role, _, content = block.partition("\n")
        role = role.strip()
        if role not in VALID_ROLES:
            raise RowDrop("unknown_role", f"role={role!r}")
        if content.endswith("\n"):  # strip exactly ONE rendering newline
            content = content[:-1]
        if not content.strip():
            raise RowDrop("empty_content", f"role={role!r} has empty content")
        messages.append({"role": role, "content": content})
        pos = end + len(IM_END)
    if not messages:
        raise RowDrop("unbalanced_markers", "no ChatML blocks found")
    return messages


def validate_messages(messages: list[dict], drop_if_no_assistant: bool) -> None:
    """Structural checks shared by ChatML-parsed and company `messages` rows."""
    if not isinstance(messages, list) or not messages:
        raise RowDrop("bad_schema", "messages is not a non-empty list")
    for m in messages:
        if (not isinstance(m, dict) or not isinstance(m.get("role"), str)
                or not isinstance(m.get("content"), str)):
            raise RowDrop("bad_schema", f"malformed message entry: {str(m)[:80]}")
        if m["role"] not in VALID_ROLES:
            raise RowDrop("unknown_role", f"role={m['role']!r}")
        if not m["content"].strip():
            raise RowDrop("empty_content", f"role={m['role']!r} has empty content")
    if drop_if_no_assistant:
        if not any(m["role"] == "assistant" for m in messages):
            raise RowDrop("no_assistant_turn", "row has no assistant turn to train on")
        if messages[-1]["role"] != "assistant":
            raise RowDrop("not_ending_with_assistant",
                          f"last role={messages[-1]['role']!r}")


def add_think_prefix(messages: list[dict], prefix: str) -> None:
    """Prepend the empty-think scaffold to EVERY assistant turn (in place).

    Every turn — not just the last — so that ms-swift's rendered multi-turn output
    matches the tokenizer's enable_thinking=false path exactly (VERIFIED-FACTS §4).
    Rows that already start with '<think>' are left alone (idempotent).
    """
    for m in messages:
        if m["role"] == "assistant" and not m["content"].startswith("<think>"):
            m["content"] = prefix + m["content"]


# ------------------------------------------------------------------ row filtering


def build_hub_record(row: dict, src_name: str, fmt: dict, exclusions: dict,
                     include: list[str] | None, license_re: re.Pattern) -> Record:
    """Filter + parse one hub row. Raises RowDrop; filter order is contractual:
    include_source_names -> exclusions.source_names -> license regex -> token cap
    -> ChatML parse/validation (so drop counters are comparable across runs)."""
    text = row.get("text")
    source_name = row.get("source_name")
    lic = row.get("license")
    n_tokens = row.get("n_tokens")
    if (not isinstance(text, str) or not isinstance(source_name, str)
            or not isinstance(lic, str) or not isinstance(n_tokens, int)):
        raise RowDrop("bad_schema", f"missing/mistyped fields, keys={sorted(row)[:8]}")
    if include is not None and source_name not in include:
        raise RowDrop("not_in_include_source_names", source_name)
    if source_name in exclusions["source_names"]:
        raise RowDrop("excluded_source_name", source_name)
    if license_re.search(lic):
        raise RowDrop("license_regex", f"license={lic!r}")
    if n_tokens > fmt["max_tokens_per_example"]:
        raise RowDrop("over_max_tokens", f"n_tokens={n_tokens}")
    messages = parse_chatml(text)
    validate_messages(messages, fmt.get("drop_if_no_assistant", True))
    add_think_prefix(messages, fmt["think_prefix"])
    return Record(messages, src_name, source_name, lic, n_tokens)


def build_company_record(row: dict, fmt: dict) -> Record:
    """Company rows: {"messages":[...]} or {"text": <ChatML>}; license/n_tokens
    optional (no tokenizer here, so rows without n_tokens skip the length cap —
    ms-swift's truncation_strategy=delete is the backstop at train time)."""
    if isinstance(row.get("messages"), list):
        messages = [dict(m) if isinstance(m, dict) else m for m in row["messages"]]
        validate_messages(messages, fmt.get("drop_if_no_assistant", True))
    elif isinstance(row.get("text"), str):
        messages = parse_chatml(row["text"])
        validate_messages(messages, fmt.get("drop_if_no_assistant", True))
    else:
        raise RowDrop("bad_schema", "row has neither 'messages' list nor 'text' str")
    n_tokens = row.get("n_tokens")
    if not isinstance(n_tokens, int):
        n_tokens = None
    if n_tokens is not None and n_tokens > fmt["max_tokens_per_example"]:
        raise RowDrop("over_max_tokens", f"n_tokens={n_tokens}")
    add_think_prefix(messages, fmt["think_prefix"])
    lic = row.get("license") if isinstance(row.get("license"), str) else "company-proprietary"
    return Record(messages, "company_data", str(row.get("source_name", "company_data")),
                  lic, n_tokens)


# ------------------------------------------------------------------ hub file access


def resolve_source_files(src: dict, api: HfApi, logger) -> list[str]:
    """Expand a source's `files` (explicit list or glob) at the pinned revision.

    Globs MUST be expanded via list_repo_files, not any in-repo manifest.json —
    keural-rag-chatml-ko's manifest is known-stale (VERIFIED-FACTS §0.2)."""
    files = src["files"]
    if isinstance(files, list):
        return list(files)
    if not isinstance(files, str):
        raise PipelineError(f"source {src['name']!r}: 'files' must be list or glob string")
    all_files = api.list_repo_files(src["repo"], repo_type="dataset",
                                    revision=src["revision"])
    matched = sorted(fnmatch.filter(all_files, files))
    if not matched:
        raise PipelineError(
            f"source {src['name']!r}: glob {files!r} matched 0 of {len(all_files)} files "
            f"in {src['repo']}@{src['revision'][:12]} — wrong revision or pattern?")
    logger.info("[%s] glob %r -> %d files", src["name"], files, len(matched))
    return matched


def iter_hub_lines(repo: str, revision: str, filename: str, smoke: bool,
                   logger) -> Iterator[str]:
    """Yield text lines of one repo file at the pinned revision.

    Full mode: hf_hub_download (cache-first; a pinned commit SHA that is already
    cached resolves with zero network calls). Smoke mode: cache-only first, and if
    the file is not cached, STREAM it with HfFileSystem range-reads so we never
    pull a multi-GB file just to read its first ~300 rows."""
    if smoke:
        try:
            path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                                   revision=revision, local_files_only=True)
            logger.debug("smoke: cached %s/%s", repo, filename)
        except LocalEntryNotFoundError:
            logger.info("smoke: streaming (not cached) %s/%s @ %s",
                        repo, filename, revision[:12])
            fs = HfFileSystem()
            with fs.open(f"datasets/{repo}@{revision}/{filename}", "rb",
                         block_size=1 << 20) as raw:
                yield from io.TextIOWrapper(raw, encoding="utf-8")
            return
    else:
        path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                               revision=revision)
    with open(path, encoding="utf-8") as f:
        yield from f


# ------------------------------------------------------------------ source ingestion


def include_fastpath_miss(line: str, include: list[str] | None) -> bool:
    """True if the raw JSON line certainly fails include_source_names, WITHOUT
    parsing it. Sound because the include names are plain-ASCII identifiers: if
    source_name equaled one, the literal '"<name>"' substring would appear in the
    serialized line. Saves ~json.loads over millions of non-matching keural-SFT
    rows (shards are grouped by source). False -> caller must fully parse."""
    if include is None:
        return False
    return not any(f'"{name}"' in line for name in include)


def ingest_source(idx: int, src: dict, cfg: dict, smoke: bool, api: HfApi,
                  logger) -> SourceState:
    """Stream one configured source into a deterministic reservoir sample."""
    seed = cfg["pipeline"]["seed"] + idx
    requested = int(src["sample"])
    k = min(requested, SMOKE_CAP) if smoke else requested
    state = SourceState(name=src["name"], requested=requested, k=k,
                        rng=random.Random(seed))
    fmt = cfg["format"]
    exclusions = cfg["exclusions"]
    license_re = re.compile(exclusions["license_regex"], re.IGNORECASE)
    include = src.get("include_source_names")
    cap = SMOKE_CAP if smoke else None

    files = resolve_source_files(src, api, logger)
    state.files = files  # type: ignore[attr-defined]  # recorded in the manifest
    logger.info("[%s] sampling %d (seed %d) from %s@%s, %d file(s)%s",
                state.name, k, seed, src["repo"], src["revision"][:12], len(files),
                f" (smoke cap {cap})" if smoke else "")

    for n_file, filename in enumerate(files, 1):
        if cap is not None and state.available >= cap:
            break
        for line in iter_hub_lines(src["repo"], src["revision"], filename, smoke, logger):
            if cap is not None and state.available >= cap:
                break
            if not line.strip():
                continue
            state.raw_rows += 1
            if include_fastpath_miss(line, include):
                state.drop("not_in_include_source_names", "fastpath skip", logger)
                continue
            try:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RowDrop("invalid_json", f"{e}: {line[:120]!r}") from e
                if not isinstance(row, dict):
                    raise RowDrop("bad_schema", f"row is {type(row).__name__}, not object")
                state.offer(build_hub_record(row, state.name, fmt, exclusions,
                                             include, license_re))
            except RowDrop as d:
                state.drop(d.reason, d.detail, logger)
        if n_file % 10 == 0 or n_file == len(files):
            logger.info("[%s] %d/%d files read: %d raw, %d eligible",
                        state.name, n_file, len(files), state.raw_rows, state.available)

    if state.available < requested and not smoke:
        logger.warning("[%s] only %d eligible rows available (< requested %d) — taking all",
                       state.name, state.available, requested)
    logger.info("[%s] done: raw=%d eligible=%d sampled=%d drops=%s",
                state.name, state.raw_rows, state.available, len(state.reservoir),
                dict(sorted(state.drops.items())) or "{}")
    return state


def ingest_company_data(cfg: dict, smoke: bool, logger) -> SourceState | None:
    """Optional local company/domain JSONL (guide §2). Missing/null -> WARN + skip:
    the guide says this data is not available yet, so absence must not block the
    pipeline — but it must be loudly visible in logs and manifest."""
    company = cfg.get("company_data") or {}
    path_cfg = company.get("path")
    if not path_cfg:
        logger.warning("company_data.path is null — skipping company/domain data "
                       "(guide §2 target of 15-20k examples NOT included)")
        return None
    path = common.resolve_path(cfg, path_cfg)
    if not path.exists():
        logger.warning("company_data.path %s does not exist — skipping company data", path)
        return None

    idx = len(cfg["sources"])  # stable seed slot right after the hub sources
    seed = cfg["pipeline"]["seed"] + idx
    requested = int(company.get("sample", 20000))
    k = min(requested, SMOKE_CAP) if smoke else requested
    state = SourceState(name="company_data", requested=requested, k=k,
                        rng=random.Random(seed))
    state.files = [str(path)]  # type: ignore[attr-defined]
    fmt = cfg["format"]
    cap = SMOKE_CAP if smoke else None
    logger.info("[company_data] sampling %d (seed %d) from %s", k, seed, path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if cap is not None and state.available >= cap:
                break
            if not line.strip():
                continue
            state.raw_rows += 1
            try:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RowDrop("invalid_json", f"{e}: {line[:120]!r}") from e
                if not isinstance(row, dict):
                    raise RowDrop("bad_schema", f"row is {type(row).__name__}, not object")
                state.offer(build_company_record(row, fmt))
            except RowDrop as d:
                state.drop(d.reason, d.detail, logger)
    logger.info("[company_data] done: raw=%d eligible=%d sampled=%d drops=%s",
                state.raw_rows, state.available, len(state.reservoir),
                dict(sorted(state.drops.items())) or "{}")
    return state


# ------------------------------------------------------------------ merge / split


def dedup_records(states: list[SourceState], enabled: bool, logger) -> list[Record]:
    """Cross-source exact dedup, first-seen wins (source config order, then
    reservoir order — deterministic). The key is a sha256 over role +
    whitespace-normalized content, computed AFTER think-prefixing so a row that
    shipped with its own '<think>' prefix collides with its prefix-added twin."""
    merged: list[Record] = []
    seen: set[str] = set()
    for state in states:
        for rec in state.reservoir:
            if enabled:
                h = hashlib.sha256()
                for m in rec.messages:
                    h.update(m["role"].encode())
                    h.update(b"\x1e")
                    h.update(" ".join(m["content"].split()).encode())
                    h.update(b"\x1f")
                key = h.hexdigest()
                if key in seen:
                    state.drop("dedup_duplicate",
                               rec.messages[0]["content"][:80], logger)
                    continue
                seen.add(key)
            merged.append(rec)
    return merged


def split_eval(merged: list[Record], eval_cfg: dict, seed: int,
               logger) -> tuple[list[Record], list[Record]]:
    """Seeded shuffle, then stratified-by-source eval split (fraction, capped at
    max_examples, never emptying a source's train side). Both returned lists keep
    the global shuffled order so the train file is not grouped by source (guide §3)."""
    rng = random.Random(seed)
    rng.shuffle(merged)
    fraction = float(eval_cfg["fraction"])
    max_examples = int(eval_cfg["max_examples"])

    by_source: dict[str, list[int]] = {}
    for i, rec in enumerate(merged):
        by_source.setdefault(rec.source, []).append(i)
    n_eval = {s: min(int(len(ix) * fraction), max(len(ix) - 1, 0))
              for s, ix in by_source.items()}
    total = sum(n_eval.values())
    if total > max_examples:
        scale = max_examples / total
        n_eval = {s: int(n * scale) for s, n in n_eval.items()}
        logger.info("eval split capped: %d -> %d (max_examples=%d)",
                    total, sum(n_eval.values()), max_examples)
    eval_idx = {i for s, ix in by_source.items() for i in ix[:n_eval[s]]}
    train = [r for i, r in enumerate(merged) if i not in eval_idx]
    eval_rows = [r for i, r in enumerate(merged) if i in eval_idx]
    if not eval_rows:
        logger.warning("eval split is EMPTY (fraction=%s over %d rows) — "
                       "training would run without eval loss", fraction, len(merged))
    return train, eval_rows


def pilot_slice(train: list[Record], total: int, logger) -> list[Record]:
    """Proportional-by-source slice of TRAIN (never eval rows — the pilot's eval
    reuses eval.jsonl, so leakage here would fake the pilot's eval loss).
    Largest-remainder rounding hits exactly `total` when enough rows exist."""
    counts: dict[str, int] = {}
    for rec in train:
        counts[rec.source] = counts.get(rec.source, 0) + 1
    grand = len(train)
    if total > grand:
        logger.warning("pilot.total_examples=%d > train size %d — pilot = whole train",
                       total, grand)
        return list(train)
    raw = {s: total * c / grand for s, c in counts.items()}
    quota = {s: int(v) for s, v in raw.items()}
    remainder = total - sum(quota.values())
    for s in sorted(raw, key=lambda s: (raw[s] - quota[s], s), reverse=True)[:remainder]:
        quota[s] += 1
    out: list[Record] = []
    for rec in train:  # keep shuffled order; take per-source quotas in stream order
        if quota.get(rec.source, 0) > 0:
            quota[rec.source] -= 1
            out.append(rec)
    return out


def assert_output_invariants(records: list[Record], think_prefix: str) -> None:
    """Final belt-and-braces pass over everything we are about to write. A failure
    here is an internal bug, not bad input — hard stop before poisoning training."""
    for n, rec in enumerate(records):
        if not rec.messages or rec.messages[-1]["role"] != "assistant":
            raise PipelineError(f"internal bug: output row {n} does not end with an "
                                f"assistant turn ({rec.source})")
        for m in rec.messages:
            if m["role"] not in VALID_ROLES or not m["content"].strip():
                raise PipelineError(f"internal bug: output row {n} has invalid message "
                                    f"(role={m['role']!r}) ({rec.source})")
            if m["role"] == "assistant" and not m["content"].startswith("<think>"):
                raise PipelineError(f"internal bug: output row {n} assistant turn lacks "
                                    f"think prefix ({rec.source})")


# ------------------------------------------------------------------ outputs


def write_jsonl(path: Path, records: list[Record], logger) -> dict:
    """Write {"messages": [...]} rows (ms-swift schema — nothing else; extra keys
    would be silently ingested or rejected depending on ms-swift version).
    tmp+rename so a killed run never leaves a truncated 'real' file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"messages": rec.messages}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    info = {"path": str(path), "rows": len(records), "sha256": common.sha256_file(path)}
    logger.info("wrote %s: %d rows, sha256 %s...", path, len(records),
                info["sha256"][:16])
    return info


def token_bucket(n_tokens: int | None) -> str:
    if n_tokens is None:
        return "unknown"
    for lo, hi in zip(TOKEN_BUCKET_EDGES, TOKEN_BUCKET_EDGES[1:]):
        if lo <= n_tokens < hi:
            return f"{lo}-{hi - 1}"
    return f"{TOKEN_BUCKET_EDGES[-1]}+"


def output_paths(cfg: dict, smoke: bool) -> dict[str, Path]:
    """Resolve the five output paths; --smoke redirects all of them (same basenames)
    into <processed>/smoke/ so smoke runs can never clobber real outputs."""
    out = cfg["output"]
    keys = ("train_file", "eval_file", "pilot_file", "manifest_file", "stats_file")
    paths = {k: common.resolve_path(cfg, out[k]) for k in keys}
    if smoke:
        smoke_dir = paths["train_file"].parent / "smoke"
        paths = {k: smoke_dir / p.name for k, p in paths.items()}
    return paths


def compute_fingerprint(cfg: dict, smoke: bool) -> str:
    """Fingerprint of everything that determines the outputs: the full dataset
    stage config + pipeline seed + smoke mode/cap. pipeline.yaml's other keys
    (model, thresholds) do not affect this stage's outputs, so they are excluded
    to avoid pointless rebuilds."""
    stage_cfg = {k: v for k, v in cfg.items() if k != "pipeline"}
    return common.config_fingerprint({
        "dataset": stage_cfg,
        "seed": cfg["pipeline"]["seed"],
        "smoke": smoke,
        "smoke_cap": SMOKE_CAP if smoke else None,
    })


def up_to_date(paths: dict[str, Path], fingerprint: str, logger) -> bool:
    """True iff the manifest exists, its fingerprint matches, and every recorded
    output file exists with a matching sha256 (catches manual edits/truncation)."""
    manifest_path = paths["manifest_file"]
    if not manifest_path.exists():
        return False
    try:
        manifest = common.read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("existing manifest unreadable (%s) — rebuilding", e)
        return False
    if manifest.get("config_fingerprint") != fingerprint:
        logger.info("config fingerprint changed (%s -> %s) — rebuilding",
                    manifest.get("config_fingerprint"), fingerprint)
        return False
    for name, entry in manifest.get("outputs", {}).items():
        p = Path(entry.get("path", ""))
        if not p.exists():
            logger.info("output %s missing (%s) — rebuilding", name, p)
            return False
        if common.sha256_file(p) != entry.get("sha256"):
            logger.warning("output %s sha256 mismatch (%s) — rebuilding", name, p)
            return False
    return True


def validate_stage_config(cfg: dict) -> None:
    """Fail fast with a precise message instead of a KeyError traceback later."""
    missing = []
    for dotted in ("output.train_file", "output.eval_file", "output.pilot_file",
                   "output.manifest_file", "output.stats_file",
                   "eval_split.fraction", "eval_split.max_examples",
                   "pilot.total_examples", "exclusions.source_names",
                   "exclusions.license_regex", "format.think_prefix",
                   "format.max_tokens_per_example", "dedup.enabled"):
        node = cfg
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.append(dotted)
                break
            node = node[part]
    if not isinstance(cfg.get("sources"), list) or not cfg.get("sources"):
        missing.append("sources (non-empty list)")
    if missing:
        raise PipelineError(f"dataset config is missing required keys: {missing}")
    names = [s.get("name") for s in cfg["sources"]]
    if len(names) != len(set(names)):
        raise PipelineError(f"duplicate source names in config: {names}")
    for src in cfg["sources"]:
        for key in ("name", "repo", "revision", "files", "sample"):
            if key not in src:
                raise PipelineError(f"source {src.get('name', '?')!r} missing {key!r}")


# ------------------------------------------------------------------ main


def build(cfg: dict, smoke: bool, paths: dict[str, Path], fingerprint: str,
          logger) -> None:
    seed = cfg["pipeline"]["seed"]
    api = HfApi()

    states = [ingest_source(i, src, cfg, smoke, api, logger)
              for i, src in enumerate(cfg["sources"])]
    company_state = ingest_company_data(cfg, smoke, logger)
    if company_state is not None:
        states.append(company_state)

    merged = dedup_records(states, bool(cfg["dedup"]["enabled"]), logger)
    if not merged:
        raise PipelineError("0 rows survived filtering/sampling — check the source "
                            "revisions, filters and drop counters in the log")
    logger.info("merged: %d rows across %d source(s) after dedup", len(merged), len(states))

    train, eval_rows = split_eval(merged, cfg["eval_split"], seed, logger)
    pilot = pilot_slice(train, int(cfg["pilot"]["total_examples"]), logger)
    logger.info("split: train=%d eval=%d pilot=%d", len(train), len(eval_rows), len(pilot))
    assert_output_invariants(merged, cfg["format"]["think_prefix"])

    outputs = {
        "train": write_jsonl(paths["train_file"], train, logger),
        "eval": write_jsonl(paths["eval_file"], eval_rows, logger),
        "pilot": write_jsonl(paths["pilot_file"], pilot, logger),
    }

    # --- stats: token/turns histograms + per-source final counts -----------------
    tok_hist: dict[str, int] = {}
    turn_hist: dict[str, int] = {}
    per_source: dict[str, dict[str, int]] = {}
    license_summary: dict[str, int] = {}
    splits = [("total", merged), ("train", train), ("eval", eval_rows), ("pilot", pilot)]
    for rec in merged:
        tok_hist[token_bucket(rec.n_tokens)] = tok_hist.get(token_bucket(rec.n_tokens), 0) + 1
        turn_hist[str(len(rec.messages))] = turn_hist.get(str(len(rec.messages)), 0) + 1
        license_summary[rec.license] = license_summary.get(rec.license, 0) + 1
    for split_name, rows in splits:
        for rec in rows:
            per_source.setdefault(rec.source, {})
            per_source[rec.source][split_name] = per_source[rec.source].get(split_name, 0) + 1

    def _hist_sorted(h: dict[str, int], numeric: bool) -> dict[str, int]:
        def key(k: str):
            head = k.split("-")[0].rstrip("+")
            return (0, int(head)) if numeric and head.isdigit() else (1, k)
        return {k: h[k] for k in sorted(h, key=key)}

    common.write_json_atomic(paths["stats_file"], {
        "config_fingerprint": fingerprint,
        "mode": "smoke" if smoke else "full",
        "created_utc": common.utc_stamp(),
        "n_tokens_histogram": _hist_sorted(tok_hist, numeric=True),
        "turns_histogram": _hist_sorted(turn_hist, numeric=True),
        "per_source_final_counts": per_source,
    })
    logger.info("wrote stats: %s", paths["stats_file"])

    # --- manifest (written LAST: its existence marks the stage complete) ---------
    sources_manifest = {}
    for i, state in enumerate(states):
        src_cfg = cfg["sources"][i] if i < len(cfg["sources"]) else (cfg.get("company_data") or {})
        sources_manifest[state.name] = {
            "repo": src_cfg.get("repo"),
            "revision": src_cfg.get("revision"),
            "files": getattr(state, "files", []),
            "requested": state.requested,
            "reservoir_size": state.k,
            "raw_rows_seen": state.raw_rows,
            "available": state.available,
            "sampled": len(state.reservoir),
            "kept": per_source.get(state.name, {}).get("total", 0),
            "dropped_by_reason": dict(sorted(state.drops.items())),
        }
    common.write_json_atomic(paths["manifest_file"], {
        "stage": STAGE,
        "mode": "smoke" if smoke else "full",
        "config_fingerprint": fingerprint,
        "created_utc": common.utc_stamp(),
        "seed": seed,
        "smoke_cap": SMOKE_CAP if smoke else None,
        "sources": sources_manifest,
        "company_data_included": company_state is not None,
        "outputs": outputs,
        "license_summary": dict(sorted(license_summary.items())),
    })
    logger.info("wrote manifest: %s", paths["manifest_file"])
    logger.info("license summary: %s", dict(sorted(license_summary.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1: build train/eval/pilot JSONL from the pinned HF sources.")
    parser.add_argument("--config", default="configs/dataset.yaml",
                        help="stage config YAML (default: configs/dataset.yaml)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the manifest says outputs are up to date")
    parser.add_argument("--smoke", action="store_true",
                        help=f"cap every source at {SMOKE_CAP} rows, stream instead of "
                             "downloading uncached files, write to data/processed/smoke/")
    args = parser.parse_args(argv)

    cfg = common.load_config(args.config)
    logger = common.setup_logging(STAGE, cfg)
    common.stage_banner(logger, STAGE + (" (SMOKE)" if args.smoke else ""), args.config)
    try:
        common.guard_environment(logger)
        validate_stage_config(cfg)
        if not args.smoke:
            common.assert_free_disk(cfg, STAGE, logger)

        fingerprint = compute_fingerprint(cfg, args.smoke)
        paths = output_paths(cfg, args.smoke)
        logger.info("config fingerprint: %s | outputs -> %s",
                    fingerprint, paths["train_file"].parent)
        if not args.force and up_to_date(paths, fingerprint, logger):
            logger.info("up to date (fingerprint %s, all outputs verified) — nothing to do; "
                        "use --force to rebuild", fingerprint)
            return 0

        build(cfg, args.smoke, paths, fingerprint, logger)
        logger.info("dataset stage complete")
        return 0
    except PipelineError as e:
        logger.error("FATAL: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
