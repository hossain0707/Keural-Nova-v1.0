# Domain eval set (guide §1: "your own domain eval")

This directory holds **our own company/domain evaluation examples** — the third leg of the
eval triad (Korean benchmarks / general regression tripwire / domain). The training guide
(§1) calls for **50–100 real examples from the actual use case, graded consistently
(rubric or LLM-judge)**. None have been provided by the team yet, so the harness treats an
empty directory as a *loudly-logged skip*, not an error.

## How it plugs in

`scripts/run_eval.py --suite domain` (or `--suite all`) scans this directory for `*.jsonl`
files. For each example it:

1. renders the prompt with the model's chat template (`enable_thinking=False`, matching
   the training data's empty-think format — see guide §3),
2. generates deterministically (temperature 0, `max_new_tokens` from
   `configs/eval.yaml: domain.max_new_tokens`) with the selected backend (vllm or hf),
3. writes `{id, prompt, reference, output}` per row to
   `results/<tag>/<stamp>/domain_generations.jsonl`,
4. scores with a **cheap heuristic only**: `exact` (normalized output == normalized
   reference) and `contains` (normalized reference substring of output), and records
   `needs_human_or_judge: true` in the summary.

> **Grading placeholder:** the guide asks for rubric- or LLM-judge-based grading. There is
> no judge endpoint in this environment, so the heuristic scores are a *tripwire*, not a
> grade. The `rubric` field is stored and carried through to `domain_generations.jsonl`
> precisely so a human grader or a future LLM-judge stage can consume it. Do **not** quote
> `exact_match_rate` / `contains_rate` as domain quality numbers — compare generations by
> hand (or wire up a judge) before drawing conclusions.

## File format

One or more `*.jsonl` files, one JSON object per line:

| field       | required | type | meaning                                                        |
|-------------|----------|------|----------------------------------------------------------------|
| `id`        | yes      | str  | unique across ALL files in this dir (harness hard-fails on dup) |
| `prompt`    | yes      | str  | the user message, exactly as a real user would send it          |
| `reference` | yes      | str  | the ideal answer (used by the contains/exact heuristic)         |
| `rubric`    | no       | str  | grading criteria for a human or LLM judge                       |

See `example.jsonl.disabled` for a concrete example. The `.disabled` suffix keeps it out of
the `*.jsonl` glob so it is **never executed** — rename a copy to `something.jsonl` to
activate real data. Files are validated strictly (missing/empty required fields, non-JSON
lines, duplicate ids all abort with the offending file:line) — a silently-mangled eval set
is worse than none.

## Authoring guidance

- Target **50–100 examples** covering the real distribution of use cases (guide §1) —
  enough to see movement, small enough to hand-grade.
- Write prompts in the language users actually use (mostly Korean for Keural).
- Keep `reference` short and canonical where possible — the `contains` heuristic works best
  when the reference is the essential fact/phrase, not a full essay.
- Put grading nuance in `rubric`, not in `reference`.
- Keep the set **frozen** between baseline and post-training runs — the whole point is a
  like-for-like delta. `run_eval.py` fingerprints the file contents, so editing the set
  invalidates prior-run reuse automatically.

## Quick check without GPUs

`run_eval.py --suite domain --dry-run` scans and validates every file here (schema, dup
ids) and prints what would run, without touching the GPUs.
