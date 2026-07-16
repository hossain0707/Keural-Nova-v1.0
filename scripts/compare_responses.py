"""Qualitative side-by-side: run one Korean prompt battery through N models
sequentially (each model gets both GPUs via vLLM, then is unloaded) and write a
side-by-side markdown report + JSON.

Sequential by necessity: three ~72 GB bf16 models cannot coexist on 2x H200.
Deterministic-ish generation (temperature 0 by default) so differences reflect
the models, not sampling luck.

Usage:
  python scripts/compare_responses.py                       # default 3 models + built-in battery
  python scripts/compare_responses.py --prompts my.jsonl    # custom battery
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

MODELS = [
    ("base (Qwen3.6-35B-A3B)",
     "/home/work/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"),
    ("Keural v1.0 (ckpt-7500)", "outputs/merged/Keural-35B-A3B-ckpt7500"),
    ("Keural v1.1 (final)", "outputs/merged/Keural-35B-A3B"),
]

# Korean-understanding battery: one prompt per capability we care about.
BATTERY = [
    ("일상 대화", "요즘 잠을 잘 못 자는데 어떻게 하면 좋을까? 친구처럼 편하게 조언해줘."),
    ("격식/존댓말", "회사에서 중요한 회의에 늦어서 팀장님께 사과 메일을 보내야 합니다. 정중한 사과 이메일을 작성해 주세요."),
    ("관용 표현", "'손이 크다'라는 한국어 표현의 의미를 설명하고, 예문 두 개를 만들어 주세요."),
    ("요약", "다음 글을 한 문장으로 요약해 주세요: 한국은행이 기준금리를 동결했다. 물가 상승세가 둔화되고 있지만 가계부채 증가와 환율 불안이 여전히 부담으로 작용하고 있기 때문이다. 시장에서는 연내 인하 가능성을 열어두면서도 그 시점은 불확실하다고 보고 있다."),
    ("뉘앙스 번역", "다음 영어 문장을 자연스러운 한국어로 번역해 주세요: \"I didn't mean to hurt your feelings — it just came out wrong.\""),
    ("지식 설명", "김치가 발효되는 과정을 과학적으로 설명해 주세요. 유산균의 역할을 중심으로요."),
    ("논리 추론", "민수는 지영이보다 키가 크고, 지영이는 현우보다 키가 큽니다. 현우는 반에서 두 번째로 키가 큽니다. 이 반에 네 명(민수, 지영, 현우, 수진)이 있다면 수진이의 키 순위는 몇 번째일까요? 차근차근 설명해 주세요."),
    ("수학 문장제", "사과 한 상자에 사과가 24개 들어 있습니다. 학교에서 상자 7개를 샀는데, 학생 156명에게 한 개씩 나눠 주었습니다. 남은 사과는 몇 개인가요?"),
    ("코드 요청", "파이썬으로 피보나치 수열의 n번째 항을 구하는 함수를 작성해 주세요."),
    ("창작", "'가을비'로 삼행시를 지어 주세요."),
    ("문장 교정", "다음 문장에서 어색한 부분을 찾아 자연스럽게 고쳐 주세요: '나는 어제 도서관에 공부를 열심히 하러 갔었었다.'"),
    ("독해 (RAG)", "다음 지문을 읽고 질문에 답하세요.\n\n지문: 세종대왕은 1443년 훈민정음을 창제하였고 1446년에 이를 반포하였다. 훈민정음은 '백성을 가르치는 바른 소리'라는 뜻으로, 당시 한자를 배우기 어려웠던 백성들을 위해 만들어졌다.\n\n질문: 훈민정음이 반포된 해는 언제이며, 만들어진 목적은 무엇인가요?"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--prompts", default=None,
                    help="optional JSONL with rows {'category','prompt'} to replace the battery")
    ap.add_argument("--out", default=None, help="output dir (default results/side_by_side/<stamp>)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (default), so differences are model differences")
    ap.add_argument("--allow-busy-gpus", action="store_true")
    args = ap.parse_args()

    cfg = common.load_config(None)
    logger = common.setup_logging("side_by_side", cfg)
    common.stage_banner(logger, "compare_responses", None)
    common.guard_environment(logger)
    if not args.allow_busy_gpus:
        common.assert_gpus_free(cfg, logger)

    battery = BATTERY
    if args.prompts:
        battery = [(r.get("category", f"prompt {i}"), r["prompt"])
                   for i, r in enumerate(json.loads(l) for l in open(args.prompts, encoding="utf-8"))]
    logger.info("battery: %d prompts, %d models, temp=%s", len(battery), len(MODELS), args.temperature)

    out_dir = Path(args.out) if args.out else (
        common.project_root() / "results" / "side_by_side" / common.utc_stamp())
    out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams  # heavy import kept below the guards
    from transformers import AutoTokenizer

    sp = SamplingParams(temperature=args.temperature,
                        top_p=0.95 if args.temperature > 0 else 1.0,
                        max_tokens=args.max_new_tokens, seed=42)

    all_outputs: dict[str, list[str]] = {}
    for name, path in MODELS:
        resolved = str(common.resolve_path(cfg, path))
        logger.info("=== loading %s (%s) ===", name, resolved)
        tok = AutoTokenizer.from_pretrained(resolved)
        prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
                   for _, p in battery]
        llm = LLM(model=resolved, tensor_parallel_size=2, gpu_memory_utilization=0.85,
                  max_model_len=8192, dtype="bfloat16", seed=42)
        gens = llm.generate(prompts, sp)
        all_outputs[name] = [g.outputs[0].text.strip() for g in gens]
        logger.info("%s: %d generations done", name, len(gens))
        # tear down fully so the next model gets clean GPUs
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    common.write_json_atomic(out_dir / "responses.json", {
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
        "battery": [{"category": c, "prompt": p} for c, p in battery],
        "models": {n: all_outputs[n] for n, _ in MODELS},
    })

    md = ["# Side-by-side: base vs Keural v1.0 vs v1.1 (Korean battery)\n",
          f"- generation: temperature={args.temperature}, max_new_tokens={args.max_new_tokens}, "
          f"non-thinking chat template, seed 42\n"]
    for i, (cat, prompt) in enumerate(battery):
        md.append(f"\n---\n\n## {i + 1}. {cat}\n\n**프롬프트:** {prompt}\n")
        for name, _ in MODELS:
            md.append(f"\n### {name}\n\n{all_outputs[name][i]}\n")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("report: %s", out_dir / "report.md")
    print(f"REPORT: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except common.PipelineError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
