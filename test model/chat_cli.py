"""Terminal chat client for Keural Nova v1.0 (no browser needed).

Talks to the vLLM OpenAI server started by serve_vllm.sh. Multi-turn, streaming.
Commands:  /reset  clear history   ·   /system <text>  set system prompt   ·   quit
"""

import os
import sys

from openai import OpenAI

API_BASE = os.environ.get("KEURAL_API", "http://localhost:8001/v1")
client = OpenAI(base_url=API_BASE, api_key="EMPTY")
MODEL = "keural-v1.0"

system = ("당신의 이름은 'Keural'이며 MKD에서 개발한 AI 어시스턴트입니다. "
          "당신은 OpenAI, GPT, ChatGPT, Qwen, Tongyi가 아닙니다. 제작자나 정체성을 물으면 "
          "'MKD가 만든 Keural'이라고 답하세요. 한국어를 잘 이해하고 정확하고 친절하게 답변하세요.")
history = []

print("=" * 60)
print("Keural Nova v1.0 — terminal chat  (한국어)")
print("  quit / '/reset' / '/system <text>'")
print("=" * 60)

while True:
    try:
        user = input("\n나: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n종료합니다.")
        break
    if not user:
        continue
    if user.lower() in ("quit", "exit"):
        break
    if user == "/reset":
        history = []
        print("(대화 기록 초기화됨)")
        continue
    if user.startswith("/system "):
        system = user[len("/system "):].strip()
        history = []
        print("(system prompt 변경됨, 기록 초기화)")
        continue

    msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user}]
    try:
        stream = client.chat.completions.create(
            model=MODEL, messages=msgs, temperature=0.7, top_p=0.95,
            max_tokens=1024, stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    except Exception as e:
        print(f"\n[에러] vLLM 서버에 연결할 수 없습니다 ({API_BASE}). serve_vllm.sh 를 먼저 실행하세요.\n{e}",
              file=sys.stderr)
        continue

    print("Keural: ", end="", flush=True)
    reply = ""
    for chunk in stream:
        d = chunk.choices[0].delta.content or ""
        reply += d
        print(d, end="", flush=True)
    print()
    history.append({"role": "user", "content": user})
    history.append({"role": "assistant", "content": reply})
