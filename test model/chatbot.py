"""Interactive web chatbot for Keural Nova v1.0.

A Gradio streaming chat UI that talks to a running vLLM OpenAI-compatible server
(start it first with serve_vllm.sh). Multi-turn memory, streaming tokens,
adjustable sampling, and the model's non-thinking Korean chat format.

  KEURAL_API   vLLM base URL   (default http://localhost:8001/v1)
  KEURAL_UIPORT gradio port    (default 7860 — exposed via KT Cloud /proxy/7860)
"""

import os

import gradio as gr
from openai import OpenAI

API_BASE = os.environ.get("KEURAL_API", "http://localhost:8001/v1")
UI_PORT = int(os.environ.get("KEURAL_UIPORT", "7861"))
MODEL = "keural-v1.0"

client = OpenAI(base_url=API_BASE, api_key="EMPTY")

DEFAULT_SYSTEM = (
    "당신의 이름은 'Keural'이며 MKD에서 개발한 AI 어시스턴트입니다. "
    "당신은 OpenAI, GPT, ChatGPT, Qwen, Tongyi가 아닙니다. 제작자나 정체성을 물으면 "
    "'MKD가 만든 Keural'이라고 답하세요. 한국어를 잘 이해하고 정확하고 친절하게 답변하세요."
)


def respond(message, history, system_prompt, temperature, top_p, max_tokens):
    """Stream a reply. `history` is Gradio 'messages' format: [{role, content}, ...]."""
    msgs = [{"role": "system", "content": system_prompt}] if system_prompt.strip() else []
    for turn in history:
        # keep only role/content (Gradio may attach metadata)
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        stream=True,
        # v1.0 is trained in the base model's non-thinking format
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        yield partial


with gr.Blocks(title="Keural Nova v1.0 — 채팅 테스트", fill_height=True) as demo:
    gr.Markdown(
        "# 🗣️ Keural Nova v1.0\n"
        "한국어 파인튜닝 모델 대화 테스트 · `mkd-hossain/Keural-Nova-v1.0` "
        "(checkpoint-7500) · 서빙: vLLM on 2× H200"
    )
    with gr.Accordion("생성 설정 (Generation settings)", open=False):
        system_prompt = gr.Textbox(value=DEFAULT_SYSTEM, label="System prompt", lines=2)
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="top_p")
            max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="max_tokens")

    gr.ChatInterface(
        fn=respond,
        type="messages",
        additional_inputs=[system_prompt, temperature, top_p, max_tokens],
        examples=[
            ["요즘 잠을 잘 못 자는데, 친구처럼 편하게 조언해줘."],
            ["회의에 늦어서 팀장님께 보낼 정중한 사과 이메일을 작성해줘."],
            ["'손이 크다'라는 표현의 뜻과 예문 두 개를 알려줘."],
            ["김치가 발효되는 과정을 유산균 중심으로 설명해줘."],
            ["'가을비'로 삼행시를 지어줘."],
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=UI_PORT, share=False)
