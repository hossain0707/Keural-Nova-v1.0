"""Live side-by-side chat: Keural Nova v1.0 vs Qwen3.6-35B-A3B (base).

One prompt box, two chat panels. Your message goes to BOTH models at once and
their replies stream in parallel, so you can compare Korean understanding and
response quality directly. Requires both vLLM servers up (serve_dual.sh).

  KEURAL_API  default http://localhost:8001/v1
  QWEN_API    default http://localhost:8002/v1
  KEURAL_UIPORT default 7861
"""

import os
import threading

import gradio as gr
from openai import OpenAI

KEURAL_API = os.environ.get("KEURAL_API", "http://localhost:8001/v1")
QWEN_API = os.environ.get("QWEN_API", "http://localhost:8002/v1")
UI_PORT = int(os.environ.get("KEURAL_UIPORT", "7861"))

kc = OpenAI(base_url=KEURAL_API, api_key="EMPTY")
qc = OpenAI(base_url=QWEN_API, api_key="EMPTY")

SYS_KEURAL = ("당신의 이름은 'Keural'이며 MKD에서 개발한 AI 어시스턴트입니다. "
              "한국어를 잘 이해하고 정확하고 친절하게 답변하세요.")
SYS_QWEN = "You are a helpful assistant. 한국어로 질문하면 한국어로 답변하세요."

ENDPOINTS = {
    "keural": (kc, "keural-v1.0"),
    "qwen": (qc, "qwen-base"),
}


def _stream_into(which, msgs, temperature, max_tokens, buf):
    """Run one model's stream, accumulating text into buf['text']; set buf['done']."""
    client, model = ENDPOINTS[which]
    try:
        stream = client.chat.completions.create(
            model=model, messages=msgs, temperature=float(temperature),
            top_p=0.95, max_tokens=int(max_tokens), stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        for chunk in stream:
            buf["text"] += (chunk.choices[0].delta.content or "")
    except Exception as e:  # surface connection/other errors inside the panel
        buf["text"] += f"\n\n[에러] {which} 서버 응답 실패: {e}"
    finally:
        buf["done"] = True


def chat(message, k_hist, q_hist, sys_k, sys_q, temperature, max_tokens):
    k_hist = (k_hist or []) + [{"role": "user", "content": message},
                              {"role": "assistant", "content": ""}]
    q_hist = (q_hist or []) + [{"role": "user", "content": message},
                              {"role": "assistant", "content": ""}]

    def build(hist, sys_prompt):
        msgs = [{"role": "system", "content": sys_prompt}] if sys_prompt.strip() else []
        # everything except the trailing empty assistant placeholder
        for t in hist[:-1]:
            if t["role"] in ("user", "assistant") and t["content"]:
                msgs.append({"role": t["role"], "content": t["content"]})
        return msgs

    kbuf = {"text": "", "done": False}
    qbuf = {"text": "", "done": False}
    tk = threading.Thread(target=_stream_into,
                          args=("keural", build(k_hist, sys_k), temperature, max_tokens, kbuf))
    tq = threading.Thread(target=_stream_into,
                          args=("qwen", build(q_hist, sys_q), temperature, max_tokens, qbuf))
    tk.start(); tq.start()

    while not (kbuf["done"] and qbuf["done"]):
        k_hist[-1]["content"] = kbuf["text"] or "▌"
        q_hist[-1]["content"] = qbuf["text"] or "▌"
        yield k_hist, q_hist, ""
        tk.join(timeout=0.15)
    k_hist[-1]["content"] = kbuf["text"]
    q_hist[-1]["content"] = qbuf["text"]
    yield k_hist, q_hist, ""


with gr.Blocks(title="Keural v1.0 vs Qwen3.6 base — side by side", fill_height=True) as demo:
    gr.Markdown("# ⚖️ Keural Nova v1.0  vs  Qwen3.6-35B-A3B (base)\n"
                "같은 질문을 두 모델에 동시에 보내고 응답을 나란히 비교하세요. "
                "서빙: 각 모델 1× H200 · non-thinking")
    with gr.Accordion("설정 (settings)", open=False):
        with gr.Row():
            sys_k = gr.Textbox(value=SYS_KEURAL, label="Keural system prompt", lines=2)
            sys_q = gr.Textbox(value=SYS_QWEN, label="Qwen system prompt", lines=2)
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
            max_tokens = gr.Slider(64, 2048, value=768, step=64, label="max_tokens")
    with gr.Row():
        k_chat = gr.Chatbot(label="🟢 Keural Nova v1.0", type="messages", height=460)
        q_chat = gr.Chatbot(label="🔵 Qwen3.6-35B-A3B (base)", type="messages", height=460)
    with gr.Row():
        box = gr.Textbox(placeholder="메시지를 입력하고 Enter — 두 모델에 동시 전송",
                         scale=8, show_label=False, autofocus=True)
        clear = gr.Button("🗑 Clear", scale=1)

    box.submit(chat, [box, k_chat, q_chat, sys_k, sys_q, temperature, max_tokens],
               [k_chat, q_chat, box])
    clear.click(lambda: ([], [], ""), None, [k_chat, q_chat, box])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=UI_PORT, share=False)
