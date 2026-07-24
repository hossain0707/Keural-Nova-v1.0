# Keural Nova v1.0 — interactive chat test

Everything here is for **testing the fine-tuned model's responses** interactively.
It serves **Keural Nova v1.0** (checkpoint-7500) with vLLM on both H200s and gives you
two front-ends: a web chatbot and a terminal client.

Model source: defaults to the local merged weights
(`../outputs/merged/Keural-35B-A3B-ckpt7500`), which are byte-identical to
[`mkd-hossain/Keural-Nova-v1.0`](https://huggingface.co/mkd-hossain/Keural-Nova-v1.0)
— so no 72 GB download. Set `KEURAL_MODEL=mkd-hossain/Keural-Nova-v1.0` to pull from the
Hub instead.

## Run it (two panes)

Use tmux so it survives disconnects (`/opt/kernel/tmux`).

**Pane 1 — start the model server** (takes ~3 min to load, wait for `Application startup complete`):
```bash
cd "/home/work/keural MoE Pro/test model"
bash serve_vllm.sh
```

**Pane 2 — pick a front-end:**

Web chatbot (open the KT Cloud proxy at `/proxy/7861`):
```bash
cd "/home/work/keural MoE Pro/test model"
source "/home/work/keural MoE Pro/venv/bin/activate" && export PYTHONPATH=
python chatbot.py
```

…or terminal chat (no browser):
```bash
cd "/home/work/keural MoE Pro/test model"
source "/home/work/keural MoE Pro/venv/bin/activate" && export PYTHONPATH=
python chat_cli.py
```

## Files

| File | What it is |
|---|---|
| `serve_vllm.sh` | Starts the vLLM OpenAI-compatible server (port 8001, model name `keural-v1.0`) |
| `chatbot.py` | Gradio streaming web chat — multi-turn, adjustable temperature/top_p, Korean examples |
| `chat_cli.py` | Terminal chat client — `/reset`, `/system <text>`, `quit` |

## Notes

- The model uses the base model's **non-thinking** chat format; both clients set
  `enable_thinking=False` automatically.
- **Not for code generation** — v1.0 has a known code regression (see the main model card).
  It's tuned for Korean conversation/comprehension; test it there.
- Env vars: `KEURAL_MODEL` (weights), `KEURAL_PORT` (server, default 8001),
  `KEURAL_API` (client → server URL), `KEURAL_UIPORT` (gradio, default 7861).
