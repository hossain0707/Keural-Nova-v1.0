"""Quick multimodal (vision) check for Keural Nova.

Keural Nova already contains the base model's vision encoder (preserved during
merge). This script tests whether image-understanding actually works after the
Korean text fine-tune — a capability we never trained or evaluated, so this is
a genuine unknown until you run it.

Needs a FREE GPU (~70 GB model in bf16 fits one H200). The public gateway holds
both GPUs, so stop one backend first (or run when GPUs are free).

Usage:
  python vision_test.py --image <path-or-URL> --prompt "이 이미지를 설명해줘"
  python vision_test.py --image cat.jpg          # uses a default Korean prompt
  KEURAL_MODEL=outputs/merged/Keural-35B-A3B-ckpt7500 python vision_test.py --image ...
"""
import argparse
import os

MODEL = os.environ.get(
    "KEURAL_MODEL",
    "/home/work/keural MoE Pro/outputs/merged/Keural-35B-A3B-ckpt7500")  # v1.0 by default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="local path or http(s) URL")
    ap.add_argument("--prompt", default="이 이미지에 무엇이 보이나요? 자세히 설명해 주세요.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info

    print(f"Loading {MODEL} on {args.device} (~70GB, a few min)...")
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": args.image},
        {"type": "text", "text": args.prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(args.device)

    print("Generating...")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(gen, skip_special_tokens=True)
    print("\n" + "=" * 60)
    print("IMAGE :", args.image)
    print("PROMPT:", args.prompt)
    print("ANSWER:", answer.strip())
    print("=" * 60)
    print("\nIf the answer describes the image sensibly -> vision works.")
    print("If it's gibberish / ignores the image -> the text fine-tune degraded vision.")


if __name__ == "__main__":
    main()
