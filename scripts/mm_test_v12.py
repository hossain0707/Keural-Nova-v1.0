#!/usr/bin/env python3
"""Verify Keural Nova v1.2 multimodal (vision) capability with a real image inference."""
import os, json, torch
from PIL import Image, ImageDraw

MODEL="/home/work/keural MoE Pro/outputs/v12/merged/Keural-Nova-v1.2-experimental"
SCRATCH="/tmp/claude-1100/-home-work-keural-MoE-Pro/431b81d4-e612-4f11-8c10-2c4f04a4892e/scratchpad"

# ---- structural check ----
cfg=json.load(open(os.path.join(MODEL,"config.json")))
print("has vision_config:", "vision_config" in cfg)
print("preprocessor present:", os.path.isfile(os.path.join(MODEL,"preprocessor_config.json")))
import safetensors, glob
vis=0
idx=json.load(open(os.path.join(MODEL,"model.safetensors.index.json")))
vis=sum(1 for k in idx["weight_map"] if "visual" in k or "vision" in k)
print(f"vision tensors in index: {vis}")

# ---- build a clear test image: blue circle (left) + red square (right) on white ----
img=Image.new("RGB",(512,512),"white")
d=ImageDraw.Draw(img)
d.ellipse([40,180,220,360], fill=(0,80,255))       # blue circle
d.rectangle([300,180,470,360], fill=(230,30,30))   # red square
p=os.path.join(SCRATCH,"mm_test.png"); img.save(p); print("test image:", p)

# ---- load as vision-language model (try the right class) ----
from transformers import AutoProcessor
proc=AutoProcessor.from_pretrained(MODEL)
model=None; err=[]
for cls in ["AutoModelForImageTextToText","AutoModelForVision2Seq","AutoModelForCausalLM"]:
    try:
        import transformers
        M=getattr(transformers, cls)
        model=M.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
        print(f"loaded via {cls}"); break
    except Exception as e:
        err.append(f"{cls}: {type(e).__name__} {str(e)[:80]}")
if model is None:
    print("MODEL LOAD FAILED:"); [print("  ",e) for e in err]; raise SystemExit(1)

msgs=[{"role":"user","content":[
    {"type":"image","image":img},
    {"type":"text","text":"이 이미지에 어떤 도형과 색깔이 보이나요? 영어로도 답해줘."}]}]
inputs=proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                return_dict=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    out=model.generate(**inputs, max_new_tokens=200, do_sample=False)
resp=proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n===== VISION RESPONSE =====")
print(resp)
low=resp.lower()
hits=[w for w in ["blue","파란","파랑","circle","원","red","빨간","빨강","square","사각","네모"] if w in low or w in resp]
print("\ncontent hits:", hits)
print("VISION WORKS:", ("blue" in low or "파란" in resp or "파랑" in resp) and ("red" in low or "빨" in resp))
