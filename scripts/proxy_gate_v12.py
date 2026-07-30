#!/usr/bin/env python3
"""Fast proxy gate for v1.2 candidate checkpoints — tool-calling pass rate.

Loads base Qwen once, applies each preserved LoRA adapter (no full merge, no disk),
runs the 12-case XML tool-call eval in NON-THINKING mode, and ranks the checkpoints.
Cheap way to pick the tool-calling winner before the expensive full merge + gate.

Writes: outputs/v12/proxy_gate.json  (+ prints a table)
Deletes nothing.
"""
import os, json, re, time, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3.6-35B-A3B"
TOK_SRC = "/home/work/keural MoE Pro/outputs/merged/Keural-35B-A3B"  # known-good fixed (non-thinking) template
KEPT = "/home/work/keural MoE Pro/outputs/v12/kept"
CANDIDATES = [3000, 4000, 5000, 5068]
OUT = "/home/work/keural MoE Pro/outputs/v12/proxy_gate.json"

TOOLS=[
 {"type":"function","function":{"name":"get_weather","description":"Get current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
 {"type":"function","function":{"name":"web_search","description":"Search the web for a query.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
 {"type":"function","function":{"name":"calculator","description":"Evaluate a math expression.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
 {"type":"function","function":{"name":"get_stock_price","description":"Get the price of a stock/crypto ticker.","parameters":{"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}}},
 {"type":"function","function":{"name":"get_current_time","description":"Get the current time in a timezone.","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}},
]
CASES=[("서울 날씨 어때?","get_weather"),("What's the weather in Tokyo right now?","get_weather"),
 ("비트코인 현재 가격 좀 알려줘.","get_stock_price"),("What is the current price of TSLA?","get_stock_price"),
 ("최신 인공지능 뉴스 좀 검색해줘.","web_search"),("Search the web for the latest iPhone release.","web_search"),
 ("1234 곱하기 5678은 얼마야?","calculator"),("Compute 987 * 654 for me.","calculator"),
 ("지금 뉴욕 시간 몇 시야?","get_current_time"),("What time is it in London right now?","get_current_time"),
 ("부산 날씨 알려줘.","get_weather"),("삼성전자 주가 알려줘.","get_stock_price")]

def parse_call(text):
    m=re.search(r"<tool_call>.*?<function\s*=\s*([A-Za-z_]+)", text, re.S)
    return m.group(1) if m else None

def main():
    print(f"[{time.strftime('%H:%M:%S')}] loading base {BASE} ...", flush=True)
    tok=AutoTokenizer.from_pretrained(TOK_SRC)
    base=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto").eval()

    # attach all adapters under distinct names
    names={}
    model=None
    for c in CANDIDATES:
        d=os.path.join(KEPT, f"checkpoint-{c}")
        if not os.path.isfile(os.path.join(d,"adapter_model.safetensors")):
            print(f"  ! missing adapter for checkpoint-{c}, skipping", flush=True); continue
        nm=f"c{c}"
        if model is None:
            model=PeftModel.from_pretrained(base, d, adapter_name=nm)
        else:
            model.load_adapter(d, adapter_name=nm)
        names[c]=nm
        print(f"  loaded adapter checkpoint-{c} as {nm}", flush=True)
    model.eval()

    res={}
    for c,nm in names.items():
        model.set_adapter(nm)
        ok=0; rows=[]
        for q,exp in CASES:
            s=tok.apply_chat_template([{"role":"user","content":q}],tools=TOOLS,tokenize=False,
                                      add_generation_prompt=True,enable_thinking=False)
            ids=tok(s,return_tensors="pt").to(model.device)
            with torch.no_grad():
                out=model.generate(**ids,max_new_tokens=256,do_sample=False)
            resp=tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=False)
            fn=parse_call(resp); correct=(fn==exp); ok+=correct
            rows.append({"q":q,"exp":exp,"got":fn,"correct":correct})
        res[f"checkpoint-{c}"]={"toolcall_correct":ok,"n":len(CASES),"rows":rows}
        print(f"[{time.strftime('%H:%M:%S')}] checkpoint-{c}: tool-calling {ok}/{len(CASES)}", flush=True)
        json.dump(res, open(OUT,"w"), ensure_ascii=False, indent=1)

    print("\n=== PROXY GATE RESULT (tool-calling) ===", flush=True)
    for k,v in sorted(res.items(), key=lambda x:-x[1]["toolcall_correct"]):
        print(f"  {k}: {v['toolcall_correct']}/{v['n']}", flush=True)
    print(f"\nwritten -> {OUT}", flush=True)

if __name__=="__main__":
    main()
