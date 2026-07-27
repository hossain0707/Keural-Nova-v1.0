import os, json, re, time
os.environ["PYTHONNOUSERSITE"]="1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES","0,1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT="/home/work/keural MoE Pro/nova_toolcall_eval"
TOOLS=[
 {"type":"function","function":{"name":"get_weather","description":"Get current weather for a city.",
   "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
 {"type":"function","function":{"name":"web_search","description":"Search the web for a query.",
   "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
 {"type":"function","function":{"name":"get_stock_price","description":"Get the price of a stock/crypto ticker.",
   "parameters":{"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}}},
 {"type":"function","function":{"name":"get_current_time","description":"Get the current time in a timezone.",
   "parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}},
]
# (query, expected_fn, canned_args, canned_tool_result, [key facts that MUST appear in the final answer])
SCEN=[
 ("서울 날씨 어때?","get_weather",{"city":"서울"},'{"city":"서울","condition":"맑음","temp_c":12}',["12","맑"]),
 ("What's the weather in Tokyo?","get_weather",{"city":"Tokyo"},'{"city":"Tokyo","condition":"Rainy","temp_c":18}',["18","ain"]),
 ("테슬라 주가 알려줘.","get_stock_price",{"ticker":"TSLA"},'{"ticker":"TSLA","price_usd":245.30}',["245"]),
 ("비트코인 가격 알려줘.","get_stock_price",{"ticker":"BTC"},'{"ticker":"BTC","price_usd":67000}',["67000","67,000","6.7","6만7"]),
 ("지금 뉴욕 몇 시야?","get_current_time",{"timezone":"America/New_York"},'{"timezone":"America/New_York","time":"09:15 AM"}',["9:15","09:15"]),
 ("최신 AI 뉴스 검색해줘.","web_search",{"query":"최신 AI 뉴스"},'{"results":[{"title":"GPT-5 공개","snippet":"OpenAI가 GPT-5를 발표했다."}]}',["GPT-5","GPT"]),
]
MODELS=[("base Qwen","Qwen/Qwen3.6-35B-A3B"),
        ("Nova v1.1","/home/work/keural MoE Pro/outputs/merged/Keural-35B-A3B")]

def toolcall_fn(text):
    m=re.search(r"<tool_call>.*?<function\s*=\s*([A-Za-z_]+)", text, re.S)
    if m: return m.group(1)
    m=re.search(r'<tool_call>\s*\{.*?"name"\s*:\s*"([A-Za-z_]+)"', text, re.S)
    return m.group(1) if m else None

def render(tok,msgs):
    try: return tok.apply_chat_template(msgs,tools=TOOLS,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs,tools=TOOLS,tokenize=False,add_generation_prompt=True)

def gen(tok,model,text,mx=400):
    ids=tok(text,return_tensors="pt").to(model.device)
    with torch.no_grad(): out=model.generate(**ids,max_new_tokens=mx,do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=False)

res={}
for tag,path in MODELS:
    print(f"[{time.strftime('%H:%M:%S')}] loading {tag} ...",flush=True)
    tok=AutoTokenizer.from_pretrained(path)
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map="auto").eval()
    rows=[]
    for i,(q,exp,args,result,keys) in enumerate(SCEN):
        # TURN 1: user query -> tool call
        r1=gen(tok,model,render(tok,[{"role":"user","content":q}]))
        t1_fn=toolcall_fn(r1); t1_ok=(t1_fn==exp)
        # TURN 2: feed the tool result back, expect a final answer that USES it
        msgs=[{"role":"user","content":q},
              {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":exp,"arguments":args}}]},
              {"role":"tool","content":result}]
        r2=gen(tok,model,render(tok,msgs))
        used = any(k.lower() in r2.lower() for k in keys)
        looped = "<tool_call>" in r2
        t2_ok = used and not looped
        rows.append({"q":q,"t1_fn":t1_fn,"t1_ok":t1_ok,"t2_used_result":used,"t2_looped":looped,"t2_ok":t2_ok,
                     "final":r2.replace("<|im_end|>","").strip()[:220]})
        print(f"[{time.strftime('%H:%M:%S')}] {tag} {i+1}/{len(SCEN)} turn1_ok={t1_ok} turn2_ok={t2_ok} (used={used} looped={looped})",flush=True)
    n=len(SCEN)
    res[tag]={"turn1":sum(r['t1_ok'] for r in rows),"turn2":sum(r['t2_ok'] for r in rows),"n":n,"rows":rows}
    del model,tok; torch.cuda.empty_cache()
    json.dump(res,open(OUT+"/multiturn_results.json","w"),ensure_ascii=False,indent=1)

print("\n=== MULTI-TURN AGENTIC LOOP (non-thinking) ===")
for tag,_ in MODELS:
    r=res[tag]; print(f"{tag:11} Turn1 emit-correct {r['turn1']}/{r['n']} | Turn2 uses-result-and-answers {r['turn2']}/{r['n']}")
print("DONE",flush=True)
