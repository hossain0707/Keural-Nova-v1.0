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
 {"type":"function","function":{"name":"calculator","description":"Evaluate a math expression.",
   "parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
 {"type":"function","function":{"name":"get_stock_price","description":"Get the price of a stock/crypto ticker.",
   "parameters":{"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}}},
 {"type":"function","function":{"name":"get_current_time","description":"Get the current time in a timezone.",
   "parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}},
]
CASES=[("서울 날씨 어때?","get_weather"),("What's the weather in Tokyo right now?","get_weather"),
 ("비트코인 현재 가격 좀 알려줘.","get_stock_price"),("What is the current price of TSLA?","get_stock_price"),
 ("최신 인공지능 뉴스 좀 검색해줘.","web_search"),("Search the web for the latest iPhone release.","web_search"),
 ("1234 곱하기 5678은 얼마야?","calculator"),("Compute 987 * 654 for me.","calculator"),
 ("지금 뉴욕 시간 몇 시야?","get_current_time"),("What time is it in London right now?","get_current_time"),
 ("부산 날씨 알려줘.","get_weather"),("삼성전자 주가 알려줘.","get_stock_price")]
MODELS=[("qwen-base","Qwen/Qwen3.6-35B-A3B"),
 ("nova-v1.0","/home/work/keural MoE Pro/outputs/merged/Keural-35B-A3B-ckpt7500"),
 ("nova-v1.1","/home/work/keural MoE Pro/outputs/merged/Keural-35B-A3B")]
def score(text, expected):
    # Qwen XML tool-call: <tool_call>...<function=NAME>...
    m=re.search(r"<tool_call>.*?<function\s*=\s*([A-Za-z_]+)", text, re.S)
    if m: return True,(m.group(1)==expected)
    # JSON style: <tool_call>{"name":"NAME","arguments":...}
    m=re.search(r'<tool_call>\s*\{.*?"name"\s*:\s*"([A-Za-z_]+)"', text, re.S)
    if m: return True,(m.group(1)==expected)
    # bare <function=NAME>
    m=re.search(r"<function\s*=\s*([A-Za-z_]+)", text)
    if m: return True,(m.group(1)==expected)
    return False,False

results={}
for tag,path in MODELS:
    print(f"[{time.strftime('%H:%M:%S')}] loading {tag} ...",flush=True)
    tok=AutoTokenizer.from_pretrained(path)
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map="auto").eval()
    rows=[]
    for i,(q,exp) in enumerate(CASES):
        try:
            s=tok.apply_chat_template([{"role":"user","content":q}],tools=TOOLS,tokenize=False,
                                      add_generation_prompt=True,enable_thinking=False)
        except TypeError:
            s=tok.apply_chat_template([{"role":"user","content":q}],tools=TOOLS,tokenize=False,
                                      add_generation_prompt=True)
        ids=tok(s,return_tensors="pt").to(model.device)
        with torch.no_grad():
            out=model.generate(**ids,max_new_tokens=512,do_sample=False)
        resp=tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=False)
        emit,correct=score(resp,exp)
        rows.append({"q":q,"expected":exp,"emit":emit,"correct":correct,"resp":resp[:600]})
        print(f"[{time.strftime('%H:%M:%S')}] {tag} {i+1}/{len(CASES)} emit={emit} correct={correct}",flush=True)
    n=len(CASES)
    results[tag]={"emit_rate":sum(r['emit'] for r in rows)/n,"correct_rate":sum(r['correct'] for r in rows)/n,"rows":rows}
    del model,tok; torch.cuda.empty_cache()
    json.dump(results,open(OUT+"/toolcall_results.json","w"),ensure_ascii=False,indent=1)

print("\n=== TOOL-CALLING PASS RATES (%d cases each) ===" % len(CASES))
print(f"{'model':12}{'emits tool_call':>18}{'correct function':>18}")
for tag,_ in MODELS:
    r=results.get(tag,{})
    if r: print(f"{tag:12}{r['emit_rate']*100:>16.0f}%{r['correct_rate']*100:>17.0f}%")
print("DONE",flush=True)
