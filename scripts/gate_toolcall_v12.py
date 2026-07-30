#!/usr/bin/env python3
"""v1.2 tool-calling gate on the merged model (vs base parity target).

Runs: 12 single-turn cases, 1 multi-turn (call->result->answer), 1 long-32k case.
Loads the merged v1.2 model via transformers, non-thinking mode. Writes JSON.
"""
import os, json, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL="/home/work/keural MoE Pro/outputs/v12/merged/Keural-Nova-v1.2-experimental"
OUT="/home/work/keural MoE Pro/outputs/v12/gate_toolcall.json"

TOOLS=[
 {"type":"function","function":{"name":"get_weather","description":"Get current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
 {"type":"function","function":{"name":"web_search","description":"Search the web for a query.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
 {"type":"function","function":{"name":"calculator","description":"Evaluate a math expression.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
 {"type":"function","function":{"name":"get_stock_price","description":"Get the price of a stock/crypto ticker.","parameters":{"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}}},
 {"type":"function","function":{"name":"get_current_time","description":"Get the current time in a timezone.","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}},
 {"type":"function","function":{"name":"lookup_order","description":"Look up an order by its ID.","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}}},
]
CASES=[("서울 날씨 어때?","get_weather"),("What's the weather in Tokyo right now?","get_weather"),
 ("비트코인 현재 가격 좀 알려줘.","get_stock_price"),("What is the current price of TSLA?","get_stock_price"),
 ("최신 인공지능 뉴스 좀 검색해줘.","web_search"),("Search the web for the latest iPhone release.","web_search"),
 ("1234 곱하기 5678은 얼마야?","calculator"),("Compute 987 * 654 for me.","calculator"),
 ("지금 뉴욕 시간 몇 시야?","get_current_time"),("What time is it in London right now?","get_current_time"),
 ("부산 날씨 알려줘.","get_weather"),("삼성전자 주가 알려줘.","get_stock_price")]

def parse_call(t):
    m=re.search(r"<tool_call>.*?<function\s*=\s*([A-Za-z_]+)", t, re.S); return m.group(1) if m else None

def gen(model,tok,msgs,tools,max_new=256):
    s=tok.apply_chat_template(msgs,tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok(s,return_tensors="pt").to(model.device)
    with torch.no_grad():
        out=model.generate(**ids,max_new_tokens=max_new,do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=False)

def main():
    print(f"[{time.strftime('%H:%M:%S')}] loading merged v1.2 ...",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="auto").eval()
    res={}

    # 1) single-turn 12-case
    ok=0; rows=[]
    for q,exp in CASES:
        r=gen(model,tok,[{"role":"user","content":q}],TOOLS)
        fn=parse_call(r); c=(fn==exp); ok+=c; rows.append({"q":q,"exp":exp,"got":fn,"ok":c})
    res["single_turn"]={"correct":ok,"n":len(CASES),"rows":rows}
    print(f"[{time.strftime('%H:%M:%S')}] single-turn: {ok}/{len(CASES)}",flush=True)

    # 2) multi-turn: call -> tool result -> final answer uses it
    msgs=[{"role":"user","content":"서울 날씨 알려줘."}]
    r1=gen(model,tok,msgs,TOOLS); fn=parse_call(r1)
    msgs+=[{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"city":"서울"}}}]},
           {"role":"tool","content":json.dumps({"city":"서울","condition":"맑음","temp_c":12},ensure_ascii=False)}]
    r2=gen(model,tok,msgs,TOOLS)
    uses = ("12" in r2) and ("맑" in r2 or "맑음" in r2)
    res["multi_turn"]={"call_emitted":fn=="get_weather","final_uses_result":bool(uses),"final":r2[:200]}
    print(f"[{time.strftime('%H:%M:%S')}] multi-turn: call={fn=='get_weather'} uses_result={bool(uses)}",flush=True)

    # 3) long 32k-ish case: needle order id buried in a long log, ask to look it up
    needle="ORD-778213"
    filler=("\n".join(f"[log] event {i}: routine status ok, nothing to report here." for i in range(6000)))
    long_ctx=f"Here is a long system log.\n{filler}\nIMPORTANT RECORD: the customer's order id is {needle}.\n{filler}\n"
    q=f"{long_ctx}\n\nUsing the order id mentioned in the log above, look up its status."
    ntok=len(tok(q).input_ids)
    r=gen(model,tok,[{"role":"user","content":q}],TOOLS,max_new=128)
    fn=parse_call(r); has_needle=needle in r
    res["long_32k"]={"approx_tokens":ntok,"tool":fn,"tool_is_lookup":fn=="lookup_order","arg_has_needle":has_needle,"snippet":r[:200]}
    print(f"[{time.strftime('%H:%M:%S')}] long(~{ntok} tok): tool={fn} needle_in_call={has_needle}",flush=True)

    json.dump(res,open(OUT,"w"),ensure_ascii=False,indent=1)
    print(f"\nSINGLE-TURN {res['single_turn']['correct']}/{res['single_turn']['n']} | MULTI-TURN uses_result={res['multi_turn']['final_uses_result']} | LONG32K tool={res['long_32k']['tool']} needle={res['long_32k']['arg_has_needle']}",flush=True)
    print(f"written -> {OUT}",flush=True)

if __name__=="__main__": main()
