#!/usr/bin/env python3
"""Build the Keural Nova tool-calling SFT slice (short + long-context).

Two output files, both in the ms-swift 4.4.0 NATIVE agent schema (see the schema
note at the bottom of this docstring and data/tooluse/README.md):

  data/tooluse/tooluse_short.jsonl    ~15k rows, mixed tool-use tasks
  data/tooluse/tooluse_long32k.jsonl  ~3k rows, same tasks in 8k-32k context

WHY THIS SCHEMA (verified against this repo's merged model + ms-swift 4.4.0):
  * Qwen/Qwen3.6-35B-A3B -> ms-swift TemplateType.qwen3_5 -> agent_template
    'qwen3_5'. That agent template renders tool calls as Qwen XML
    <tool_call><function=NAME><parameter=P>VAL</parameter></function></tool_call>
    BYTE-IDENTICALLY to the model's own chat_template.jinja (confirmed by
    rendering both). No --agent_template override is needed at train time.
  * ms-swift does NOT read the OpenAI-style `tool_calls` field on assistant
    messages. Its dataset preprocessor (swift/dataset/preprocessor/core.py
    _check_messages) POPS every message key except {role, content, loss,
    loss_scale}. Tool calls MUST therefore be their own message with
    role == "tool_call" and content == a JSON string {"name","arguments"}.
    Tool results use role == "tool" (or "tool_response"; equivalent).
  * The function schemas are carried in a per-row top-level `tools` field,
    stored as a JSON STRING (ms-swift _preprocess_tools parses str -> list).
    A string keeps the Arrow/`datasets` column flat — nested tool objects have
    per-row-varying `parameters` shapes and would break schema inference.
  * NO manual <think> scaffold is written into content. The qwen3_5 template
    injects the empty `<think>\n\n</think>` itself for the non-thinking path,
    and train_full.yaml's `loss_scale: ignore_empty_think` masks it. Writing a
    <think> prefix into a tool_call's JSON content would also break parsing.

Sources (both Apache-2.0 -> commercially usable):
  glaiveai/glaive-function-calling-v2      (single/multi-turn + natural negatives)
  NousResearch/hermes-function-calling-v1  (multi-tool / agentic variety)
Salesforce/xlam-function-calling-60k is gated -> skipped (kept reproducible).

Korean tool-use rows are hand-built here (weather/stock/search/calculator);
neither open source ships Korean function-calling data.

CPU / data-only. Never imports torch, never touches the GPUs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

PROJECT = Path("/home/work/keural MoE Pro")
MERGED = PROJECT / "outputs/merged/Keural-35B-A3B"
OUT_DIR = PROJECT / "data/tooluse"

GLAIVE_REPO = "glaiveai/glaive-function-calling-v2"
GLAIVE_REV = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"
GLAIVE_FILE = "glaive-function-calling-v2.json"
HERMES_REPO = "NousResearch/hermes-function-calling-v1"
HERMES_REV = "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
HERMES_FILES = ["func-calling-singleturn.json", "func-calling.json"]

VALID_ROLES = ("system", "user", "assistant", "tool_call", "tool")
SEED = 42

# ------------------------------------------------------------------ helpers


def wrap_tool(tool: dict) -> dict:
    """Ensure OpenAI wrapped form {"type":"function","function":{...}}."""
    if isinstance(tool, dict) and "type" in tool and "function" in tool:
        return tool
    return {"type": "function", "function": tool}


def tool_names(tools: list) -> set:
    out = set()
    for t in tools:
        fn = t.get("function", t)
        if isinstance(fn, dict) and fn.get("name"):
            out.add(fn["name"])
    return out


def mk_toolcall_msg(name: str, arguments: dict) -> dict:
    return {"role": "tool_call",
            "content": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}


def trim_trailing(messages: list) -> list:
    """Drop trailing turns that aren't a trainable assistant-side turn."""
    while messages and messages[-1]["role"] not in ("assistant", "tool_call"):
        messages.pop()
    return messages


def row_ok(messages: list, tools: list) -> bool:
    """Structural gate mirroring build_dataset.validate_messages, tool-aware."""
    if not messages:
        return False
    names = tool_names(tools)
    has_user = any(m["role"] == "user" for m in messages)
    has_trainable = any(m["role"] in ("assistant", "tool_call") for m in messages)
    if not (has_user and has_trainable):
        return False
    if messages[-1]["role"] not in ("assistant", "tool_call"):
        return False
    for m in messages:
        if m["role"] not in VALID_ROLES:
            return False
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return False
        if m["role"] == "tool_call":
            try:
                obj = json.loads(m["content"])
            except json.JSONDecodeError:
                return False
            if not isinstance(obj.get("name"), str) or not isinstance(obj.get("arguments"), dict):
                return False
            if names and obj["name"] not in names:
                return False
    return True


def make_row(messages: list, tools: list) -> Optional[dict]:
    messages = trim_trailing(list(messages))
    if not row_ok(messages, tools):
        return None
    return {"messages": messages,
            "tools": json.dumps([wrap_tool(t) for t in tools], ensure_ascii=False)}


# ------------------------------------------------------------------ glaive


def parse_glaive_system(system: str) -> list:
    """Return list of (unwrapped) function schemas from a glaive system string."""
    marker = "access to the following functions"
    if marker not in system:
        return []  # "no access to external functions" -> negative conv
    body = system.split(" -", 1)[1] if " -" in system else ""
    tools, dec, i, n = [], json.JSONDecoder(), 0, len(body)
    while i < n:
        while i < n and body[i] in " \n\r\t":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(body, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and obj.get("name"):
            tools.append(obj)
        i = end
    return tools


_GLAIVE_SPLIT = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")


def _parse_functioncall(text: str) -> Optional[tuple]:
    idx = text.find("<functioncall>")
    fc = text[idx + len("<functioncall>"):].strip()
    fc = fc.replace("<|endoftext|>", "").strip()
    m = re.search(r'"name"\s*:\s*"([^"]*)"', fc)
    if not m:
        return None
    name = m.group(1)
    am = re.search(r'"arguments"\s*:\s*\'(.*)\'\s*}\s*$', fc, re.DOTALL)
    if am:
        raw = am.group(1)
    else:
        am2 = re.search(r'"arguments"\s*:\s*(\{.*\})\s*}\s*$', fc, re.DOTALL)
        raw = am2.group(1) if am2 else "{}"
    try:
        args = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    return name, args


def parse_glaive_chat(chat: str) -> Optional[list]:
    parts = _GLAIVE_SPLIT.split(chat)
    messages: list = []
    i = 1
    while i < len(parts) - 1:
        marker, content = parts[i], parts[i + 1]
        content = content.replace("<|endoftext|>", "").strip()
        i += 2
        if not content and marker != "ASSISTANT:":
            continue
        if marker == "USER:":
            messages.append({"role": "user", "content": content})
        elif marker == "FUNCTION RESPONSE:":
            messages.append({"role": "tool", "content": content})
        else:  # ASSISTANT:
            if "<functioncall>" in content:
                pre = content[:content.find("<functioncall>")].strip()
                parsed = _parse_functioncall(content)
                if parsed is None:
                    return None
                if pre:  # rare leading prose -> keep as its own assistant turn
                    messages.append({"role": "assistant", "content": pre})
                messages.append(mk_toolcall_msg(parsed[0], parsed[1]))
            elif content:
                messages.append({"role": "assistant", "content": content})
    return messages


def iter_glaive(path: Path) -> Iterator[dict]:
    import ijson
    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            yield item


# ------------------------------------------------------------------ hermes

_HERMES_TOOLS = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
_HERMES_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_HERMES_TR = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def parse_hermes_conv(item: dict) -> Optional[tuple]:
    conv = item.get("conversations") or item.get("messages") or []
    tools: Optional[list] = None
    # Preferred: the item ships a top-level `tools` JSON string (already wrapped).
    raw_tools = item.get("tools")
    if isinstance(raw_tools, str) and raw_tools.strip():
        try:
            tools = json.loads(raw_tools)
        except json.JSONDecodeError:
            tools = None
    messages: list = []
    for turn in conv:
        frm = turn.get("from") or turn.get("role")
        val = turn.get("value") or turn.get("content") or ""
        if frm == "system":
            if tools is None:
                m = _HERMES_TOOLS.search(val)
                if m:
                    try:
                        tools = json.loads(m.group(1))
                    except json.JSONDecodeError:
                        return None
        elif frm in ("human", "user"):
            if val.strip():
                messages.append({"role": "user", "content": val.strip()})
        elif frm in ("gpt", "assistant", "model"):
            tcs = _HERMES_TC.findall(val)
            if tcs:
                for tc in tcs:
                    try:
                        obj = json.loads(tc)
                    except json.JSONDecodeError:
                        return None
                    name = obj.get("name")
                    args = obj.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    if not isinstance(name, str) or not isinstance(args, dict):
                        return None
                    messages.append(mk_toolcall_msg(name, args))
            elif val.strip():
                messages.append({"role": "assistant", "content": val.strip()})
        elif frm in ("tool", "observation", "tool_response"):
            m = _HERMES_TR.search(val)
            content = (m.group(1) if m else val).strip()
            if content:
                messages.append({"role": "tool", "content": content})
    if tools is None or not isinstance(tools, list) or not tools:
        return None
    return messages, tools


# ------------------------------------------------------------------ Korean (hand-built)

KO_TOOLS = {
    "get_weather": {"name": "get_weather", "description": "특정 도시의 현재 날씨를 조회합니다.",
                    "parameters": {"type": "object", "properties": {
                        "city": {"type": "string", "description": "도시 이름 (예: 서울)"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                                 "description": "온도 단위"}}, "required": ["city"]}},
    "get_stock_price": {"name": "get_stock_price", "description": "종목의 현재 주가를 조회합니다.",
                        "parameters": {"type": "object", "properties": {
                            "ticker": {"type": "string", "description": "종목 코드 또는 회사명"},
                            "market": {"type": "string", "description": "시장 (KOSPI, NASDAQ 등)"}},
                            "required": ["ticker"]}},
    "web_search": {"name": "web_search", "description": "웹에서 최신 정보를 검색합니다.",
                   "parameters": {"type": "object", "properties": {
                       "query": {"type": "string", "description": "검색어"},
                       "top_k": {"type": "integer", "description": "가져올 결과 수"}},
                       "required": ["query"]}},
    "calculate": {"name": "calculate", "description": "산술 수식을 계산합니다.",
                  "parameters": {"type": "object", "properties": {
                      "expression": {"type": "string", "description": "계산할 수식 (예: 1234*5678)"}},
                      "required": ["expression"]}},
}

KO_CITIES = ["서울", "부산", "인천", "대구", "광주", "대전", "제주", "수원", "울산", "성남",
             "도쿄", "뉴욕", "런던", "파리", "베이징"]
KO_TICKERS = [("삼성전자", "KOSPI"), ("SK하이닉스", "KOSPI"), ("현대차", "KOSPI"),
              ("카카오", "KOSPI"), ("네이버", "KOSPI"), ("테슬라", "NASDAQ"),
              ("애플", "NASDAQ"), ("엔비디아", "NASDAQ")]
KO_SEARCH = ["2026년 최저임금", "이번 주 개봉 영화", "가까운 국경일", "환율 전망",
             "전기차 보조금 정책", "올해 벚꽃 개화 시기", "최신 아이폰 출시일"]
KO_EXPR = [("1234 곱하기 5678", "1234*5678"), ("987654 나누기 321", "987654/321"),
           ("2의 15제곱", "2**15"), ("45와 78의 합에 12를 곱한 값", "(45+78)*12"),
           ("15만원의 8% 부가세", "150000*0.08")]
KO_WEATHER_Q = ["지금 {c} 날씨 어때?", "{c} 오늘 날씨 알려줘.", "{c} 현재 기온이 궁금해.",
                "{c} 밖에 나가도 될 만한 날씨야?"]
KO_STOCK_Q = ["{t} 주가 지금 얼마야?", "{t} 현재 주가 알려줘.", "{t} 오늘 얼마에 거래되고 있어?"]
KO_SEARCH_Q = ["{q} 검색해줘.", "{q}에 대해 최신 정보 찾아줘.", "{q} 관련해서 알아봐 줘."]
KO_NEG_QA = [
    ("안녕하세요! 오늘 기분이 어때요?", "안녕하세요! 저는 도움을 드릴 준비가 되어 있어요. 무엇을 도와드릴까요?"),
    ("고마워, 도움이 많이 됐어.", "천만에요! 더 궁금한 점이 있으면 언제든지 물어봐 주세요."),
    ("대한민국의 수도는 어디야?", "대한민국의 수도는 서울입니다."),
    ("물은 몇 도에서 끓어?", "표준 대기압(1기압)에서 물은 섭씨 100도에서 끓습니다."),
    ("좋은 아침 인사말 하나 추천해줘.", "\"오늘도 좋은 하루 보내세요!\" 같은 인사말은 어떨까요?"),
    ("1년은 몇 개월이야?", "1년은 12개월입니다."),
]


def gen_korean(n: int, rng: random.Random) -> list:
    rows, seen = [], set()
    all_tools = list(KO_TOOLS.values())

    def push(messages, tools):
        r = make_row(messages, tools)
        if r is None:
            return
        key = json.dumps(r["messages"], ensure_ascii=False)
        if key in seen:
            return
        seen.add(key)
        rows.append(r)

    guard = 0
    while len(rows) < n and guard < n * 40:
        guard += 1
        kind = rng.choices(["wx_single", "wx_multi", "stk_single", "stk_multi",
                            "search_single", "calc_single", "calc_multi",
                            "neg_tools", "neg_notools"],
                           weights=[14, 14, 10, 10, 10, 8, 8, 18, 8])[0]
        # each tool-use row exposes the relevant tool plus 0-2 distractor tools
        def tools_for(primary):
            extra = [t for t in all_tools if t["name"] != primary["name"]]
            rng.shuffle(extra)
            chosen = [primary] + extra[:rng.randint(0, 2)]
            rng.shuffle(chosen)
            return chosen

        if kind == "wx_single":
            c = rng.choice(KO_CITIES)
            args = {"city": c}
            if rng.random() < 0.3:
                args["unit"] = "celsius"
            push([{"role": "user", "content": rng.choice(KO_WEATHER_Q).format(c=c)},
                  mk_toolcall_msg("get_weather", args)], tools_for(KO_TOOLS["get_weather"]))
        elif kind == "wx_multi":
            c = rng.choice(KO_CITIES)
            temp = rng.randint(-8, 34)
            cond = rng.choice(["맑음", "흐림", "비", "눈", "구름 많음"])
            result = json.dumps({"city": c, "condition": cond, "temp_c": temp}, ensure_ascii=False)
            push([{"role": "user", "content": rng.choice(KO_WEATHER_Q).format(c=c)},
                  mk_toolcall_msg("get_weather", {"city": c}),
                  {"role": "tool", "content": result},
                  {"role": "assistant",
                   "content": f"현재 {c}의 날씨는 {cond}이고 기온은 {temp}도입니다."}],
                 tools_for(KO_TOOLS["get_weather"]))
        elif kind == "stk_single":
            t, mkt = rng.choice(KO_TICKERS)
            push([{"role": "user", "content": rng.choice(KO_STOCK_Q).format(t=t)},
                  mk_toolcall_msg("get_stock_price", {"ticker": t, "market": mkt})],
                 tools_for(KO_TOOLS["get_stock_price"]))
        elif kind == "stk_multi":
            t, mkt = rng.choice(KO_TICKERS)
            price = rng.randint(20, 900) * 100
            cur = "원" if mkt == "KOSPI" else "달러"
            result = json.dumps({"ticker": t, "price": price, "currency": cur}, ensure_ascii=False)
            push([{"role": "user", "content": rng.choice(KO_STOCK_Q).format(t=t)},
                  mk_toolcall_msg("get_stock_price", {"ticker": t, "market": mkt}),
                  {"role": "tool", "content": result},
                  {"role": "assistant",
                   "content": f"{t}의 현재 주가는 {price:,}{cur}입니다."}],
                 tools_for(KO_TOOLS["get_stock_price"]))
        elif kind == "search_single":
            q = rng.choice(KO_SEARCH)
            args = {"query": q}
            if rng.random() < 0.3:
                args["top_k"] = rng.choice([3, 5])
            push([{"role": "user", "content": rng.choice(KO_SEARCH_Q).format(q=q)},
                  mk_toolcall_msg("web_search", args)], tools_for(KO_TOOLS["web_search"]))
        elif kind == "calc_single":
            phrase, expr = rng.choice(KO_EXPR)
            push([{"role": "user", "content": f"{phrase} 계산해줘."},
                  mk_toolcall_msg("calculate", {"expression": expr})],
                 tools_for(KO_TOOLS["calculate"]))
        elif kind == "calc_multi":
            phrase, expr = rng.choice(KO_EXPR)
            try:
                val = eval(expr, {"__builtins__": {}})  # noqa: S307 - fixed safe exprs
            except Exception:
                continue
            result = json.dumps({"result": val}, ensure_ascii=False)
            push([{"role": "user", "content": f"{phrase}는 얼마야?"},
                  mk_toolcall_msg("calculate", {"expression": expr}),
                  {"role": "tool", "content": result},
                  {"role": "assistant", "content": f"{phrase}의 결과는 {val}입니다."}],
                 tools_for(KO_TOOLS["calculate"]))
        elif kind == "neg_tools":
            q, a = rng.choice(KO_NEG_QA)
            # tools available, but the query needs none -> answer directly
            primary = rng.choice(all_tools)
            push([{"role": "user", "content": q},
                  {"role": "assistant", "content": a}], tools_for(primary))
        else:  # neg_notools
            q, a = rng.choice(KO_NEG_QA)
            push([{"role": "user", "content": q},
                  {"role": "assistant", "content": a}], [])
    return rows


# ------------------------------------------------------------------ long-context

LONG_TOOLS = {
    "lookup_account_balance": {
        "name": "lookup_account_balance", "description": "Look up the current balance for an account id.",
        "parameters": {"type": "object", "properties": {
            "account_id": {"type": "string", "description": "The account identifier, e.g. ACC-48213"}},
            "required": ["account_id"]}},
    "get_ticket_status": {
        "name": "get_ticket_status", "description": "Fetch the status of a support ticket by id.",
        "parameters": {"type": "object", "properties": {
            "ticket_id": {"type": "string", "description": "Ticket id, e.g. TCK-90312"}},
            "required": ["ticket_id"]}},
    "get_order_details": {
        "name": "get_order_details", "description": "Retrieve details for an order number.",
        "parameters": {"type": "object", "properties": {
            "order_no": {"type": "string", "description": "Order number, e.g. ORD-771204"}},
            "required": ["order_no"]}},
    "convert_currency": {
        "name": "convert_currency", "description": "Convert an amount between two currencies.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from_currency": {"type": "string"},
            "to_currency": {"type": "string"}}, "required": ["amount", "from_currency", "to_currency"]}},
}

_FILLER_SENTENCES = [
    "The quarterly review meeting covered the migration timeline and the outstanding blockers.",
    "Engineering reported that the staging cluster was stable throughout the sprint.",
    "Marketing requested additional lead time for the upcoming launch campaign.",
    "The finance team reconciled the ledgers and flagged two pending invoices for follow-up.",
    "Operations confirmed that the warehouse restock was completed ahead of schedule.",
    "A retrospective note reminded the team to document deployment steps more carefully.",
    "Customer success shared that satisfaction scores improved after the latest release.",
    "The security audit found no critical issues, only a handful of low-severity items.",
    "Product management prioritized the backlog and moved three items into the next milestone.",
    "The data pipeline processed the nightly batch without errors for the fifth day running.",
    "Support volume was steady, with most tickets resolved within the target window.",
    "Legal reviewed the updated terms and requested a minor clarification in section four.",
    "The design team iterated on the onboarding flow based on recent usability sessions.",
    "Infrastructure costs stayed within budget after the reserved-instance changes.",
    "The release notes were drafted and are awaiting a final editorial pass.",
]


def _filler_block(rng: random.Random, n_sent: int) -> str:
    return " ".join(rng.choice(_FILLER_SENTENCES) for _ in range(n_sent))


def _long_task(rng: random.Random):
    """Return (needle_label, needle_value, tool_name, args, tool_result, final_answer_fn)."""
    kind = rng.choice(["account", "ticket", "order", "currency"])
    if kind == "account":
        val = f"ACC-{rng.randint(10000, 99999)}"
        bal = rng.randint(1000, 900000)
        return ("account id", val, "lookup_account_balance", {"account_id": val},
                json.dumps({"account_id": val, "balance_usd": bal}),
                f"The balance for account {val} is ${bal:,}.")
    if kind == "ticket":
        val = f"TCK-{rng.randint(10000, 99999)}"
        st = rng.choice(["open", "in_progress", "resolved", "escalated"])
        return ("ticket id", val, "get_ticket_status", {"ticket_id": val},
                json.dumps({"ticket_id": val, "status": st}),
                f"Support ticket {val} is currently '{st}'.")
    if kind == "order":
        val = f"ORD-{rng.randint(100000, 999999)}"
        st = rng.choice(["shipped", "processing", "delivered", "cancelled"])
        return ("order number", val, "get_order_details", {"order_no": val},
                json.dumps({"order_no": val, "status": st}),
                f"Order {val} is {st}.")
    amt = rng.randint(50, 9000)
    return (f"amount of {amt} EUR", str(amt), "convert_currency",
            {"amount": amt, "from_currency": "EUR", "to_currency": "USD"},
            json.dumps({"amount": amt, "from": "EUR", "to": "USD", "converted": round(amt * 1.08, 2)}),
            f"{amt} EUR is about ${round(amt * 1.08, 2):,} USD.")


def _big_filler(rng: random.Random, approx_chars: int) -> str:
    out, n = [], 0
    while n < approx_chars:
        blk = _filler_block(rng, rng.randint(18, 30)) + "\n"
        out.append(blk)
        n += len(blk)
    return "".join(out)


def gen_long(n: int, rng: random.Random, measure) -> list:
    """Long rows: a bulky document with a needle, then a query that needs a tool
    call using the needle -> forces attention across the long context.

    Length control is calibration-based (one estimate + at most one correction)
    to keep the number of expensive tokenizer passes ~2-3 per row.
    """
    # calibrate chars->tokens on the filler prose (English ~= 3.8 chars/token)
    cal = _big_filler(rng, 40000)
    cpt = len(cal) / max(1, measure({"messages": [{"role": "user", "content": cal}], "tools": "[]"}))

    rows, guard = [], 0
    while len(rows) < n and guard < n * 6:
        guard += 1
        label, needle, tool, args, tool_result, final = _long_task(rng)
        target = rng.randint(8200, 31500)
        tools = [LONG_TOOLS[tool]]
        header = ("You are reviewing an internal operations log. Read the log below; "
                  "a specific identifier is recorded in it that you will need.\n\n=== OPERATIONS LOG ===\n")
        needle_sentence = (f"Note: the {label} on record for this case is {needle}; "
                           f"please use it for any lookup.\n")
        query = ("\n=== END LOG ===\n\nBased on the operations log above, look up the record for the "
                 f"{label} mentioned in the log and tell me the result.")

        def assemble(filler_chars: int) -> dict:
            body = _big_filler(rng, filler_chars)
            # place the needle at a random position inside the filler
            cut = rng.randint(len(body) // 4, max(len(body) // 4, 3 * len(body) // 4))
            doc = header + body[:cut] + needle_sentence + body[cut:] + query
            if rng.random() < 0.6:
                msgs = [{"role": "user", "content": doc},
                        mk_toolcall_msg(tool, args),
                        {"role": "tool", "content": tool_result},
                        {"role": "assistant", "content": final}]
            else:
                msgs = [{"role": "user", "content": doc}, mk_toolcall_msg(tool, args)]
            return make_row(msgs, tools)

        est_chars = int(target * cpt)
        r = assemble(est_chars)
        if r is None:
            continue
        n_tok = measure(r)
        if not (8000 <= n_tok <= 32768):
            # one calibrated correction
            correction = int((target - n_tok) * cpt)
            r2 = assemble(max(2000, est_chars + correction))
            if r2 is not None:
                n2 = measure(r2)
                if 8000 <= n2 <= 32768:
                    r, n_tok = r2, n2
        if 8000 <= n_tok <= 32768:
            rows.append(r)
    return rows


# ------------------------------------------------------------------ length measurement

def build_measure(tokenizer):
    """Return a fn(row)->token length using the HF chat template on an
    OpenAI-projection of the native row (byte-identical to the swift render)."""
    def native_to_openai(messages: list):
        out = []
        for m in messages:
            if m["role"] == "tool_call":
                obj = json.loads(m["content"])
                out.append({"role": "assistant", "content": "",
                            "tool_calls": [{"type": "function",
                                            "function": {"name": obj["name"], "arguments": obj["arguments"]}}]})
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        # merge consecutive assistant empty+toolcall handled by template; fine as-is
        return out

    def measure(row: dict) -> int:
        tools = json.loads(row["tools"]) or None
        oai = native_to_openai(row["messages"])
        text = tokenizer.apply_chat_template(oai, tools=tools, tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return measure, native_to_openai


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short-target", type=int, default=15000)
    ap.add_argument("--long-target", type=int, default=3000)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--smoke", action="store_true", help="tiny run for a quick end-to-end check")
    args = ap.parse_args()
    if args.smoke:
        args.short_target, args.long_target = 400, 40

    rng = random.Random(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    print("loading tokenizer:", MERGED)
    tok = AutoTokenizer.from_pretrained(str(MERGED), trust_remote_code=True)
    measure, native_to_openai = build_measure(tok)

    # -- category quotas for the short file --
    st = args.short_target
    quota = {
        "glaive_single": int(st * 0.26),
        "glaive_multi": int(st * 0.26),
        "glaive_neg_tools": int(st * 0.16),
        "glaive_neg_notools": int(st * 0.06),
        "hermes": int(st * 0.18),
        "korean": max(int(st * 0.08), 200 if not args.smoke else 40),
    }
    print("short-file quotas:", quota)
    buckets = {k: [] for k in quota}
    seen_keys = set()

    def add(bucket, messages, tools):
        if len(buckets[bucket]) >= quota[bucket]:
            return False
        r = make_row(messages, tools)
        if r is None:
            return False
        key = json.dumps(r["messages"], ensure_ascii=False)
        if key in seen_keys:
            return False
        seen_keys.add(key)
        buckets[bucket].append(r)
        return True

    # ---- glaive ----
    print("downloading glaive ...")
    gpath = Path(hf_hub_download(GLAIVE_REPO, GLAIVE_FILE, repo_type="dataset", revision=GLAIVE_REV))
    need_glaive = ["glaive_single", "glaive_multi", "glaive_neg_tools", "glaive_neg_notools"]
    g_seen = 0
    for item in iter_glaive(gpath):
        if all(len(buckets[b]) >= quota[b] for b in need_glaive):
            break
        g_seen += 1
        tools = parse_glaive_system(item.get("system", ""))
        msgs = parse_glaive_chat(item.get("chat", ""))
        if not msgs:
            continue
        has_tc = any(m["role"] == "tool_call" for m in msgs)
        if tools and has_tc:
            # single-turn: first user + first tool_call
            first_u = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
            first_tc = next((i for i, m in enumerate(msgs) if m["role"] == "tool_call"), None)
            if first_u is not None and first_tc is not None and first_tc > first_u:
                add("glaive_single", msgs[first_u:first_tc + 1], tools)
            # multi-turn: full conv if it has a tool result followed by a final answer
            if any(m["role"] == "tool" for m in msgs) and msgs[-1]["role"] == "assistant":
                add("glaive_multi", msgs, tools)
        elif tools and not has_tc:
            add("glaive_neg_tools", msgs, tools)
        elif not tools:
            add("glaive_neg_notools", msgs, [])
    print(f"glaive: scanned {g_seen} convs -> "
          + ", ".join(f"{b}={len(buckets[b])}" for b in need_glaive))

    # ---- hermes ----
    print("downloading hermes ...")
    for hf in HERMES_FILES:
        if len(buckets["hermes"]) >= quota["hermes"]:
            break
        hpath = Path(hf_hub_download(HERMES_REPO, hf, repo_type="dataset", revision=HERMES_REV))
        data = json.loads(hpath.read_text())
        rng.shuffle(data)
        for item in data:
            if len(buckets["hermes"]) >= quota["hermes"]:
                break
            if not (item.get("conversations") or item.get("messages")):
                continue
            parsed = parse_hermes_conv(item)
            if parsed is None:
                continue
            msgs, tools = parsed
            add("hermes", msgs, tools)
    print(f"hermes: {len(buckets['hermes'])}")

    # ---- korean ----
    print("generating korean ...")
    for r in gen_korean(quota["korean"], rng):
        key = json.dumps(r["messages"], ensure_ascii=False)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        buckets["korean"].append(r)
    print(f"korean: {len(buckets['korean'])}")

    short_rows = [r for b in buckets.values() for r in b]
    rng.shuffle(short_rows)

    # ---- long ----
    print("generating long-context rows (measuring token lengths) ...")
    long_rows = gen_long(args.long_target, rng, measure)
    print(f"long: {len(long_rows)}")

    # ---- write ----
    short_path = out_dir / "tooluse_short.jsonl"
    long_path = out_dir / "tooluse_long32k.jsonl"
    _write(short_path, short_rows)
    _write(long_path, long_rows)

    # ---- stats ----
    print("\n=== RESULT ===")
    print(f"{short_path}: {len(short_rows)} rows")
    for b, rs in buckets.items():
        print(f"    {b}: {len(rs)}")
    lens = sorted(measure(r) for r in long_rows)
    if lens:
        import statistics
        print(f"{long_path}: {len(long_rows)} rows | tokens "
              f"min={lens[0]} median={int(statistics.median(lens))} max={lens[-1]}")
        dist = {"8-12k": 0, "12-16k": 0, "16-24k": 0, "24-32k": 0}
        for L in lens:
            if L < 12000: dist["8-12k"] += 1
            elif L < 16000: dist["12-16k"] += 1
            elif L < 24000: dist["16-24k"] += 1
            else: dist["24-32k"] += 1
        print("    long token distribution:", dist)
    return 0


def _write(path: Path, rows: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
