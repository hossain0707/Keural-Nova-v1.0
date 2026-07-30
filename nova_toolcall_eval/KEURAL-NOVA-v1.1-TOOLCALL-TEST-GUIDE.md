# Keural Nova v1.1 — Tool-Calling Test Guide (dev.keural.ai)

A step-by-step guide for verifying that **tool-calling** (web search, deep research, file/RAG)
works correctly in Keural Nova v1.1 on **dev.keural.ai**. Anyone on the team can run this — just
paste the prompts and record whether each behaves as described.

---

## 0. Setup (do this first)

1. Open **dev.keural.ai** → click **New chat**.
2. In the model dropdown, select **`Keural Nova v1.1`** (⚠️ **v1.1**, not v1.0).
3. Enable the tool features you want to test (web search / **DEEP MAX** for research; **AI Drive**
   / file upload for RAG).
4. If tests fail with a repeating loop, the server likely needs the model **cache cleared + a
   re-pull** — see [Troubleshooting](#troubleshooting).

> Run each prompt in a **fresh New chat** unless a section says otherwise (e.g. follow-up tests).

---

## 1. How to judge each result

| Outcome | What you'll see |
|---|---|
| ✅ **PASS** | A tool step appears ("웹에서 검색하는 중…" / "Searching…" / "Reading file…"), it **completes**, and the final answer **uses the retrieved info** (real data / file content). |
| ⚠️ **PARTIAL** | It answers reasonably but **skips the tool** on a case that clearly needed one (e.g. asks for "today's" info but answers from memory). |
| ❌ **FAIL (the bug)** | It **repeats** "웹 에이전트가 응답하지 않습니다 / the web agent isn't responding…" over and over, or says "I'll search…" then stops with no answer. |

---

## 2. Test cases

### A · Web search — real-time / factual (single tool call)
Each should trigger **one** search and answer using it.

1. `오늘 서울 날씨 어때?`
2. `부산 지금 날씨랑 기온 알려줘.`
3. `What's the weather in Tokyo right now?`
4. `현재 비트코인 가격 알려줘.`
5. `테슬라(TSLA) 주가 지금 얼마야?`
6. `What is the latest news about Apple today?`
7. `오늘 원/달러 환율 알려줘.`
8. `이번 주 개봉한 한국 영화 뭐 있어?`

**PASS:** search runs → answer contains fresh/real info → no loop.

### B · Deep research — multi-step (use DEEP MAX)
Each needs **several** searches, then a synthesized answer.

9. `삼성전자와 애플의 최신 시가총액을 비교해서 표로 정리해줘.`
10. `2026년 전기차 인기 모델 3가지를 조사해서 주행거리를 비교해줘.`
11. `Research the top 3 AI chatbots in 2026 and summarize their strengths and weaknesses.`
12. `최근 한국 부동산 시장 동향을 3가지 핵심 포인트로 정리해줘.`
13. `Compare the latest iPhone and the latest Galaxy phone — camera, battery, price.`

**PASS:** multiple tool steps run → **final answer synthesizes** the findings into a clear
comparison/summary → no loop, and it doesn't stop after only saying "I will search."

### C · RAG — file / document (upload a file with "+" or via AI Drive, then ask)
Upload a PDF/DOCX/TXT first, then in the same chat:

14. `이 문서의 핵심 내용을 3가지로 요약해줘.`
15. `이 파일에서 가장 중요한 숫자(또는 결론)가 뭐야?`
16. `이 문서에 [특정 주제]에 대한 내용이 있어? 있으면 관련 부분을 알려줘.`
17. `Summarize this document in 5 bullet points in English.`
18. (Upload 2 files) `두 문서의 차이점을 비교해줘.`

**PASS:** the answer reflects the **actual file content** — specific facts/numbers from the file,
not a generic guess.

### D · Calculation / utility tools (if enabled)
19. `1234 곱하기 5678은 얼마야?`
20. `현재 뉴욕 시간 몇 시야?`
21. `What time is it in London right now?`

**PASS:** uses the right tool (calculator/time) or gives a correct direct answer — no loop.

### E · Multi-turn follow-up (tool result → follow-up question)
Do these as a **continuing conversation** (same chat):

22. First: `오늘 서울 날씨 어때?` → wait for the answer. Then follow up: `그럼 우산 가져가야 할까?`
23. First: `테슬라 주가 알려줘.` → then: `그럼 애플 주가는?`
24. First: `최신 AI 뉴스 검색해줘.` → then: `방금 첫 번째 뉴스 더 자세히 설명해줘.`

**PASS:** the follow-up **uses the earlier tool result / context** and answers sensibly (calls a
new tool if needed) — no loop, no "forgetting" the previous result.

### F · Negative tests — must NOT call a tool
A correct model also **avoids** tools when they're unnecessary.

25. `2 더하기 2는 얼마야?`
26. `안녕! 오늘 기분이 좋아.`
27. `한국의 수도는 어디야?`  *(general knowledge — no live search needed)*
28. `짧은 시 하나 써줘.`

**PASS:** answers **directly**, without triggering a web search. (Searching for "2+2" or "capital
of Korea" = over-calling; note it as ⚠️.)

### G · Identity (confirms the correct model is live)
29. `너 누구야? 누가 만들었어?`
30. `너 Gemma야? 아니면 GPT야?`

**PASS:** answers it's **Keural / MKD**; must **not** claim to be Gemma, Qwen, Google, or OpenAI.

---

## 3. Results tracking (copy this table and fill it in)

| # | Category | Prompt (short) | Tool triggered? | Answer used result? | Verdict (✅/⚠️/❌) | Notes |
|---|---|---|:---:|:---:|:---:|---|
| 1 | Web | 오늘 서울 날씨 | | | | |
| 6 | Web | latest Apple news | | | | |
| 9 | Deep research | 삼성 vs 애플 시총 | | | | |
| 11 | Deep research | top 3 AI chatbots | | | | |
| 14 | RAG | 문서 3줄 요약 | | | | |
| 15 | RAG | 파일 핵심 숫자 | | | | |
| 22 | Multi-turn | 날씨 → 우산? | | | | |
| 25 | Negative | 2+2 | | (should be N) | | |
| 29 | Identity | 너 누구야 | (N) | | | |

*(Add rows as needed. The full prompt list is in Section 2.)*

---

## 4. Overall verdict

- **PASS (tool-calling works):** Sections A & B mostly trigger tools and answer with real data;
  C reflects file content; E follow-ups keep context; and F does **not** over-call. Occasional
  ⚠️ on a borderline single case is acceptable.
- **FAIL:** **any** prompt loops with "web agent not responding" repeating, or search prompts
  consistently answer from memory / stop without answering.

**Reference (from offline testing, non-thinking mode):** single tool-call accuracy **≈ 91%**
(on par with the base model); multi-turn (call → result → answer) **worked 6/6**. So in the UI you
should see most tool-needing prompts work; **repeated looping is the failure to watch for.**

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix / who to tell |
|---|---|---|
| Loops: "웹 에이전트가 응답하지 않습니다…" repeats | Server still using the **old (thinking-mode) template** | Ops: **clear HF cache + re-pull** Keural Nova v1.1 so the updated `chat_template.jinja` / `tokenizer_config.json` load |
| Says "I'll search…" then stops, no tool ran | Tool call not parsed | Ops: serve with `--tool-call-parser qwen3_xml --enable-auto-tool-choice` |
| Tool never triggers for anything | Tools not registered / auto-tool-choice off | Ops: check tool registration + `--enable-auto-tool-choice` |
| Tool runs but answer ignores the result | Multi-turn wiring | Ops: backend must feed the tool **result** back (as a `tool` message) before the final answer |
| Model says it's "Gemma"/"Qwen"/"GPT" | Wrong model or old cache | Confirm **Keural Nova v1.1** selected + re-pulled |

**Model to deploy for tool use:** **Keural Nova v1.1** (v1.0 tool-calls but is weaker, ~67%).
