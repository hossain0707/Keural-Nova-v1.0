# Keural Nova v1.1 — Tool-Calling QA Checklist (for dev.keural.ai)

A hands-on checklist to verify that tool-calling works in the live product. Run each prompt in a
**New chat**, note the result, and use the pass/fail criteria.

## 0. Before you start (preconditions)
- **Model selector = `Keural Nova v1.1`** (not v1.0 — v1.1 is the strong tool-caller).
- **Server updated:** the HF cache was **cleared and the model re-pulled** so the non-thinking
  template fix is live. (If you skipped this, tests will loop — see the failure signature below.)
- The features that use tools are enabled (web search / DEEP MAX, file/RAG via AI Drive).

## 1. How to read a result

| | Looks like |
|---|---|
| ✅ **Working** | A tool step appears (e.g. "웹에서 검색하는 중" / "Searching…" / "Reading file…"), it **completes**, and the **final answer uses the retrieved info** (fresh data / file content). |
| ❌ **Broken (the old bug)** | It **repeats** "웹 에이전트가 응답하지 않습니다 / the web agent isn't responding, I'll check the URL directly" **over and over** → looping = the tool call isn't being emitted/parsed. |

---

## A. Single tool call — web search / current info
Type these; each should trigger **one** search and answer using it.
1. `오늘 서울 날씨 어때?`
2. `What's the latest news about Apple today?`
3. `현재 비트코인 시세 알려줘.`

**PASS:** a search step runs and completes; the answer contains fresh/real info; **no loop.**

## B. Deep research — multi-step (DEEP MAX mode)
These need **several** searches, then a synthesized answer.
4. `삼성전자와 애플의 최신 시가총액을 비교해서 정리해줘.`
5. `Research the top 3 electric vehicles of 2026 and compare their driving range.`

**PASS:** multiple tool steps run, and the **final answer synthesizes** the results into a
coherent comparison; **no loop**, and it doesn't stop after saying "I will search" without actually
producing the answer.

## C. RAG — file / document (AI Drive or upload with "+")
6. Upload a PDF/DOCX, then: `이 문서의 핵심 내용을 3가지로 요약해줘.`
7. Upload a file and ask a question **only answerable from that file** (e.g. a specific number/name).

**PASS:** the answer reflects the **actual file content** (not a generic guess); for #7 it states
the specific fact from the file.

## D. Negative test — must NOT over-call a tool
A correct model also **doesn't** call tools when it shouldn't.
8. `안녕, 간단히 자기소개 해줘.`
9. `2 더하기 2는 얼마야?`

**PASS:** answers **directly**, without triggering a web search. (If it web-searches "2+2", that's
an over-calling problem — note it.)

## E. Identity (bonus, confirms the right model is live)
10. `너 누구야? 누가 만들었어?`

**PASS:** answers it's **Keural / MKD** — should **not** say "Gemma", "Qwen", "Google", or "OpenAI".

---

## Scoring
- Run **A–D** (≈9 prompts). If tools **trigger, complete, and the answers use the results**, and
  the negative tests (D) don't over-call → **tool-calling is working.**
- **Any** prompt that loops with the "web agent not responding" message repeating → the fix is
  **not live** on the server. Go to triage.

## If something fails — quick triage
| Symptom | Likely cause | Check |
|---|---|---|
| Loops / repeats "web agent not responding" | Still running **thinking mode** (old template) | Re-pull model (cache cleared); confirm `chat_template.jinja` has `enable_thinking is not defined or ... is false` |
| Says "I'll search…" then stops, no tool step | Tool call not parsed | Serve flags: `--tool-call-parser qwen3_xml --enable-auto-tool-choice` |
| Tool never triggers at all | Tools not registered / auto-tool-choice off | Backend tool registration + `--enable-auto-tool-choice` |
| Tool runs but answer ignores the result | Multi-turn wiring | Backend must feed the tool **result** back as a `tool` message before the final turn |

## Expected reference (from offline eval, non-thinking)
- Single tool-call emission: **v1.1 ≈ 91%** (≈ base Qwen). Occasionally answers directly instead
  of calling — not every borderline case must call a tool.
- Multi-turn (call → result → answer): **worked 6/6** with no looping.
- So in the UI you should see **most** tool-needing prompts trigger + answer correctly; a rare
  direct-answer on a borderline case is normal, **but repeated looping is not** and means the
  server fix isn't live.
