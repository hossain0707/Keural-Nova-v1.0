# Tool-calling confirmation — Qwen3.6-35B-A3B (base)  vs  Keural Nova v1.1  (non-thinking mode)


**Pass rate (correct tool call, 12 cases):** Qwen3.6-35B-A3B (base) = 11/12  ·  Keural Nova v1.1 = 11/12


| # | Query | Qwen3.6-35B-A3B (base) | Keural Nova v1.1 | Match |
|---|---|---|---|:---:|
| 1 | 서울 날씨 어때? | `get_weather` | `get_weather` | ✅ |
| 2 | What's the weather in Tokyo right now? | `get_weather` | `get_weather` | ✅ |
| 3 | 비트코인 현재 가격 좀 알려줘. | `get_stock_price` | `get_stock_price` | ✅ |
| 4 | What is the current price of TSLA? | `get_stock_price` | `get_stock_price` | ✅ |
| 5 | 최신 인공지능 뉴스 좀 검색해줘. | `web_search` | `web_search` | ✅ |
| 6 | Search the web for the latest iPhone release. | `web_search` | `web_search` | ✅ |
| 7 | 1234 곱하기 5678은 얼마야? | `calculator` | `— no tool call` | ⚠️ |
| 8 | Compute 987 * 654 for me. | `— no tool call` | `calculator` | ⚠️ |
| 9 | 지금 뉴욕 시간 몇 시야? | `get_current_time` | `get_current_time` | ✅ |
| 10 | What time is it in London right now? | `get_current_time` | `get_current_time` | ✅ |
| 11 | 부산 날씨 알려줘. | `get_weather` | `get_weather` | ✅ |
| 12 | 삼성전자 주가 알려줘. | `get_stock_price` | `get_stock_price` | ✅ |