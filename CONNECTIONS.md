# Подключения: что нужно завести снаружи проекта

Всё, что требует регистрации, ключа или внешнего сервиса, собрано здесь.
Код проекта ничего из этого **не требует для запуска**: без ключей он работает
на детерминированном mock-режиме (`LLM_*` не заданы → `mock`). Подключения
нужны только для прогонов с реальными моделями и для публикации.

Ключи кладутся в `.env` (в `.gitignore`), шаблон — `.env.example`.

## 1. Нужно сейчас (Этап B)

Нужен **хотя бы один** LLM-провайдер. Ранжировано по «бесплатно и без карты»:

| # | Провайдер | Бесплатно? | Что получить | Куда положить | Модель в `.env` |
|---|---|---|---|---|---|
| 1 | **Google AI Studio (Gemini API)** | Да, free tier без карты (лимиты RPM/RPD — см. страницу лимитов) | API key: https://aistudio.google.com/apikey | `GEMINI_API_KEY` | `gemini/gemini-2.5-flash` |
| 2 | **Groq** | Да, free tier (лимиты в консоли) | API key: https://console.groq.com/keys | `LLM_API_KEY` + `LLM_BASE_URL=https://api.groq.com/openai/v1` | `hosted_vllm/llama-3.3-70b-versatile` |
| 3 | **OpenRouter** | Ключ бесплатно; есть модели с суффиксом `:free` | API key: https://openrouter.ai/keys | `OPENROUTER_API_KEY` | `openrouter/<org>/<model>:free` |
| 4 | **Cloudflare Workers AI** | 10 000 нейронов/день бесплатно (аккаунт уже есть — advien.tech) | Account ID + API token с правом Workers AI: https://dash.cloudflare.com → AI → Workers AI | `LLM_API_KEY=<token>` + `LLM_BASE_URL=https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1` | `hosted_vllm/@cf/meta/llama-3.1-8b-instruct` |
| 5 | **Ollama** (локально) | Да; нужна RAM ≈ 6 ГБ на 8B-модель | https://ollama.com/download, затем `ollama pull llama3.1:8b` | ничего (по умолчанию `localhost:11434`) | `ollama/llama3.1:8b` |
| 6 | **Anthropic Console** | Нет — предоплата, от $5 | https://console.anthropic.com → API keys | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-5`, `anthropic/claude-haiku-4-5`, `anthropic/claude-opus-5` |
| 7 | **OpenAI Platform** | Нет — предоплата | https://platform.openai.com/api-keys | `OPENAI_API_KEY` | `openai/gpt-5-mini` и т.п. |

Рекомендация:
- **разработка и отладка** — №1 (Gemini) или №2 (Groq): бесплатно, быстро;
- **финальный прогон для скриншотов/демо и эвалов** — №6 (Anthropic) с моделью
  на роль (Haiku → Sonnet → Opus на Safety-гейте): ~$0.05–0.20 за прогон кейса;
- №4 (Workers AI) — как «своя модель на своём домене» для одной дешёвой роли
  (Documentation): демонстрирует гетерогенность провайдеров.

Модель задаётся **на роль**: `LLM_INTAKE`, `LLM_BIOMECH`, `LLM_RECOMMENDATION`,
`LLM_SAFETY`, `LLM_DOCUMENTATION`; `LLM_DEFAULT` — для всех остальных.
К любой роли можно добавить `LLM_<ROLE>_BASE_URL` / `LLM_<ROLE>_API_KEY`.

Проверка: `python -m prosthetics_agents.stage_b.run --case case_01_transtibial_k3 --show-llm`
— в шапке видно, какая модель на какой роли, в таблице — вызовы и токены.

## 2. Для публикации (когда появится что показывать)

| Что | Бесплатно? | Зачем | Где |
|---|---|---|---|
| GitHub-репозиторий | Да | Код + README для портфолио | https://github.com/new |
| Streamlit Community Cloud | Да (публичные приложения) | Хостинг интерактивного демо/дашборда | https://share.streamlit.io |
| advien.tech (Cloudflare) | Уже есть | Ссылка на демо и схемы в разделе Portfolio | — |

## 3. Понадобится на следующих этапах — ничего заводить пока не нужно

| Этап | Что | Подключение? |
|---|---|---|
| B+ (эвалы) | eval-harness на golden-кейсах, сравнение crew vs одиночный промпт | Только LLM-ключ из п.1 |
| B+ (визуализация) | HTML-таймлайн/граф из `runs/<id>/events.jsonl` | Нет — статический файл |
| C (LangGraph) | `trimesh`, `CadQuery`/`build123d` | Нет — локальные библиотеки, pip |
| C (трейсинг, опц.) | LangSmith | Есть free tier: https://smith.langchain.com — необязательно, у нас свой JSONL-лог |
| A (AutoGen) | свой FastAPI-сервис с ML-классификатором намерения | Нет локально; для хостинга опц. Hugging Face Spaces (free) |

## 4. Что специально НЕ используется

Без базы данных, без векторного хранилища, без платной observability, без
CrewAI AMP/tracing (отключено через `CREWAI_TRACING_ENABLED=false`) — всё
состояние живёт в `runs/<run_id>/` как JSON.
