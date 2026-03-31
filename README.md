# LLM API Mini Project (Claude)

Мини-проект для демонстрации первого запроса к LLM через API:
- CLI режим (`python3 cli.py`)
- Web режим (`python3 web.py`)
- Telegram-бот на локальной LLM (`python3 telegram_local_bot.py`)
- MCP demo (`python3 mcp_list_tools.py`)

## 1) Setup

```bash
cd /Users/useruserowicz/work/llm-api-mini-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Откройте `.env` и вставьте ваш ключ:

```env
ANTHROPIC_API_KEY=your_new_api_key_here
ANTHROPIC_MODEL=
LOCAL_LLM_ENDPOINT=http://127.0.0.1:8088
LOCAL_LLM_MODEL=qwen-local
TELEGRAM_BOT_TOKEN=
```

`ANTHROPIC_MODEL` optional: если оставить пустым, проект сам подберет доступную модель.
`TELEGRAM_BOT_TOKEN` нужен только для Telegram-бота.

## Telegram бот с локальной LLM (без облака)

Этот режим не использует Anthropic/OpenAI. Бот отправляет сообщения только в локальный endpoint:
- `POST {LOCAL_LLM_ENDPOINT}/v1/chat/completions`

### Шаги

1. Поднимите локальную LLM, совместимую с OpenAI Chat Completions API.
2. Укажите в `.env`:
   - `LOCAL_LLM_ENDPOINT` (например `http://127.0.0.1:8088`)
   - `LOCAL_LLM_MODEL` (например `qwen-local`)
   - `TELEGRAM_BOT_TOKEN` (токен от `@BotFather`)
3. Запустите:

```bash
python3 telegram_local_bot.py
```

Команды в Telegram:
- `/start` — справка
- `/health` — пробный запрос к локальной модели
- `/help <вопрос>` — ассистент разработчика: ответы по документации (RAG по README, `docs/` в т.ч. `docs/API.md`, `requirements.txt`, `.env.example`) и контексту репозитория через **MCP** (`get_current_git_branch`, список файлов, `git diff --stat`; при сбое MCP — тот же контекст через локальный `git`)
- `/review_pr <id|url>` — AI-ревью PR (получает diff и changed files, использует RAG по docs + коду, возвращает баги/архитектуру/рекомендации)

Перед первым запуском `/help` соберите индекс (или он создаётся при первом обращении, но дольше):

```bash
python3 build_dev_assistant_index.py
```

Те же сведения о репозитории доступны через MCP в `mcp_local_server.py`: `get_current_git_branch`, `list_project_tracked_files`, `get_working_tree_diff_stat`.

В веб-интерфейсе (`python3 web.py`, режим **«Автоматизация»** в левой колонке) откройте блок **«Ассистент разработчика»**: введите вопрос, при демонстрации задания можно включить галочку «контекст через MCP». Нужны поднятая локальная LLM и собранный индекс (`build_dev_assistant_index.py`).

### Пошаговая проверка (end-to-end)

1. Установите зависимости и активируйте окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Проверьте, что `.env` заполнен:

```env
LOCAL_LLM_ENDPOINT=http://127.0.0.1:8088
LOCAL_LLM_MODEL=qwen-local
TELEGRAM_BOT_TOKEN=ваш_токен_от_BotFather
```

3. Быстрый smoke-test локальной LLM:

```bash
python3 check_local_llm.py
```

Ожидаемо:
- в `[health]` есть статус `ok`,
- в `[chat]` приходит короткий ответ модели.

4. Запустите веб-приложение:

```bash
python3 web.py
```

Проверьте в UI:
- в `Quick settings` выбрано `LLM backend = Local (no cloud)`,
- в `Local LLM checks` нажмите `Run local LLM checks`,
- отправьте сообщение через `Send` и убедитесь, что пришел ответ от локальной модели.

5. Запустите Telegram-бота:

```bash
python3 telegram_local_bot.py
```

6. Проверка в Telegram:
- откройте бота, отправьте `/start`,
- отправьте `/health`,
- отправьте обычный вопрос (например: `Объясни REST API в 3 пунктах`),
- бот должен ответить текстом из локальной LLM.

7. Критерий "без облака":
- не нужен `ANTHROPIC_API_KEY`,
- бот и local-режим в web работают только через `LOCAL_LLM_ENDPOINT`.

## Приватный AI-сервис на локальной LLM (HTTP API)

Сервис: `private_ai_service.py`
- разворачивается на VPS или домашнем сервере
- дает HTTP API для чата: `POST /v1/chat`
- дает OpenAI-совместимый endpoint: `POST /v1/chat/completions` (удобно для web UI этого проекта)
- имеет базовые ограничения: API key, rate limit, max context

### 1) Настройка `.env`

Добавьте:

```env
PRIVATE_AI_HOST=0.0.0.0
PRIVATE_AI_PORT=8099
PRIVATE_AI_API_KEY=change-me-private-key
PRIVATE_AI_RATE_LIMIT_PER_MIN=30
PRIVATE_AI_MAX_CONTEXT_TOKENS=8192
PRIVATE_AI_DEFAULT_MAX_TOKENS=384
PRIVATE_AI_DEFAULT_TEMPERATURE=0.3
PRIVATE_AI_MAX_COMPLETION_TOKENS=1024
PRIVATE_AI_ALLOW_LOCALHOST_NO_AUTH=1
```

### 2) Запуск

Сначала локальная модель:

```bash
llama-server -m "/Users/useruserowicz/work/llm-api-mini-project/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf" -a qwen-local --host 127.0.0.1 --port 8088
```

Затем приватный сервис:

```bash
python3 private_ai_service.py
```

### 3) Проверка доступа по сети

На сервере:

```bash
curl -H "X-API-Key: change-me-private-key" http://127.0.0.1:8099/v1/config
```

С другой машины (замените `<server-ip>`):

```bash
curl -X POST "http://<server-ip>:8099/v1/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me-private-key" \
  -d '{"messages":[{"role":"user","content":"Дай 3 пункта о плюсах приватного AI-сервиса"}]}'
```

### 4) Проверка стабильности при нескольких запросах

```bash
python3 check_private_ai_service.py \
  --base-url "http://127.0.0.1:8099" \
  --api-key "change-me-private-key" \
  --requests 8 \
  --parallel 4
```

Смотрите в JSON:
- `success_rate`
- `avg_model_latency_ms`
- `failed`

### 5) Проверка ограничений

1. **Rate limit**: поставьте `PRIVATE_AI_RATE_LIMIT_PER_MIN=2` и отправьте 3+ запроса подряд -> получите `429`.
2. **Max context**: поставьте `PRIVATE_AI_MAX_CONTEXT_TOKENS=300` и отправьте длинный диалог -> контекст будет урезан/ограничен сервисом.

Это и есть минимально необходимый приватный AI-сервис на локальной LLM.

### Показ прямо из web UI (без curl)

1. Запустите `private_ai_service.py`.
2. Откройте `http://127.0.0.1:5051`.
3. В секции **Local LLM checks + optimization lab** укажите:
   - `Local endpoint`: `http://127.0.0.1:8099`
   - `Model name`: `qwen-local`
4. Нажмите `Run local LLM checks` и затем обычный `Send` в чате.

Почему это работает:
- web UI проекта отправляет запросы в OpenAI-формате на `/v1/chat/completions`
- приватный сервис теперь поддерживает этот endpoint.

## 2) Run CLI

```bash
python3 cli.py
```

## AI code review on PR (GitHub Actions)

Автоматическое ревью PR запускается workflow:
- `.github/workflows/ai-pr-review.yml`
- скрипт: `scripts/ai_pr_review.py`

Что делает пайплайн:
- берет diff и список измененных файлов (`base...head`);
- строит RAG-контекст по `README.md`, `docs/` и коду репозитория;
- публикует комментарий в PR с блоками:
  - потенциальные баги,
  - архитектурные проблемы,
  - рекомендации.

Настройка:
1. В GitHub репозитории добавьте secret: `ANTHROPIC_API_KEY`.
2. (Опционально) добавьте repo variable: `ANTHROPIC_MODEL`.
3. Откройте/обновите PR — комментарий AI-ревью появится автоматически.

### Наглядный запуск через Telegram-бота

1. Запустите бота:

```bash
python3 telegram_local_bot.py
```

2. В Telegram отправьте:
   - `/start`
   - `/review_pr 1`  
     или `/review_pr https://github.com/nixusUM/llm-api-mini-project/pull/1`

Бот вернет текст ревью по PR в формате:
- потенциальные баги,
- архитектурные проблемы,
- рекомендации.

### Автоматическое создание demo PR (одной командой)

Если нужно быстро показать полный цикл (изменение -> push -> PR -> `/review_pr`), запустите:

```bash
python3 scripts/create_demo_pr.py
```

Скрипт автоматически:
- создаст ветку `demo/tg-review-auto-<timestamp>`,
- добавит demo-строку в `docs/API.md`,
- сделает commit и push,
- откроет PR в `main` и выведет URL.

После этого в Telegram отправьте:

```text
/review_pr <номер_PR_из_URL>
```

## 3) Run Web

```bash
python3 web.py
```

Откройте: `http://127.0.0.1:5000`

## 4) Create GitHub repository

### Option A: through GitHub CLI

```bash
git init
git add .
git commit -m "Initial mini project: Claude API CLI + web demo"
gh repo create llm-api-mini-project --public --source=. --remote=origin --push
```

### Option B: through github.com manually

1. Создайте пустой репозиторий `llm-api-mini-project`.
2. Выполните:

```bash
git init
git add .
git commit -m "Initial mini project: Claude API CLI + web demo"
git branch -M main
git remote add origin https://github.com/<your-username>/llm-api-mini-project.git
git push -u origin main
```

## Security note

Вы уже отправили API-ключ в чат. Для безопасности отзовите старый ключ в Anthropic Console и создайте новый.

## MCP: minimal connection + tool listing

Локальный сценарий (без внешнего MCP):

```bash
python3 mcp_list_tools.py
```

Ожидаемый результат:
- `MCP connection: OK`
- список инструментов (например: `ping`, `sum_two_numbers`)

Подключение к другому MCP-серверу:

```bash
python3 mcp_list_tools.py <command> [args...]
```

Пример:

```bash
python3 mcp_list_tools.py python3 mcp_local_server.py
```

## First MCP tool (mock API)

В локальном MCP-сервере добавлен инструмент:
- `get_todo_from_mock_api(todo_id: int)` — получает todo из JSONPlaceholder.

Пример CLI-вызова инструмента через MCP-клиент:

```bash
python3 mcp_list_tools.py --call get_todo_from_mock_api '{"todo_id": 3}'
```

В web UI:
- кнопка `Run MCP tool: get_todo_from_mock_api`
- параметр `todo_id`
- результат подставляется в поле сообщения, чтобы агент мог его использовать.

## MCP scheduler and background summaries (24/7)

В локальном MCP-сервере добавлены инструменты планировщика:
- `configure_periodic_summary(job_id, interval_seconds, user_id, enabled)`
- `run_due_summaries()`
- `get_summary_report(limit)`

Что делает:
- сохраняет состояние задач и историю запусков в `data/mcp_periodic_state.json`
- выполняет задачи по расписанию
- возвращает агрегированную сводку (runs + average completion rate)

В web UI:
- `Configure periodic job`
- `Run periodic tick now`
- `Start 24/7 scheduler` / `Stop`

Фоновый планировщик в приложении периодически вызывает MCP-инструменты и обновляет сводку.

## MCP tool composition pipeline

Добавлены три MCP-инструмента для цепочки:
- `search_data(query, limit)` — получает данные (поиск по mock API)
- `summarize_data(search_payload_json)` — обрабатывает результат поиска
- `save_to_file(file_name, content)` — сохраняет summary в файл

Автоматический пайплайн в web UI:
- кнопка `Run MCP pipeline: search -> summarize -> save_to_file`
- результат каждого шага показывается во вкладке `Pipeline`

CLI примеры:

```bash
# 1) Search
python3 mcp_list_tools.py --call search_data '{"query":"qui","limit":3}'

# 2) Summarize (передайте JSON из search_data)
python3 mcp_list_tools.py --call summarize_data '{"search_payload_json":"{\"ok\":true,\"query\":\"qui\",\"returned_count\":1,\"items\":[{\"id\":2,\"title\":\"qui est esse\",\"body_excerpt\":\"...\"}]}"}'

# 3) Save
python3 mcp_list_tools.py --call save_to_file '{"file_name":"pipeline_summary.txt","content":"Your summary text"}'
```

## Orchestration MCP (multi-server)

Регистрируются несколько MCP-серверов (локальный + публичный `@modelcontextprotocol/server-everything`). Агент/оркестратор выбирает инструмент и маршрутизирует вызов на нужный сервер.

- **Модуль**: `mcp_orchestrator.py` — конфиг серверов `SERVERS`, `get_tool_to_server_map()`, `run_long_flow(query, limit, output_file)`.
- **Длинный флоу**: `search_data` (local) → `summarize_data` (local) → `echo` (public) → `save_to_file` (local). Порядок шагов и выбор сервера заданы в коде; результат каждого шага сохраняется.

В web UI:
- Вкладка **Orchestration**: зарегистрированные серверы, таблица «инструмент → сервер», результат последнего запуска флоу.
- В левой панели: поля запроса/лимита/имени файла и кнопка **Run orchestration flow (local + public MCP)**.

Для работы флоу с публичным сервером нужны `node`/`npx` (запуск `npx -y @modelcontextprotocol/server-everything`).

## Document Indexer Pipeline

Пайплайн индексации документов с chunking, эмбеддингами и метаданными.

### Компоненты

- **Chunking strategies**:
  - `FixedSizeChunker` — фиксированный размер с overlap
  - `StructureBasedChunker` — разбиение по заголовкам/разделам

- **Embedder** — генерация эмбеддингов через OpenAI API (или mock для тестирования)

- **Storage backends**:
  - `FAISSStorage` — векторный поиск (при наличии faiss)
  - `SQLiteStorage` — SQL-based с brute-force поиском
  - `JSONStorage` — простой JSON-файл

### Запуск

```bash
python3 document_indexer_app.py
```

Откройте: `http://127.0.0.1:5052/document_indexer`

### Использование

1. Выберите документы из смешанного корпуса:
   - `README.md`
   - статьи из `sample_documents/`
   - код (`*.py` в корне и `document_indexer/*.py`)
   - PDF-файлы из `sample_documents/` (если добавлены)
2. Выберите стратегию chunking и хранилище
3. Нажмите "Индексировать"
4. Перейдите на вкладку "⚖️ Сравнение стратегий" для сравнения
5. Используйте поиск для проверки ретривала

### Первый RAG-запрос (в этой же странице)

1. В блоке **Первый RAG-запрос** введите вопрос и задайте `Top-k` для извлечения релевантных чанков.
2. Нажмите **Сравнить режимы** — вы сразу увидите ответы модели без RAG и с подключёнными документами.
3. Обратите внимание на извлечённые чанки и источники: они лежат рядом в результате.
4. Используйте список из 10 контрольных вопросов под формой, чтобы проанализировать качество выдач.

### Day 23: reranking + фильтрация + query rewrite

В RAG-блоке добавлены настройки для второго этапа после векторного поиска:
- `Top-k до rerank/filter`
- `Top-k после filter`
- `Порог релевантности (threshold)`
- переключатель `Query rewrite`

Интерфейс показывает сравнение:
- **Baseline RAG**: без rewrite/фильтрации
- **Improved RAG**: с rewrite + rerank/filter

Для каждого режима видны:
- чанки до/после фильтрации,
- источники,
- итоговый ответ модели.

### Day 26: локальная LLM + локальный RAG (Week 6 index)

Теперь RAG можно прогонять полностью локально:
- retrieval работает по сохраненному индексу Week 6: `document_indices/index_fixed_size_json_20260316_195228.json`
- генерация ответа идет через локальную модель (`llama-server`, модель `qwen-local`)
- cloud-режим оставлен для сравнения качества и стабильности

Переключение backend генерации:

```bash
export RAG_GENERATION_BACKEND=local   # local | cloud
```

По умолчанию используется `local`.

Сервис также использует:
- `LOCAL_LLM_ENDPOINT` (например `http://127.0.0.1:8088`)
- `LOCAL_LLM_MODEL` (например `qwen-local`)

#### Сравнение local vs cloud (качество / скорость / стабильность)

Запустите:

```bash
python3 rag_local_eval.py --repeats 2
```

Скрипт сохранит отчет:
- `data/pipeline_outputs/rag_local_vs_cloud_report.json`

В отчете:
- `semantic_match` — прокси-оценка качества по control questions
- `avg_latency_ms` — скорость
- `success_rate` и `error_runs` — стабильность

### Структура модуля

```
document_indexer/
├── __init__.py
├── chunker.py              # Стратегии chunking
├── embedder.py             # Генерация эмбеддингов
├── index_storage.py        # Хранилища (FAISS/SQLite/JSON)
├── document_loader.py      # Загрузка документов
├── indexer_pipeline.py     # Оркестратор пайплайна
└── EVALUATION_PROMPTS.md   # Промпты для тестирования

sample_documents/           # Тестовые документы
├── 01_vector_databases_guide.md
├── 02_python_async_guide.md
├── 03_machine_learning_basics.md
├── 04_api_design_principles.md
├── 05_data_structures_algorithms.md
├── 06_system_design.md
├── 07_database_optimization.md
├── 08_security_best_practices.md
├── 09_rag_pdf_brief.pdf
├── 10_kotlin_mobile_basics.md
├── 11_compose_multiplatform_guide.md
└── 12_mobile_architecture_release.md
```

### Актуальные источники для Day 22 (mobile RAG)

Для контрольных вопросов и сравнения качества сейчас используются в первую очередь:
- `sample_documents/10_kotlin_mobile_basics.md`
- `sample_documents/11_compose_multiplatform_guide.md`
- `sample_documents/12_mobile_architecture_release.md`

Именно из них ожидаются основные факты по Kotlin, Compose Multiplatform и мобильной архитектуре.

### API Endpoints

```bash
# Индексация
curl -X POST http://127.0.0.1:5052/api/index \
  -H "Content-Type: application/json" \
  -d '{"strategy": "fixed_size", "storage": "faiss", "documents": []}'

# Поиск
curl -X POST http://127.0.0.1:5052/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "векторные базы данных", "strategy": "fixed_size", "top_k": 5}'

# Сравнение стратегий
curl -X POST http://127.0.0.1:5052/api/compare \
  -H "Content-Type: application/json" \
  -d '{"document": "sample_documents/01_vector_databases_guide.md"}'

# Статистика
curl http://127.0.0.1:5052/api/stats
```
