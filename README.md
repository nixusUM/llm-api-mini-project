# LLM API Mini Project (Claude)

Мини-проект для демонстрации первого запроса к LLM через API:
- CLI режим (`python3 cli.py`)
- Web режим (`python3 web.py`)
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
```

`ANTHROPIC_MODEL` optional: если оставить пустым, проект сам подберет доступную модель.

## 2) Run CLI

```bash
python3 cli.py
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
└── 09_rag_pdf_brief.pdf
```

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
