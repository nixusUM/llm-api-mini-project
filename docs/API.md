# HTTP API и данные (кратко)

Порт веб-приложения задаётся при запуске `python3 web.py` (часто `5051`).

## Веб (Flask)

| Метод | Путь | Назначение |
|--------|------|------------|
| GET, POST | `/` | Главная страница UI, формы (в т.ч. ассистент разработчика в режиме «Автоматизация»). |
| GET | `/scheduler_status` | JSON: `running`, `poll_seconds`, `last_tick_at`, `last_status`, `last_report`. |
| POST | `/api/rag_query` | Тело JSON: `question`, опционально `top_k_before`, `top_k_after`, `threshold`, `enable_query_rewrite`. Ответ — результат RAG. |
| POST | `/api/rag_audit` | Параметры как у RAG; прогон контрольных вопросов. |
| POST | `/api/rag_chat` | JSON: `session_id`, `question`, опции top_k/threshold. |
| POST | `/api/rag_chat_reset` | JSON: `session_id`. |
| POST | `/api/rag_chat_scenarios` | Сценарии чата RAG. |
| POST | `/api/rag_compare` | JSON: `repeats`, `question_limit`, параметры RAG. |
| GET | `/document_indexer` | Редирект/подсказка для отдельного индексатора. |

## Локальная LLM (OpenAI-совместимый чат)

Telegram-бот и блок ассистента в UI ходят в endpoint из `.env`:

- `POST {LOCAL_LLM_ENDPOINT}/v1/chat/completions`
- Поля: `model`, `messages` (`role`/`content`), `temperature`, `max_tokens`.

## MCP: контекст репозитория

Сервер: `mcp_local_server.py` (stdio). Инструменты для ассистента разработчика:

- `get_current_git_branch` — ветка и короткий hash; ответ: `ok`, `branch`, `commit_short`, при ошибке `error`.
- `list_project_tracked_files` — аргументы: `max_files`, опционально `path_pattern`, `repo_root`; ответ: `ok`, `files`, `truncated`.
- `get_working_tree_diff_stat` — аргументы: `max_lines`, опционально `repo_root`; ответ: `ok`, `diff_stat`.

## Файлы состояния (схемы на уровне файлов)

- `data/mcp_periodic_state.json` — периодические задачи MCP-демо (`jobs`, `history`).
- `data/agent_state.json` — состояние агента UI (если используется).
- `data/chat_history.json` — история чата (если пишется из UI).

Пересборка RAG для `/help`: `python3 build_dev_assistant_index.py` → `document_indices/index_dev_assistant.json`.
# demo change
