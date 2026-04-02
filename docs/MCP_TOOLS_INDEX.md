# MCP Tools Index (Auto-generated)

Этот файл обновляется ассистентом операций с файлами.

| Tool | Description |
|------|-------------|
| `clear_periodic_state` | Delete all periodic jobs/history and reset scheduler storage. |
| `configure_periodic_summary` | Create or update periodic summary job with schedule settings. |
| `get_current_git_branch` | Return current git branch and short commit for the project (or repo_root if inside project). |
| `get_exchange_rate` | Get latest exchange rate (Frankfurter API, free). Example: get_exchange_rate('USD', 'EUR'). |
| `get_random_quote` | Get a random quote (Quotable API, free). |
| `get_summary_report` | Return latest periodic runs and aggregated completion metrics. |
| `get_support_ticket` | Load support ticket from local JSON ticket store. |
| `get_support_ticket_context` | Return joined ticket + user context for support assistant prompts. |
| `get_support_user` | Load support user profile from local JSON CRM mock. |
| `get_todo_from_mock_api` | Fetch one todo item from JSONPlaceholder mock API by todo_id. |
| `get_weather` | Get current weather for a city (Open-Meteo, no API key). Example: get_weather('London') or get_weather('Moscow', 'celsius'). |
| `get_working_tree_diff_stat` | Unstaged changes summary (git diff --stat). |
| `list_project_tracked_files` | List git-tracked files (optional pathspec glob, e.g. '*.py'). |
| `ping` | - |
| `run_due_summaries` | Execute all due periodic jobs and return aggregated summary. |
| `save_to_file` | Save text content to data/pipeline_outputs and return file metadata. |
| `search_data` | Search mock post data by query string (title/body). |
| `sum_two_numbers` | - |
| `summarize_data` | Summarize search tool output (expects JSON string from search_data). |

## Working tree snapshot

- `README.md`
- `templates/index_modern.html`
- `web.py`
