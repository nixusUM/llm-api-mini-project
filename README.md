# LLM API Mini Project (Claude)

Мини-проект для демонстрации первого запроса к LLM через API:
- CLI режим (`python3 cli.py`)
- Web режим (`python3 web.py`)

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
