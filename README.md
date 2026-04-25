@"

# 🤖 Multi-Chatbot Gateway

A scalable multi-tenant chatbot gateway that serves multiple AI-powered customer support bots for different websites.

## 🚀 Features

- **Multi-Bot Support**: Single server handling multiple chatbots
- **Database Integration**: Real-time data from PostgreSQL via Supabase
- **RAG Architecture**: Context-aware responses using static docs + live data
- **Production Ready**: Docker support and GitHub Actions

## 🛠️ Tech Stack

- FastAPI
- Groq (LLaMA 3.1)
- Supabase (PostgreSQL)
- Python 3.11+

## 📦 Installation

1. Clone the repository
2. Copy `.env.example` to `.env` and add your API keys
3. Run `pip install -r requirements.txt`
4. Run `python main.py`

## 🔗 API Endpoints

- `POST /api/chat/{bot_id}` - Chat with specific bot
- `GET /api/health` - Health check

## 📝 License

MIT
"@ | Out-File -FilePath README.md -Encoding utf8
