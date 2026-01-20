# CrewAI-Style Multi-Agent System (Local, Free)

This project demonstrates agent-to-agent communication using three agents (Data Cleaner, Algorithm Selector, Code Generator) with CrewAI (open source, free) and FastAPI.

## Features
- Upload a CSV file via API
- Data flows through three agents, each able to pass messages/state
- Final output is generated Python code
- All runs locally, no paid services

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the API:**
   ```bash
   uvicorn crewai_app:app --reload
   ```
3. **POST a CSV file** to `/run` using Postman, curl, or a simple frontend.

## File Structure
- `crewai_app.py` — FastAPI app and CrewAI agent orchestration
- `requirements.txt` — dependencies
- `README.md` — this file

## Notes
- All LLM calls are local via Ollama (no API keys needed)
- You can extend the agent graph for more complex workflows
