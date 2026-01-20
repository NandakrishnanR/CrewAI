import os
import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
import uvicorn

# 1. Setup Env
os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
os.environ["CREWAI_LLM_PROVIDER"] = "ollama"

app = FastAPI(title="CrewAI Backend")

# 2. CORS & Proxy Support (Crucial for React + Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Local LLM
llm = LLM(model="ollama/llama3.1:latest", base_url="http://localhost:11434")

# 4. Agents
cleaner = Agent(
    role='Data Cleaner',
    goal='Summarize data issues in <3 sentences.',
    backstory='Senior Data Scientist specializing in preprocessing.',
    llm=llm, verbose=True
)

selector = Agent(
    role='Algorithm Selector',
    goal='Return JSON ONLY: {"task": "...", "model": "...", "reason": "..."}',
    backstory='ML Engineer expert in model selection.',
    llm=llm, verbose=True
)

generator = Agent(
    role='Code Generator',
    goal='Write full runnable sklearn Python code.',
    backstory='Senior Python Developer.',
    llm=llm, verbose=True
)

# 5. API Endpoint (Matches vite.config.js proxy to /api)
@app.post("/api/run")
async def run_process(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        summary = f"Rows: {len(df)}, Cols: {list(df.columns)}, Types: {df.dtypes.to_dict()}"

        t1 = Task(description=f"Analyze: {summary}", agent=cleaner, expected_output="Summary")
        t2 = Task(description="Recommend JSON algo", agent=selector, context=[t1], expected_output="JSON")
        t3 = Task(description="Write sklearn code", agent=generator, context=[t1, t2], expected_output="Python Code")

        crew = Crew(agents=[cleaner, selector, generator], tasks=[t1, t2, t3], verbose=True)
        crew.kickoff()

        return {
            "data_cleaner": str(t1.output),
            "algorithm_selector": str(t2.output),
            "code_generator": str(t3.output)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Run on port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
