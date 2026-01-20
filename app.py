import os
import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from crewai import Agent, Task, Crew
from crewai.llm import LLM
import uvicorn

# 1. Setup Env
os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
os.environ["CREWAI_LLM_PROVIDER"] = "ollama"

app = FastAPI(title="CrewAI Backend")

# 2. CORS (Allows connection from React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Local LLM
llm = LLM(model="ollama/llama3.1:latest", base_url="http://localhost:11434")

# 4. Agents Definitions
cleaner = Agent(
    role='Data Cleaner',    # Job title( It tells the agent what they are )
    goal='Summarize data issues in <3 sentences.',     # What to do
    backstory='Senior Data Scientist specializing in preprocessing.',    # Why they're qualified(This gives the LLM context to respond like an expert in that field)
    llm=llm, verbose=True
)

selector = Agent(
    role='Algorithm Selector',
    goal='"Evaluate the dataset and select the best SKLEARN MACHINE LEARNING ALGORITHM (classification or regression). Return JSON ONLY:" {"task": "...", "model": "...", "reason": "..."}',
    backstory='ML Engineer expert in model selection.',
    llm=llm, verbose=True
)

generator = Agent(
    role='Code Generator',
    goal='Write full runnable sklearn Python code.',
    backstory='Senior Python Developer.',
    llm=llm, verbose=True
)

# 5. API Endpoint
@app.post("/api/run")
async def run_process(file: UploadFile = File(...)):
    try:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            return JSONResponse({"error": "Invalid CSV file"}, status_code=400)
            
        summary = f"Rows: {len(df)}, Cols: {list(df.columns)}, Types: {df.dtypes.to_dict()}"

        t1 = Task(description=f"Analyze: {summary}", agent=cleaner, expected_output="Summary")
        t2 = Task(description="Recommend the best Machine Learning algorithm (e.g. Random Forest, Logistic Regression) for this dataset. Output ONLY valid JSON.", agent=selector, context=[t1], expected_output="JSON")
        t3 = Task(description="Write sklearn code", agent=generator, context=[t1, t2], expected_output="Python Code")

        crew = Crew(agents=[cleaner, selector, generator], tasks=[t1, t2, t3], verbose=True)
        crew.kickoff()

        # Helper to safely get output string
        def get_output(task):
            if hasattr(task.output, 'raw'): return task.output.raw
            if hasattr(task.output, 'raw_output'): return task.output.raw_output
            return str(task.output)

        return {
            "data_cleaner": get_output(t1),
            "algorithm_selector": get_output(t2),
            "code_generator": get_output(t3)
        }
    except Exception as e:
        print(f"Error: {e}") # Log to terminal
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
