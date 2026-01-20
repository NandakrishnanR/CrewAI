
# --- CrewAI-based implementation ---
import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from crewai import Agent, Task, Crew, Process

app = FastAPI()

def build_csv_summary(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    max_cols = 12
    column_names = list(df.columns[:max_cols])
    if cols > max_cols:
        column_names.append("...")
    dtype_snapshot = {col: str(df[col].dtype) for col in df.columns[:max_cols]}
    missing_snapshot = {col: int(df[col].isnull().sum()) for col in df.columns[:max_cols] if df[col].isnull().any()}
    sample_row = {}
    if rows:
        sample_row = {col: str(val)[:40] for col, val in df.iloc[0].items() if col in df.columns[:max_cols]}
    numeric_preview = {}
    numeric_subset = df.select_dtypes(include="number").iloc[:, :5]
    if not numeric_subset.empty:
        describe = numeric_subset.describe().round(3).iloc[1:]
        numeric_preview = {col: {idx: str(val)[:40] for idx, val in describe[col].items()} for col in describe.columns}
    summary = {
        "rows": int(rows),
        "cols": int(cols),
        "columns": column_names,
        "dtypes": dtype_snapshot,
        "missing": missing_snapshot,
        "sample": sample_row,
        "numeric_stats": numeric_preview,
    }
    import json
    summary_text = json.dumps(summary, separators=(",", ":"))
    if len(summary_text) > 1800:
        summary_text = summary_text[:1800] + "..."
    return summary_text

@app.post("/run")
async def run_agents(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return JSONResponse({"error": f"Invalid CSV: {str(e)}"}, status_code=400)
    summary = build_csv_summary(df)

    # Define agents
    data_cleaner = Agent(
        role="Data Cleaner",
        goal="Summarize data issues in <=2 sentences: missing %, dtype notes, obvious scaling needs.",
        backstory="You are a data scientist who quickly inspects and summarizes data issues."
    )
    algorithm_selector = Agent(
        role="Algorithm Selector",
        goal="Recommend exactly one algorithm. Respond ONLY as JSON with keys task, model, reason. Set task to classification or regression based on target characteristics; set model to the precise sklearn class name; keep reason under 12 words.",
        backstory="You are a machine learning expert who always chooses the best estimator."
    )
    code_generator = Agent(
        role="Code Generator",
        goal="Produce a full runnable Python script in triple backticks. Use only the estimator recommended by the Algorithm Selector. Include preprocessing, training, metrics, and a plot.",
        backstory="You are a Python ML engineer who writes clean, ready-to-run code."
    )

    # Define tasks
    task1 = Task(
        description=f"Given this data summary, {data_cleaner.goal}\nData: {summary}",
        agent=data_cleaner
    )
    task2 = Task(
        description=f"Given the Data Cleaner output, {algorithm_selector.goal}",
        agent=algorithm_selector
    )
    task3 = Task(
        description=f"Given the Algorithm Selector output, {code_generator.goal}",
        agent=code_generator
    )

    # Orchestrate with Crew
    crew = Crew(
        agents=[data_cleaner, algorithm_selector, code_generator],
        tasks=[task1, task2, task3],
        process=Process.sequential
    )
    results = crew.kickoff()

    # Parse results
    agents_out = [
        {"role": agent.role, "content": str(res)}
        for agent, res in zip([data_cleaner, algorithm_selector, code_generator], results)
    ]
    # Try to extract code from the last agent
    import re
    code = ""
    if results and isinstance(results[-1], str):
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', results[-1], re.DOTALL | re.IGNORECASE)
        if code_blocks:
            code = code_blocks[-1].strip()
        else:
            code = results[-1].strip()

    return {"agents": agents_out, "code": code}
