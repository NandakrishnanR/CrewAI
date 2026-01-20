
<img src="public/Screenshot%202026-01-20%20112756.png" alt="Full Screen Screenshot" style="width:100%; display:block; margin-bottom:10px;" />
<img src="public/Screenshot%202026-01-20%20113243.png" alt="Detail Screenshot" style="width:100%; display:block; margin-bottom:20px;" />

# CrewAI Multi-Agent System

This project shows how multiple AI agents can work together to solve a problem. You upload a CSV file, and three agents (Data Cleaner, Algorithm Selector, and Code Generator) process it and give you Python code as output. Everything runs locally and is free.

## Why I Did This

I built this project to show how you can automate data science tasks using multiple AI agents, all running locally and for free. It’s a practical example for anyone interested in agent-based workflows, and it helps you understand how to connect different tools and ideas in one system.

## Main Difference: LangGraph vs CrewAI

LangGraph is a framework for building complex agent graphs, where agents can interact in flexible ways and you can design custom workflows. CrewAI is focused on making agent roles and tasks simple to define, so you can quickly set up a team of agents with clear responsibilities. LangGraph is more about structure and connections; CrewAI is more about agent behavior and teamwork.

## CrewAI Concepts: Role, Background, etc.

- **Role:** This is what each agent is supposed to do. For example, one agent is a Data Cleaner, another is an Algorithm Selector, and another is a Code Generator. The role defines their main job in the workflow.
- **Background:** This is extra information about the agent, like their experience or specialty. It helps the agent make better decisions and gives context to their actions.
- **Goal:** What the agent is trying to achieve. For example, the Data Cleaner’s goal is to prepare the data for analysis.
- **Instructions:** These are the steps or rules the agent follows to do its job. You can customize instructions to change how the agent works.


## What does it do?

You send a CSV file to the API. The Data Cleaner agent cleans your data, the Algorithm Selector agent picks the best algorithm, and the Code Generator agent writes Python code for you. The process is automatic and does not use paid services or cloud APIs.

## How was it built?

- CrewAI is used to build the agent workflow.
- FastAPI provides the web API for uploading files and getting results.
- All AI runs locally using Ollama, so you don’t need API keys or internet access.

## Getting Started

1. Install Python dependencies:
   pip install -r requirements.txt
2. Start the API server:
   uvicorn crewai_app:app --reload
3. Upload your CSV file to the `/run` endpoint using Postman, curl, or any frontend.

## Project Files

- crewai_app.py: Main FastAPI app and agent logic
- requirements.txt: List of required Python packages
- README.md: This guide

## Notes

All AI processing is done locally. You can extend the workflow by adding more agents or changing their tasks.
