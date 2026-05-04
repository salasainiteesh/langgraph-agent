from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

# ---- Tool ----
@tool
def calculator(expression: str) -> str:
    """Use this to solve math calculations. Input should be a math expression like '25 * 48'"""
    try:
        return str(eval(expression))
    except:
        return "Could not calculate that"

# ---- State ----
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# ---- LLM ----
tools = [calculator]
llm = ChatOllama(
    model="llama3-groq-tool-use",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)
llm_with_tools = llm.bind_tools(tools)

# ---- Nodes ----
def call_llm(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_use_tool(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# ---- Graph ----
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_use_tool)
graph.add_edge("tools", "agent")

# ---- Compile with memory ----
memory = MemorySaver()
app_graph = graph.compile(checkpointer=memory)

# ---- FastAPI ----
app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    result = app_graph.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config
    )
    return ChatResponse(
        response=result["messages"][-1].content,
        thread_id=request.thread_id
    )