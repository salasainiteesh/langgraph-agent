from dotenv import load_dotenv
load_dotenv()

import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
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
    messages: Annotated[Sequence[AnyMessage], add_messages]

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

# ---- Compile ----
# No checkpointer needed - LangGraph Platform handles persistence!
app_graph = graph.compile()