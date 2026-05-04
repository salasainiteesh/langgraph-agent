from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

# 1. Define a tool
@tool
def calculator(expression: str) -> str:
    """Use this to solve math calculations. Input should be a math expression like '25 * 48'"""
    try:
        return str(eval(expression))
    except:
        return "Could not calculate that"

# 2. State
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# 3. LLM with tools bound to it
tools = [calculator]
llm = ChatOllama(model="llama3-groq-tool-use")
llm_with_tools = llm.bind_tools(tools)

# 4. Agent node - calls LLM
def call_llm(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 5. Router - should we use a tool or end?
def should_use_tool(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"   # go to tool node
    return END           # no tool needed, end

# 6. Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode(tools))   # tool executor node

graph.set_entry_point("agent")

# Conditional edge - agent decides which way to go
graph.add_conditional_edges("agent", should_use_tool)

# After tool runs, go back to agent
graph.add_edge("tools", "agent")

# 7. Compile with memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "chat-1"}}

# Test 1 - needs tool
result1 = app.invoke(
    {"messages": [HumanMessage(content="What is 25 * 48?")]},
    config=config
)
print("Math question:", result1["messages"][-1].content)

# Test 2 - no tool needed
result2 = app.invoke(
    {"messages": [HumanMessage(content="What is the capital of France?")]},
    config=config
)
print("General question:", result2["messages"][-1].content)