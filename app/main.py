import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# 1. Load context
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
db_url = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/agent_db")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment")

# 2. Define State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 3. Define the Agent
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Using a stable model for production
    google_api_key=api_key,
    temperature=0
)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 4. Build the Graph
workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 5. Connect to Postgres for Memory
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

pool = ConnectionPool(
    conninfo=db_url,
    max_size=20,
    kwargs=connection_kwargs,
)

checkpointer = PostgresSaver(pool)
# IMPORTANT: Run migrations once to setup the schema
# checkpointer.setup() # This is handled automatically in recent versions, but check docs if issues arise

# 6. Compile the App with Checkpointer
app = workflow.compile(checkpointer=checkpointer)

def main():
    print("Agent is ready!")
    
    # Example usage (in a real app, this would be triggered by an API or CLI)
    config = {"configurable": {"thread_id": "1"}}
    
    # First turn
    # response = app.invoke({"messages": [HumanMessage(content="Hi, I'm Abhishek")]}, config)
    # print(response["messages"][-1].content)
    
    # Second turn (it should remember the name)
    # response = app.invoke({"messages": [HumanMessage(content="What's my name?")]}, config)
    # print(response["messages"][-1].content)

if __name__ == "__main__":
    main()
