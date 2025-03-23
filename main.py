import os
from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agents import Guardian, Navigator, Printer, Recognizer, Router, Receiver
from agents.utils import State
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 定义请求模型
class ChatRequest(BaseModel):
    messages: str
    config: dict

graph_builder = StateGraph(State)
config = {"configurable": {"thread_id": "25315"}}
agents_list = ["guardian", "navigator", "recognizer"]

# Initialize the agents
guardian = Guardian()
navigator = Navigator()
printer = Printer()
receiver = Receiver()
recognizer = Recognizer()
router = Router(agents_list=agents_list)

# Build the graph
graph_builder.add_node("guardian", guardian)
graph_builder.add_node("navigator", navigator)
graph_builder.add_node("printer", printer)
graph_builder.add_node("receiver", receiver)
graph_builder.add_node("recognizer", recognizer)
graph_builder.add_node("router", router)

graph_builder.add_edge(START, "receiver")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates():
    for event in graph.stream({"messages": [{"role": "system", "content": "start"}]}, config):
        print("Event done")

# 定义 API 路由
@app.post("/chat")
def chat(request: ChatRequest):
    """
    接收导航请求并返回事件流结果。
    """
    results = []
    for event in graph.stream({"messages": [{"role": "user", "content": request.messages}]}, request.config):
        results.append(event)
    return {"status": "success", "results": results}

@app.get("/")
def read_root():
    return {"Hello": "World"}

stream_graph_updates()# 我要从集悦城A区导航至深圳湾公园
    