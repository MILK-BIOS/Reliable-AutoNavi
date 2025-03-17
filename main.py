import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agents import Guardian, Navigator, Printer, Recognizer, Router
from agents.utils import State


graph_builder = StateGraph(State)
config = {"configurable": {"thread_id": "25315"}}
agents_list = ["guardian", "navigator", "recognizer"]

# Initialize the agents
guardian = Guardian()
navigator = Navigator()
printer = Printer()
recognizer = Recognizer()
router = Router(agents_list=agents_list)

# Build the graph
graph_builder.add_node("guardian", guardian)
graph_builder.add_node("navigator", navigator)
graph_builder.add_node("printer", printer)
graph_builder.add_node("recognizer", recognizer)
graph_builder.add_node("router", router)

graph_builder.add_edge(START, "router")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
        print("Event done")


while True:
    user_input = input("Enter your message: ") # 我要从集悦城A区导航至深圳湾公园
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)
    