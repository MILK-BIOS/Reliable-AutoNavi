from langchain_core.runnables import Runnable
from langgraph.types import Command
from langgraph.graph import END
from .utils import State


class Receiver(Runnable):
    def __init__(self):
        self.receiver = None

    def set_receiver(self, receiver):
        self.receiver = receiver

    def receive(self, message):
        self.receiver.receive(message)

    def invoke(self, state: State, *args, **kwargs):
        print("--------Receiver Working--------")
        # user_input = input("Enter your message: ") # 我要从集悦城A区导航至深圳湾公园
        if isinstance(state, dict):
            user_input = state["messages"][-1].content
        else:
            if "content" in state.messages[-1]:
                user_input = state.messages[-1]["content"]
            elif hasattr(state.messages[-1], "content"):
                user_input = state.messages[-1].content
            else:
                return Command(goto="receiver", update={"messages": [{"role": "user", "content": user_input}]})
        # if user_input.lower() in ["quit", "exit", "q"]:
        #     print("Goodbye!")
        #     return Command(goto=END, update={"messages": ["END"]})
        return Command(goto="router", update={"messages": [{"role": "user", "content": user_input}]})