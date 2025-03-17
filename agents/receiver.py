from langchain_core.runnables import Runnable
from langgraph.types import Command
from .utils import State


class Receiver(Runnable):
    def __init__(self):
        self.receiver = None

    def set_receiver(self, receiver):
        self.receiver = receiver

    def receive(self, message):
        self.receiver.receive(message)

    def invoke(self, state: State, *args, **kwargs):
        message = state["messages"][-1]
        content_data = message.content[0]
        if 'steps' in content_data:
            steps = content_data["steps"]
            for step in steps:
                print(step)
        else:
            print(message.content)
        return Command(goto="router", update={"messages": ["Receiver invoked"]})