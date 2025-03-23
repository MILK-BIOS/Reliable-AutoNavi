from rich.console import Console
from rich.table import Table
from langchain_core.runnables import Runnable
from langgraph.types import Command
from langgraph.graph import END
from .utils import State


class Printer(Runnable):
    def __init__(self):
        self.printer = None

    def set_printer(self, printer):
        self.printer = printer

    def print(self, message):
        self.printer.print(message)

    def invoke(self, state: State, *args, **kwargs):
        print("--------Printer Working--------")
        message = state["messages"][-1]
        content_data = message.content[0]
        if 'steps' in content_data:
            steps = content_data["steps"]
            console = Console()
            table = Table(title="Steps")
            table.add_column("Instrution", style="cyan")
            table.add_column("Orientation", style="magenta")
            table.add_column("Distance", style="yellow")
            table.add_column("Duration", style="green")
            table.add_column("action", style="blue")
            for step in steps:
                table.add_row(str(step["instruction"]),
                              str(step["orientation"]), 
                              str(step["distance"]), 
                              str(step["duration"]), 
                              str(step["action"]))
            console.print(table)
        else:
            print(message.content)
        return Command(goto=END, update={"messages": ["Printer invoked"]})