import os
import json
import time
from utils import parse_output
from tools import human_assistance
from typing import Annotated
from typing_extensions import TypedDict
import requests

from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_core.tools import Tool, tool
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition


os.environ["BING_SUBSCRIPTION_KEY"] = "686d2128-1300-4586-9f55-240c7846af25"
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
os.environ["GOOGLE_CSE_ID"] = "428c301505bfc454d"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBht5XZMoOIDU8B_MykOzN09Yncov0tzLQ"


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            try:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
            except requests.exceptions.SSLError as e:
                print(f"HTTP error occurred: {e} retry later")
               
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    # if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0 and ai_message.tool_calls[0]["name"] == "END":
    #     return END
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

graph_builder = StateGraph(State)
config = {"configurable": {"thread_id": "25315"}}


llm = OllamaLLM(model="deepseek-r1:70b", base_url="http://localhost:11434")
search = GoogleSearchAPIWrapper()
tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)
human_assistance = Tool(
    name="human_assistance",
    description="Request assistance from a human.",
    func=human_assistance,
    args_schema={}
)

end_tool = Tool(
    name="END",
    description="End the conversation.",
    func=lambda x: "Goodbye!",
)
tools = [tool, human_assistance]
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)



prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你现在是聊天机器人，在不必要调用工具时不要调用工具，可用工具列表：
    {tool_descs}
    请注意,ATTENTION!必须严格按以下JSON格式响应：
    ```json{{
      "role": "assistant", (Must have)
      "content": "回答内容", (Must have)
      "tool_calls": [
        {{
            "fuction":{{
                "name":"工具名称(可选值：{tool_names})", 
                "arguments":{{"arg"}},
            }},
            "id": "id",
        }}
      ],
      "tool_call_id": "id"
    }}
    """),
    ("human", "当前输入：{input}"),
    ("system", "历史输入：{history}")
])
tool_names = [t.name for t in tools]
tool_descs = "\n".join([f"- {t.name}: {t.description}" for t in tools])
chain = (
    prompt.partial(tool_descs=tool_descs, tool_names=", ".join(tool_names)) 
    | llm 
    | RunnableLambda(parse_output)
)

def chatbot(state: State):
    return {"messages": [chain.invoke({"input":state["messages"][-1], "history":state["messages"][-6:-1]})]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
        for value in event.values():
            if isinstance(value, dict):
                if hasattr(value["messages"][-1], "content"):
                    print("Assistant:", value["messages"][-1].content)
                else:
                    print("Assistant:", value["messages"][-1]["content"])
            else:
                print("Assistant:", value)
        human_response = (
            "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
            " It's much more reliable and extensible than simple autonomous agents."
        )

        human_command = Command(resume={"data": human_response})

        events = graph.stream(human_command, config, stream_mode="values")
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
        # snapshot = graph.get_state(config)
        # print(snapshot)


while True:
    # try:
    #     user_input = input("User: ")
    #     if user_input.lower() in ["quit", "exit", "q"]:
    #         print("Goodbye!")
    #         break

    #     stream_graph_updates(user_input)
    # except:
    #     # fallback if input() is not available
    #     user_input = "What do you know about LangGraph?"
    #     print("User: " + user_input)
    #     stream_graph_updates(user_input)
    #     break
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)