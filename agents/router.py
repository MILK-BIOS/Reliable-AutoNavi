import os
import json
import time
from .utils import parse_output, State
from .tools import human_assistance
from typing import Annotated
from typing_extensions import TypedDict
import requests

from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool, tool
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal


class Router(Runnable):
    def __init__(self, agents_list):
        self.llm = OllamaLLM(model="deepseek-r1:32b", base_url="http://localhost:11434")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你现在是工作的Router，你需要调用其他Agents完成指令，可用Agents列表：
            {agents_list}
            请注意,ATTENTION!必须严格按以下JSON格式响应，将下一个Agent必要的信息写到content中：
            若无需调用Agent，则在next_agent中使用printer
            ```json{{
                "role": "assistant", (Must have)
                "content": "回答内容", (Must have)
                "next_agent": "下一个Agent名称",
            }}
            """),
            ("human", "当前输入：{input}"),
            ("system", "历史输入：{history}")
        ])
        self.chain = (
            self.prompt.partial(agents_list=agents_list)
            | self.llm
            | RunnableLambda(parse_output)
        )

    def invoke(self, state: State, *args, **kwargs):
        print("--------Router Working--------")
        response = self.chain.invoke({"input":state["messages"][-1], "history":state["messages"][-3:-1]})
        print("Handoff to agent: ", response["next_agent"])
        return Command(goto=response["next_agent"], update={
            "messages": [{"role": "assistant", "content": response["content"]}],
        })