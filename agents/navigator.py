import langchain
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.types import Command
from langchain_ollama import OllamaLLM
from .utils import parse_output, State


class Navigator(Runnable):
    def __init__(self):
        self.api_key = "68c551c86aa0a3b983bd8e383e900e14"
        self.prompt = ChatPromptTemplate.from_messages([
        ("system", """
            你现在导航路线规划者，请找出路线的起点与终点
            请注意,ATTENTION!必须严格按以下JSON格式响应：
            ```json{{
                "role": "assistant", (Must have)
                "content": "content", (Must have)
                "origin": "起点",
                "destination": "终点",
            }}
            """),
            ("human", "当前输入：{input}"),
        ])
        self.llm = OllamaLLM(model="deepseek-r1:7b", base_url="http://localhost:11434")
        self.chain = (
            self.prompt
            | self.llm
            | RunnableLambda(parse_output)
        )
        
    def geocode(self, address, api_key):
        url = "https://restapi.amap.com/v3/geocode/geo"
        params = {
            "key": api_key,
            "address": address
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1" and data["count"] != "0":
            location = data["geocodes"][0]["location"]
            return location
        else:
            return None

    def get_route(self, origin, destination, api_key):
        url = "https://restapi.amap.com/v3/direction/walking"
        params = {
            "key": api_key,
            "origin": origin,
            "destination": destination
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "1":
            route = data["route"]["paths"][0]
            return route
        else:
            return None
        
    def invoke(self, state: State, *args, **kwargs):
        print("--------Navigator Working--------")
        inputs = self.chain.invoke({"input":state["messages"][-1]})
        origin = inputs["origin"]
        destination = inputs["destination"]
        origin_location = self.geocode(origin, self.api_key)
        destination_location = self.geocode(destination, self.api_key)
        if origin_location and destination_location:
            print(f"起点经纬度: {origin_location}")
            print(f"终点经纬度: {destination_location}")
            route = self.get_route(origin_location, destination_location, self.api_key)
            if route:
                return Command(goto="printer", update={"messages": [{"role": "assistant", "content": [route]}]})
            else:
                return Command(goto="printer", update={"messages": ["无法获取导航路线"]})
        else:
            return Command(goto="printer", update={"messages": ["无法获取经纬度"]})
        
        
    def get_location(self, address):
        pass


if __name__ == "__main__":
    navigator = Navigator()
    navigator.run({"origin": "深圳市清华国际一期", "destination": "深圳市世界之窗"})