import os
import langchain
import langgraph
from langchain_ollama import OllamaLLM
from langchain_core.runnables import Runnable

class Guardian(Runnable):
    def __init__(self):
        self.llm = OllamaLLM(model="deepseek-r1:70b", base_url="http://localhost:11434")

    def invoke(self, input, config = None, **kwargs):
        return super().invoke(input, config, **kwargs)