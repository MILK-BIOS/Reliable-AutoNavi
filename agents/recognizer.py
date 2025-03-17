import langchain
from langchain_core.runnables import Runnable

class Recognizer(Runnable):
    def __init__(self):
        pass

    def invoke(self, input, config = None, **kwargs):
        return super().invoke(input, config, **kwargs)