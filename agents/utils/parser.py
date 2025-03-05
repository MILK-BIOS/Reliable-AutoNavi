from typing import Union
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

def parse_output(text: str) -> Union[dict, str]:
    """解析模型输出中的工具调用指令"""
    try:
        # 提取JSON部分（适配不同模型的输出风格）
        json_str = text.split("```json")[1].split("```")[0].strip()
        return JsonOutputParser().parse(json_str)
    except:
        return {"role": "assistant", "content": text}