import os
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agents import Guardian, Navigator, Printer, Recognizer, Router, Receiver
from agents.utils import State
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from detect import YOLODetector
from pathlib import Path
import threading
import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 定义请求模型
class ChatRequest(BaseModel):
    messages: str
    current_address: str = "" 
    latitude: float = 0.0      
    longitude: float = 0.0           
    config: dict


graph_builder = StateGraph(State)
config = {"configurable": {"thread_id": "25315"}}
agents_list = ["guardian", "navigator", "recognizer"]

# Initialize the agents
YOLODetector = YOLODetector()
guardian = Guardian()
navigator = Navigator()
printer = Printer()
receiver = Receiver()
recognizer = Recognizer()
router = Router(agents_list=agents_list)

# Build the graph
graph_builder.add_node("guardian", guardian)
graph_builder.add_node("navigator", navigator)
graph_builder.add_node("printer", printer)
graph_builder.add_node("receiver", receiver)
graph_builder.add_node("recognizer", recognizer)
graph_builder.add_node("router", router)

graph_builder.add_edge(START, "receiver")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates():
    for event in graph.stream({"messages": [{"role": "system", "content": "start"}]}, config):
        print("Event done")

# 定义 API 路由
@app.post("/chat")
def chat(request: ChatRequest):
    """
    接收导航请求并返回事件流结果。
    """
    try:
        initial_state = {
            "messages": [{"role": "user", "content": request.messages}],
            "current_address": request.current_address,
            "latitude": request.latitude,
            "longitude": request.longitude
        }
        results = []
        for event in graph.stream(initial_state, request.config):
            results.append(event)
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/upload")
async def upload_frame(frame: UploadFile = File(...)):
    """
    接收视频帧并保存到本地
    """
    try:
        # 保存上传的帧到本地
        file_path = SAVE_DIR / frame.filename
        with file_path.open("wb") as f:
            content = await frame.read()
            f.write(content)
        return {"message": "帧上传成功", "file_path": str(file_path), "processed_file": frame.filename}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"帧上传失败: {str(e)}"})

@app.get("/processed/{filename}")
async def get_processed_frame(filename: str):
    """
    返回处理好的图片帧并删除文件
    """
    file_path = RETURN_DIR / filename
    if file_path.exists():
        # 返回文件响应
        response = FileResponse(file_path)
        # 删除文件
        try:
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        except Exception as e:
            print(f"删除文件时发生错误: {e}")
        return response
    return {"message": "文件不存在"}


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    # stream_graph_updates()# 我要从集悦城A区导航至深圳湾公园 
    import uvicorn
    uvicorn.run(
        "main:app",  # 模块路径（字符串形式）
        host="0.0.0.0",
        port=9543,
        reload=True,    # ⚠️ 注意：编程式调用不支持 reload 参数
        workers=1,      # 进程数（生产环境推荐）
        log_level="info",
        access_log=False # 禁用访问日志
    )
    