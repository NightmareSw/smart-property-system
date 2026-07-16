from fastapi import FastAPI

# 1. 创建 FastAPI 应用实例
app = FastAPI()

# 2. 定义一个路径操作装饰器，告诉 FastAPI 下面的函数处理根路径 "/" 的 GET 请求
@app.get("/")
async def read_root():
    # 3. 返回一个 JSON 响应
    return {"message": "Hello, FastAPI!"}