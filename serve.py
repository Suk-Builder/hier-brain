#!/usr/bin/env python3
# serve.py — 层级脑热切换推理服务（消费级 Mixture-of-LoRA 服务器）
# 基座常驻 + LoRA 按角色热切换（set_adapter）= MoE 专家切换的显式版
#
# 端点:
#   GET  /v1/health                   存活 + 可用角色
#   GET  /v1/models                   已加载基座/适配器
#   POST /v1/swap     {"role"}        切换当前角色（预热）
#   POST /v1/generate {"role","prompt"}  指定角色生成（自动热切换）
#   POST /v1/load_adapter {"role","base"} 运行时热挂载新 LoRA（学科扩展）
#   POST /v1/brain    {"query"}       完整层级脑流水线（丘脑路由→专家→前额叶→海马体）
#
# 跑法: python serve.py  (默认 127.0.0.1:8017)

import uvicorn, threading
from fastapi import FastAPI
from pydantic import BaseModel

import peft_runtime as PR
import hier_brain as HB

app = FastAPI(title="hier-brain serve", version="0.1")

class SwapReq(BaseModel):
    role: str
class GenReq(BaseModel):
    role: str
    prompt: str
    max_tokens: int = 120
class LoadReq(BaseModel):
    role: str
    base: str = "8b"
class BrainReq(BaseModel):
    query: str

@app.get("/v1/health")
def health():
    return {"status": "ok", "roles": list(PR.ROLE_BASE.keys())}

@app.get("/v1/models")
def models():
    return {
        "loaded_bases": list(PR._runtime.keys()),
        "available_roles": list(PR.ROLE_BASE.keys()),
    }

@app.post("/v1/swap")
def swap(req: SwapReq):
    """热切换：确保角色适配器已加载，下次 generate 直接换帽。"""
    PR._get_runtime(req.role)
    return {"active": req.role}

@app.post("/v1/generate")
def generate(req: GenReq):
    """指定角色生成：基座常驻，按请求热切换 LoRA。"""
    return {"role": req.role, "output": PR.generate(req.role, req.prompt, req.max_tokens)}

@app.post("/v1/load_adapter")
def load_adapter(req: LoadReq):
    """运行时热挂载新 LoRA（学科扩展）：adapter 须已存在于 adapters/<role>/。"""
    return PR.register_role(req.role, req.base)

@app.post("/v1/brain")
def brain(req: BrainReq):
    """完整层级脑流水线：丘脑路由 → 专家(热切换) → 前额叶整合 → 海马体巩固。"""
    return {"answer": HB.run(req.query)}

if __name__ == "__main__":
    print("[serve] hier-brain hot-swap server starting on 127.0.0.1:8017 ...")
    uvicorn.run(app, host="127.0.0.1", port=8017, log_level="warning")
