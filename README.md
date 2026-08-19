# hier-brain

层级脑热切换推理 PoC。ollama 版，Cerebro-Swap 的前代。

## 真实状态

只有 4 个文件的 PoC，**已被 Cerebro-Swap 替代**。Cerebro-Swap 将 ollama 调用替换为 transformers + PEFT，增加了 ExpertCachePool 和 HotSwapEngine。

## 架构

```
Input → Thalamus(0.6B 路由) → Prefrontal(9B 整合) → [Experts(8B LoRA) + Hippocampus]
      → Prefrontal(整合) → Output
```

| 组件 | 模型 | 常驻/切换 | 文件 |
|---|---|---|---|
| 丘脑 | peft:thalamus | 常驻 | hier_brain.py |
| 前额叶 | qwen3.5-9b-abl | 常驻 | hier_brain.py |
| 海马体 | peft:hippocampus | 常驻 | hier_brain.py |
| 专家 | peft:{role} | 热切换(set_adapter) | peft_runtime.py |

## 核心文件

| 文件 | 说明 |
|---|---|
| hier_brain.py | 主类：HierBrain，信号流编排，ollama 调用 |
| peft_runtime.py | PEFT 运行时：双基座(0.6b/8b)，LoRA 热切换，输出截断 |
| serve.py | FastAPI 服务：/v1/health, /v1/models, /v1/swap, /v1/generate, /v1/load_adapter, /v1/brain |
| requirements.txt | 依赖 |

## 与 Cerebro-Swap 的区别

| | hier-brain | Cerebro-Swap |
|---|---|---|
| 模型加载 | ollama | transformers + PEFT |
| 基座 | 0.6b + 9b | 0.6b + 8b |
| 专家切换 | set_adapter | set_adapter + ExpertCachePool |
| SSD 热切换 | 无 | 骨架 |
| 记忆巩固 | sqlite 海马体 | 未接入 runtime |
| 状态 | PoC，已归档 | 当前开发中 |

## API

端口 8017。

| 端点 | 说明 |
|---|---|
| GET /v1/health | 存活 + 可用角色 |
| GET /v1/models | 已加载基座/适配器 |
| POST /v1/swap | 切换角色（预热） |
| POST /v1/generate | 指定角色生成（自动热切换） |
| POST /v1/load_adapter | 运行时热挂载新 LoRA |
| POST /v1/brain | 完整层级脑流水线 |

## 运行

```bash
python serve.py  # 端口 8017
```

## 依赖

```
torch>=2.6, transformers>=5.0, peft>=0.20, bitsandbytes>=0.50, fastapi>=0.110, uvicorn>=0.27
```

## 状态

- PoC 完成，功能验证通过
- **已归档**，后续开发在 Cerebro-Swap
