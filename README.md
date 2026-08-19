# hier-brain

层级脑热切换推理服务。消费级 Mixture-of-LoRA 实现。

## 架构

```
Input → Thalamus(0.6B 路由) → Prefrontal(9B 整合) → [Experts(8B LoRA) + Hippocampus]
      → Prefrontal(整合) → Output
```

| 组件 | 基座 | 常驻/切换 |
|---|---|---|
| 丘脑 | Qwen3-0.6B | 常驻 |
| 前额叶 | Qwen3.5-9B | 常驻（目标 70B） |
| 专家 | Qwen3-8B + LoRA | 热切换(set_adapter) |
| 海马体 | Qwen3-0.6B | 常驻（编码/索引，非全量存储） |

## API 端点

| 方法 | 端点 | 说明 |
|---|---|---|
| GET | /v1/health | 存活 + 可用角色 |
| GET | /v1/models | 已加载基座/适配器 |
| POST | /v1/swap | 切换角色（预热） |
| POST | /v1/generate | 指定角色生成（自动热切换） |
| POST | /v1/load_adapter | 运行时热挂载新 LoRA |
| POST | /v1/brain | 完整层级脑流水线 |

## 运行

```bash
python serve.py  # 默认 127.0.0.1:8017
```

## 依赖

```
torch>=2.6, transformers>=5.0, peft>=0.20, bitsandbytes>=0.50, fastapi>=0.110, uvicorn>=0.27
```

## 与 Cerebro-Swap 的关系

hier-brain 是 PoC 版本，Cerebro-Swap 是生产化重构。hier-brain 用 ollama 调用，Cerebro-Swap 用本地 transformers + PEFT。
