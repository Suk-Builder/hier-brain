# hier-brain

层级角色 LoRA 训练流水线。消费级 MoL（Mixture-of-LoRA）实现。

## 核心差异

| 特性 | 工业 MoE | hier-brain |
|---|---|---|
| 路由 | 隐式 token 级 | 显式语义域级 |
| 基座 | 749B | 8B |
| 硬件 | A100/H100 集群 | RTX 3060/4090 |
| LoRA | 百万级基础设施 | 本地热切换 |

## 架构

```
hier-brain/
├── hier_brain.py      # 主入口：语义路由 + 专家切换
├── peft_runtime.py    # PEFT 运行时
├── serve.py           # 推理服务
└── requirements.txt   # 依赖
```

## 运行

```bash
python serve.py --config config.yaml
```

## 依赖

```bash
pip install -r requirements.txt
```
