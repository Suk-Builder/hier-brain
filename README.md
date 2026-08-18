# hier-brain: Hierarchical Role-LoRA Training Pipeline

A consumer-grade implementation of Mixture-of-LoRA (MoL) that replaces implicit token-level routing with explicit semantic-domain routing via a dedicated thalamus module.

## Motivation

Industry MoE/MoL implementations (e.g., Macaron-V1 749B) target datacenter deployment: 749B base models, million-LoRA infrastructure, A100/H100 clusters. This leaves a hardware gap at the consumer tier.

hier-brain occupies the opposite end of the spectrum: **12 GB VRAM (RTX 3060) runs the full stack**.

## Core Design

Instead of training a single 284B monolithic model, hier-brain trains:
- **One 8B base model** (4-bit quantized ≈ 4.5 GB)
- **Multiple domain-specific LoRA adapters** (tens of MB each)
- **A 0.6B thalamus router** for explicit semantic gating
- **A 0.6B hippocampus module** for memory encoding and consolidation

At inference time, the base model remains resident while LoRA adapters are hot-swapped based on thalamus routing decisions. This is semantically equivalent to MoE but with explicit, interpretable routing rather than implicit token-level gating.

## Architecture

```
Input
  └── Thalamus (0.6B, QLoRA)
        ├── Decision: domain + memory-lookup flag
        ├── Expert Execution (8B base + domain LoRA)
        └── Hippocampus (0.6B, QLoRA)
              ├── Pattern separation
              ├── Consolidation
              └── Index retrieval
                    └── Prefrontal Integration (8B–70B)
                          └── Output
```

| Module | Parameters | Training | Function |
|--------|-----------|----------|----------|
| Thalamus | 0.6B | QLoRA | Semantic routing, domain classification |
| Hippocampus | 0.6B | QLoRA | Memory encoding, consolidation, pattern separation |
| Expert Pool | 8B base + LoRA | QLoRA per domain | Domain computation |
| Prefrontal | 8B–70B | QLoRA / placeholder | Cross-module integration |

## Training Pipeline (RTX 3060 12GB)

```bash
# 1. Generate training data
python gen_training_data.py --role math      # → data/math.jsonl
python gen_hippo_v2.py                     # → data/hippocampus.jsonl
python gen_experts.py --role code          # → data/code.jsonl

# 2. QLoRA fine-tuning
python train_lora.py --base <HF_MODEL> --role math   --data data/math.jsonl --epochs 3 [--ffn-only]
# → adapters/math/ (adapter_model.safetensors + config.json)

# 3a. Local inference (dual-base, multi-adapter, no ollama dependency)
python peft_runtime.py --base <HF_MODEL> --adapters adapters/

# 3b. Ollama deployment
# Convert adapters to GGUF, register in Modelfile, serve via ollama
```

## License

MIT
