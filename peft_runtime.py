#!/usr/bin/env python3
# peft_runtime.py — 本地 PEFT 运行时 v3：多基座 + 共享基座多适配器 + 热切换
#   0.6B 基座 → 丘脑/海马体（常驻前台）
#   8B  基座 → experts兜底；9B（Qwen3.5 纯文本）→ math35 主力
#   热切换：基座常驻，按角色 set_adapter 换 LoRA（=MoE 专家切换）
#   register_role() 支持运行时热挂载新学科 LoRA
# 绕开 ollama 注册/WSL 依赖。格式与训练一致: "user: ...\nassistant:"

import gc, os, threading, torch, warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

QWEN_DIR = os.environ.get("QWEN_DIR", "./models")
BASES = {
    "0.6b": os.path.join(QWEN_DIR, "Qwen3-0.6B"),
    "8b":   os.path.join(QWEN_DIR, "Qwen3-8B"),
    "9b":   os.path.join(QWEN_DIR, "Qwen3.5-9B"),   # Qwen3.5（纯文本解码器）
}
ROLE_BASE = {
    "thalamus": "0.6b", "hippocampus": "0.6b",
    "math": "8b", "code": "8b", "general": "8b", "prefrontal": "8b",
    "math35": "9b",
}
FIRST_ADAPTER = {"0.6b": "thalamus", "8b": "math", "9b": "math35"}
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters")
_LOCK = threading.Lock()  # 模型不线程安全，生成串行化

_runtime = {}  # base_key -> {"model","tok","roles":[]}
_BIG_BASES = ("8b", "9b")  # 大基座最多留一个（12GB 物理墙），0.6B 常驻

def _evict_big(exclude):
    """12GB 保底：加载新大基座前驱逐其他大基座，只留 0.6B 常驻 + 一个大基座。"""
    for bk in list(_runtime):
        if bk in _BIG_BASES and bk != exclude:
            del _runtime[bk]
            gc.collect()  # transformers 模型有循环引用，必须显式 GC 才真释放
            torch.cuda.empty_cache()
            print(f"[peft] evicted big base {bk} (VRAM 保底)")

def _get_runtime(role):
    bk = ROLE_BASE[role]
    rt = _runtime.get(bk)
    if rt is None:
        if bk in _BIG_BASES:
            _evict_big(bk)
        tok = AutoTokenizer.from_pretrained(BASES[bk])
        if bk == "0.6b":  # 0.6B 直接 bf16
            base = AutoModelForCausalLM.from_pretrained(
                BASES[bk], dtype=torch.bfloat16, device_map="auto")
        else:             # 8B / 9B 用 4-bit 推理
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            if bk == "9b":  # Qwen3.5 VL 检查点 → 纯文本 Qwen3_5ForCausalLM（跳过视觉塔）
                from transformers import Qwen3_5ForCausalLM
                cls = Qwen3_5ForCausalLM
            else:
                cls = AutoModelForCausalLM
            base = cls.from_pretrained(BASES[bk], quantization_config=bnb, device_map="auto")
        first = FIRST_ADAPTER[bk]
        model = PeftModel.from_pretrained(
            base, os.path.join(ADAPTER_DIR, first), adapter_name=first)
        rt = {"model": model, "tok": tok, "roles": [first]}
        _runtime[bk] = rt
        print(f"[peft] base {bk} ready (adapter: {first})")
    if role not in rt["roles"]:
        rt["model"].load_adapter(os.path.join(ADAPTER_DIR, role), adapter_name=role)
        rt["roles"].append(role)
        print(f"[peft] adapter loaded: {role}")
    return rt

def generate(role, prompt, max_tokens=120):
    rt = _get_runtime(role)
    with _LOCK:  # 热切换 + 生成串行化（模型非线程安全）
        rt["model"].set_adapter(role)
        text = f"user: {prompt}\nassistant:"
        tok = rt["tok"]
        enc = tok(text, return_tensors="pt").to(rt["model"].device)
        with torch.no_grad():
            out = rt["model"].generate(
                **enc, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.3)
        resp = tok.decode(out[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
    # 截断：答案后模型可能继续演对话（用户/助理/emoji 续写），只留第一段
    for marker in ("\n用户", "\nuser:", "\nassistant:", "\n助理", "用户：", "user：",
                   "assistant：", "Assistant:", "🎯"):
        idx = resp.find(marker)
        if idx > 0:
            resp = resp[:idx]
    return resp.strip()

def register_role(role, base_key, adapter_dir=None):
    """运行时热挂载新 LoRA：role 加入路由表并立即加载（学科热扩展）。"""
    if base_key not in BASES:
        raise ValueError(f"未知基座: {base_key}，可用: {list(BASES)}")
    ROLE_BASE[role] = base_key
    _get_runtime(role)
    return {"role": role, "base": base_key, "status": "loaded"}
