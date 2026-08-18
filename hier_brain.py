#!/usr/bin/env python3
# hier_brain.py — hier-brain架构 PoC
# 信号流（用户定义）:
#   Input → 丘脑(路由) → 前额叶(boss整合) → [experts + 海马体(记忆)] → 前额叶(整合) → Output
#
# 对应生物结构:
#   丘脑      = 路由网关（门控/分发，不计算）
#   前额叶    = 整合瓶颈 boss（将来换 70B；本 PoC 用 9B 占位）
#   experts  = 专门化皮层模块（将来各 3B）
#   海马体    = 记忆/索引模块（将来 0.6B；做模式分离+索引，不背全量）
#
# 跑法:  python hier_brain.py "<query>"
# 依赖:  仅标准库 + 本地 ollama (localhost:11434)。模型名在 CONFIG 里可换。
#
# 两处待定（已按占位实现，换真模型时决策）:
#   1. 前额叶尺寸未指定。本 PoC 默认 9B 占位；目标架构中 prefrontal 为 70B 级，
#      故 prefrontal 字段将来应换大模型，此处留作配置。
#   2. 海马体按生物正确版实现=编码/巩固/索引（非裸知识库）。recall 用 sqlite + token 重叠占位，
#      真·模式分离（防灾难性干扰）是后续研究点。

import urllib.request, json, sys, sqlite3, os, re
import peft_runtime

OLLAMA = "http://localhost:11434"
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hippocampus.db")

CONFIG = {
    # 目标架构（用户 2026-08-18 定稿）：
    #   0.6B 常驻前台 = 丘脑(路由) + 海马体(记忆索引/巩固)   —— 小、便宜、常开
    #   9B 基座 + LoRA 切换 = experts + 前额叶(boss)        —— 按需加载，换帽子
    #   "peft:<role>" = 本地 Qwen3-0.6B + adapters/<role> 适配器
    #   完全体（远程 GPU）：prefrontal -> 70B 级 boss；specialists -> 各独立 3B；
    #     hippocampus -> 独立 0.6B（若要硬记忆隔离就别挂共享基座）。
    "thalamus":   "peft:thalamus",     # 0.6B 常驻：真实路由 LoRA
    "prefrontal": "qwen3.5-9b-abl",    # 9B ollama 占位（将来 peft:prefrontal 或 70B）
    "hippocampus":"peft:hippocampus",  # 0.6B 常驻：真实巩固 LoRA
    "specialists": {
        "math":    "peft:math35",     # Qwen3.5-9B 数学主力（实测全对）
        "code":    "peft:code",       # 8B 基座 + code LoRA
        "general": "peft:general",    # 8B 基座 + general LoRA
    },
}

# ---------- 海马体存储（模式分离占位：存指针+摘要，带 source 标签防覆盖）----------
def _init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS memory(
        id INTEGER PRIMARY KEY, query TEXT, answer TEXT, source TEXT, ts REAL)""")
    conn.commit(); conn.close()

def hippo_recall(query, top=3):
    """召回相关记忆。PoC 用 token 重叠打分；真·模式分离(embedding+去干扰)后替换。"""
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT query,answer,source FROM memory").fetchall()
    conn.close()
    if not rows:
        return []
    q_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for q, a, s in rows:
        t = set(re.findall(r"\w+", q.lower()))
        overlap = len(q_tokens & t) / (len(q_tokens | t) + 1e-9)
        scored.append((overlap, (q, a, s)))
    scored.sort(key=lambda x: -x[0])
    return [x[1] for x in scored[:top] if x[0] > 0]

def hippo_consolidate(query, answer, source):
    """巩固：先用 0.6B 海马体把内容压缩成索引条目，再写库（带 source 标签防覆盖）。"""
    try:
        entry = peft_runtime.generate("hippocampus", f"巩固：{answer}", max_tokens=60)
    except Exception as e:
        entry = answer  # 降级：直接存原文
    if not entry or "记忆条目" not in entry:
        entry = f"记忆条目：{source}｜{answer[:60]}"
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO memory(query,answer,source,ts) VALUES(?,?,?,?)",
                 (query, entry, source, __import__("time").time()))
    conn.commit(); conn.close()

# ---------- 模型调用：ollama 或本地 PEFT ----------
def _gen(model, prompt, system=None, num_predict=512):
    if model.startswith("peft:"):
        role = model[5:]
        return peft_runtime.generate(role, prompt, max_tokens=num_predict)
    payload = {"model": model, "prompt": prompt, "stream": False,
               "think": False,
               "options": {"num_predict": num_predict, "temperature": 0.3}}
    if system:
        payload["system"] = system
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read().decode())["response"]
    # 剥离 qwen 的 <think>...</think> 内部独白，避免污染下游调用
    resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
    return resp

def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

# ---------- 丘脑：路由（域 + 难度） ----------
# 难度梯级（用户定）：1=简单聊天(3B) 2=稍复杂(9B地板) 3=难(14B) 4=极难(32B)
TIERS = {1: "3b", 2: "9b", 3: "14b", 4: "32b"}  # 服务器映射；本地统一走 9B 地板

def thalamus(query):
    sys = ("你是丘脑路由中枢。判断领域、是否查记忆、难度等级。"
           "难度: 1=简单聊天/琐事, 2=标准知识/简单计算, 3=难/多步推理, 4=极难/研究级。"
           "严格只输出 JSON: {\"domain\":\"math|code|general\", \"use_mem\":true|false, "
           "\"difficulty\":1-4, \"reason\":\"一句\"}")
    out = _gen(CONFIG["thalamus"], query, system=sys, num_predict=200)
    j = _parse_json(out)
    domain = j.get("domain", "general")
    if domain not in CONFIG["specialists"]:
        domain = "general"
    try:
        diff = int(j.get("difficulty", 2))
    except (TypeError, ValueError):
        diff = 2
    if diff not in (1, 2, 3, 4):
        diff = 2
    return {"domain": domain, "use_mem": bool(j.get("use_mem", False)),
            "difficulty": diff, "reason": j.get("reason", "")}

# ---------- experts：专门化计算 ----------
def specialist(domain, query):
    model = CONFIG["specialists"][domain]
    sys = f"你是{domain}领域专家，简洁准确，不废话。"
    text = _gen(model, query, system=sys, num_predict=400)
    return {"domain": domain, "text": text}

# ---------- 前额叶：整合 boss ----------
def prefrontal(query, spec, mems):
    mem_txt = "\n".join(f"[记忆-{m[2]}] {m[1]}" for m in mems) or "（无相关记忆）"
    sys = "你是前额叶整合中枢。综合专家输出与相关记忆，给出最终回答。不重复、不绕。"
    prompt = (f"问题：{query}\n"
              f"专家({spec['domain']})输出：{spec['text']}\n"
              f"相关记忆：\n{mem_txt}\n\n最终回答：")
    return _gen(CONFIG["prefrontal"], prompt, system=sys, num_predict=600)

# ---------- 主循环 ----------
def run(query):
    route = thalamus(query)
    mems = hippo_recall(query) if route["use_mem"] else []
    spec = specialist(route["domain"], query)
    final = prefrontal(query, spec, mems)
    hippo_consolidate(query, final, route["domain"])
    return final

if __name__ == "__main__":
    _init_db()
    q = " ".join(sys.argv[1:]) or "1+1=?"
    print(">>", q)
    print(run(q))
