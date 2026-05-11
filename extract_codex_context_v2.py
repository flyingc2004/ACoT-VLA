import json
from pathlib import Path

P = Path("/mnt/a/ljz/.codex/disabled_sessions/rollout-2026-05-01T10-21-14-019de30e-8d16-7e52-9ac0-214a2d097985.jsonl")
OUT = Path("codex_migration_context_v2.md")

def text_from(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(text_from(i) for i in x if text_from(i))
    if isinstance(x, dict):
        parts = []
        for k in ("text", "content", "message", "output", "summary"):
            if k in x:
                t = text_from(x[k])
                if t:
                    parts.append(t)
        if parts:
            return "\n".join(parts)
        return ""
    return str(x)

def walk_find_text(obj):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("text", "content", "message", "output", "summary"):
                t = text_from(v)
                if t:
                    found.append(t)
            else:
                found.extend(walk_find_text(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(walk_find_text(v))
    return found

records = []
with P.open("r", encoding="utf-8", errors="replace") as f:
    for line_no, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
        except Exception:
            continue

        typ = obj.get("type", "")
        payload = obj.get("payload", {})

        role = ""
        if isinstance(payload, dict):
            role = payload.get("role") or payload.get("type") or payload.get("kind") or ""

        texts = walk_find_text(payload)
        text = "\n".join(t for t in texts if t).strip()

        if text:
            # 去重一点
            if len(text) > 12000:
                text = text[:12000] + "\n...[TRUNCATED]"
            records.append((line_no, typ, role, text))

# 过滤掉明显噪音
noise_keys = [
    "exec_command", "stdout", "stderr", "token", "uuid",
    "query-cache-invalidate", "thread-stream-state-changed"
]
filtered = []
for r in records:
    text_lower = r[3].lower()
    if len(r[3]) < 5:
        continue
    filtered.append(r)

recent = filtered[-120:]

with OUT.open("w", encoding="utf-8") as w:
    w.write("# Codex 旧会话迁移上下文\n\n")
    w.write("请基于下面内容继续工作。这是从旧 Codex JSONL 中提取的文本，不要尝试 replay 原 session。\n\n")

    w.write("## 最近 120 条有效文本记录\n\n")
    for line_no, typ, role, text in recent:
        w.write(f"### line {line_no} | type={typ} | role={role}\n\n")
        w.write(text)
        w.write("\n\n")

print("records:", len(records))
print("filtered:", len(filtered))
print("output:", OUT.resolve())
