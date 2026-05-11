import json
from pathlib import Path

SESSION = Path("/mnt/a/ljz/.codex/sessions/2026/05/01/rollout-2026-05-01T10-21-14-019de30e-8d16-7e52-9ac0-214a2d097985.jsonl")
OUT = Path("./codex_migration_context.md")

def short(s, limit=6000):
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "\n...[TRUNCATED]..."

events = []
with SESSION.open("r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            obj["_line"] = i
            events.append(obj)
        except Exception:
            pass

messages = []
for e in events:
    text = None
    role = None

    # 尽量兼容不同 Codex JSONL event 格式
    if "role" in e and ("content" in e or "message" in e):
        role = e.get("role")
        text = e.get("content") or e.get("message")
    elif e.get("type") in ("user_message", "assistant_message", "message"):
        role = e.get("role") or e.get("type")
        text = e.get("content") or e.get("message") or e.get("text")
    elif "item" in e:
        item = e["item"]
        role = item.get("role") or item.get("type")
        text = item.get("content") or item.get("message") or item.get("text")

    if isinstance(text, list):
        parts = []
        for x in text:
            if isinstance(x, dict):
                parts.append(x.get("text") or x.get("content") or json.dumps(x, ensure_ascii=False))
            else:
                parts.append(str(x))
        text = "\n".join(parts)

    if role and text:
        messages.append((e["_line"], role, str(text)))

# 最近 80 条消息，通常足够迁移上下文
recent = messages[-80:]

# 抽取可能有用的关键词行
keywords = [
    "TODO", "todo", "待办", "下一步", "问题", "报错", "error", "warning",
    "修改", "实现", "fix", "bug", "ACoT", "VLA", "openpi", "harness",
]
important = []
for line, role, text in messages:
    if any(k in text for k in keywords):
        important.append((line, role, text))
important = important[-60:]

with OUT.open("w", encoding="utf-8") as w:
    w.write("# Codex 旧会话迁移上下文\n\n")
    w.write("下面内容来自旧 Codex session，用于粘贴到新会话继续工作。\n\n")

    w.write("## 1. 最近有效对话\n\n")
    for line, role, text in recent:
        w.write(f"### line {line} | {role}\n\n")
        w.write(short(text, 4000))
        w.write("\n\n")

    w.write("## 2. 可能重要的历史片段\n\n")
    for line, role, text in important:
        w.write(f"### line {line} | {role}\n\n")
        w.write(short(text, 2500))
        w.write("\n\n")

print(f"完成：{OUT}")
print(f"总事件数: {len(events)}")
print(f"抽取消息数: {len(messages)}")
print(f"输出文件: {OUT.resolve()}")
