# PI05 Subtask 路径改造策略（历史总结 + 执行方案）

## 1. 历史回顾（按问题演进）

### 阶段 A：目标与边界
- 目标是把 subtask 语义输出迁移到 B1K baseline 的 openpi 推理闭环。
- 约束是尽量最小改动，不改训练主流程，优先打通推理返回 subtask_text。

### 阶段 B：基础链路打通
- 已完成 tuple 输出解析，推理返回中可包含 subtask_tokens / subtask_text。
- 已新增 PI05 子任务配置路径，确保可以走 PI05 子任务模型分支。

### 阶段 C：运行时错误修复
- 修过 websocket 1011 的多类触发点（包括服务端异常透传、reset 包处理、握手噪声）。
- 修过 token_ar_mask 类型问题（由 bool 改为 int），解决 jaxtyping 报错。
- 修过 norm stats 回退加载问题，避免资产路径偶发失配。

### 阶段 D：能力退化与文本退化
- 现象 1：走 pi05_subtask 路径时，机器人动作能力明显下滑（原地高频抖动）。
- 现象 2：subtask 文本输出退化为 yes/no 等极短词。
- 当前已知关键修复：subtask tokenization 恢复 state 条件输入，避免动作分支失去 proprio 信息。


## 2. 当前最可能根因（按优先级）

### P0：训练配置与评测配置不一致
- 如果训练用的是 pi05_b1k（Pi0Config pi05=True）而评测用 pi05_b1k_subtask_lora（Pi05Config + LoRA），结构与参数语义不等价。
- 加载器会做“可匹配键合并”，这会导致部分参数来自 checkpoint，部分参数保持初始化值，模型能跑但行为严重退化。

### P0：LoRA 与非 LoRA 混用
- LoRA checkpoint 必须配 LoRA 模型配置。
- 非 LoRA checkpoint 必须配非 LoRA 模型配置。
- 混用时常见表现就是语言输出极短、动作抖动、成功率骤降。

### P1：subtask tokenization 与训练模板不一致
- 子任务路径如果没有保持与训练一致的 Prompt 模板（尤其 State 段），动作质量会先崩。
- 目前已把 State 条件补回，属于必要条件，但不是充分条件。

### P2：冻结配置和训练超参数不一致
- 部分配置里 freeze_filter 的 action_horizon 与 model.action_horizon 不一致，虽然不一定直接致命，但会增加训练/评测漂移风险。


## 3. 已落地改动（确认项）

- 在 [src/openpi/transforms.py](src/openpi/transforms.py#L270) 的 TokenizeSubtaskPrompt 中，将 state 传入 subtask tokenization。
- 在 [src/openpi/models/tokenizer.py](src/openpi/models/tokenizer.py#L51) 与 [src/openpi/models/tokenizer.py](src/openpi/models/tokenizer.py#L82) 中，high-level / high+low tokenization 支持 State 前缀。
- token_ar_mask 统一为 int32，避免 Observation 类型检查失败。


## 4. 建议的修改策略（最小改动，按顺序执行）

### 步骤 1：先做“配置-权重同构”收敛（必须先做）
- 严格执行一一对应：
  - 训练配置是 pi05_b1k 的 checkpoint，只用 pi05_b1k 推理。
  - 训练配置是 pi05_b1k_subtask 的 checkpoint，只用 pi05_b1k_subtask 推理。
  - 训练配置是 pi05_b1k_subtask_lora 的 checkpoint，只用 pi05_b1k_subtask_lora 推理。
- 任何跨配置评测都视作无效对比，不进入行为结论。

### 步骤 2：固定 subtask tokenization 模板
- 保持当前 Task + State + Subtask 模板，不再频繁改 prompt 文案。
- 在同一 checkpoint 上只做 A/B：
  - A: 原路径
  - B: 当前 state-aware 子任务路径
- 只比较动作成功率和 subtask 可读性，不同时改其他模块。

### 步骤 3：建立两条基线回归
- 动作基线：同任务、同随机种子、同 checkpoint 下，比较到达率/完成率/抖动次数。
- 文本基线：统计 subtask_text 的去重词数、平均长度、空串率、yes/no 占比。

### 步骤 4：仅在必要时扩大改动面
- 若 yes/no 仍高占比，才引入训练侧对齐（high/low prompt 专用监督字段）。
- 在此之前，不改模型结构，不改采样器，不改控制器。


## 5. 验证清单（每次改动都要过）

### 功能正确性
- websocket 无 1011 类型错误。
- Observation 构造无 token_ar_mask 类型错误。
- subtask 字段可稳定返回，不出现随机缺失。

### 行为质量
- 机器人不再原地高频抖动。
- 关键任务指标恢复到 pi05_b1k 基线附近。

### 文本质量
- subtask_text 平均长度提升。
- yes/no 占比明显下降。
- 空字符串比例下降。


## 6. 风险与回滚

### 风险
- 在配置不一致时继续迭代会制造伪结论，浪费调试时间。
- 同时改多个变量（tokenizer、模型、控制）会让根因不可追踪。

### 回滚策略
- 先回到“同构配置 + 同构 checkpoint”的最后稳定组合。
- 只保留 state-aware tokenization 这一项改动做单变量验证。


## 7. 本轮推荐执行顺序（实操版）

1. 选定一个 checkpoint，并确认它来自哪一个 config 名称。
2. 使用同名 config 启动服务，不做跨配置测试。
3. 跑 5 个固定任务，记录动作成功率和 yes/no 占比。
4. 若动作恢复但文本仍退化，再进入训练侧 high/low 监督对齐。


## 8. 结论

- 你的判断是成立的：训练用 pi05_b1k、评测用 pi05_subtask_lora 很可能直接导致能力大幅下滑和 yes/no 退化。
- 先把“配置-权重同构”作为硬门槛，再做子任务语义改造，成功率会显著提高。