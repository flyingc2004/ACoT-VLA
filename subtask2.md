# Subtask Generation 子任务输出实现详解（中文）

## 目标

本文详细说明在当前 B1K OpenPI 代码中，subtask 文本是如何被：
- 触发
- 生成
- 解码
- 返回给客户端

本文仅覆盖推理链路，不展开训练器通用框架细节。

适用范围：
- PI05 子任务路径
- 推理阶段输出流程


## 0. 一句话总览

当配置路由到 PI05 子任务路径后，输入会先经过子任务专用 token 化（含 State 条件），模型在采样动作前先自回归生成低层 token，再把这些 token 与动作一起返回；策略层将 token 清洗并反解码为 subtask_text，最终通过 websocket 打包发出。


## 1. 路由层：什么情况下会启用 subtask generation

子任务输出不是所有 PI05 路径都启用，而是由配置和模型类型共同决定。

关键路由代码：
- [src/openpi/training/config.py](src/openpi/training/config.py#L137)

路由行为：
1. model_type = PI05 时进入 PI05 分支。
2. 如果配置对象是 Pi05Config，则使用 TokenizeSubtaskPrompt。
3. 如果是 Pi0Config(pi05=True)，仍走常规 TokenizePrompt。

结论：
- 是否能稳定输出 subtask_text，不只取决于 model_type，还取决于具体 config 类。


## 2. 输入层：子任务 token 是怎么构造的

### 2.1 子任务 transform 入口

- 入口类：TokenizeSubtaskPrompt
- 文件位置：[src/openpi/transforms.py](src/openpi/transforms.py#L270)

主要逻辑：
1. 读取 high-level prompt。
2. 尝试读取 low-level prompt（subtask 或 low_prompt）。
3. 读取 state（重要）。
4. 有 low prompt 时走 high+low token 化；否则走 high-only token 化。

对应代码点：
- 读取 state: [src/openpi/transforms.py](src/openpi/transforms.py#L286)
- high+low 调用: [src/openpi/transforms.py](src/openpi/transforms.py#L289)
- high-only 调用: [src/openpi/transforms.py](src/openpi/transforms.py#L293)

### 2.2 tokenizer 细节

文件：
- [src/openpi/models/tokenizer.py](src/openpi/models/tokenizer.py#L51)
- [src/openpi/models/tokenizer.py](src/openpi/models/tokenizer.py#L82)

实现要点：
1. 高层文本清洗：lower、去换行、去尾部标点、补句号。
2. 若 state 存在：
   - 将 state 离散到 256 桶。
   - 拼入前缀文本中的 State 段。
3. 输出统一 padding/截断到 max_token_len。

当前输出 dtype 约定：
- tokenized_prompt: int32
- tokenized_prompt_mask: bool
- token_ar_mask: int32
- token_loss_mask: bool

这个 dtype 约定必须和 Observation 类型签名一致，否则会触发 jaxtyping 报错。


## 3. 模型层：subtask token 如何被生成

### 3.1 生成入口

PI05 模型在 sample_actions 中先调用 sample_low_level_task 生成低层 token，再做动作扩散采样。

关键代码：
- sample_actions 入口: [src/openpi/models/pi05.py](src/openpi/models/pi05.py#L356)
- 调用低层生成: [src/openpi/models/pi05.py](src/openpi/models/pi05.py#L374)

### 3.2 sample_low_level_task 解码机制

关键代码段：
- 解码循环核心: [src/openpi/models/pi05.py](src/openpi/models/pi05.py#L300)
- deembed 取 logits: [src/openpi/models/pi05.py](src/openpi/models/pi05.py#L319)
- 迭代写入 token: [src/openpi/models/pi05.py](src/openpi/models/pi05.py#L329)

机制说明：
1. 先构建 prefix（图像+语言）嵌入。
2. 每一步把已生成 token 重新 embed，拼到 prefix 后做一次前向。
3. 取最后有效位置的 embedding，deembed 到词表 logits。
4. 温度 > 0 时采样，否则 argmax。
5. 循环到 max_decoding_steps。

### 3.3 生成 token 如何参与动作采样

生成结束后会构建 prefix KV cache，供后续动作 denoise 过程复用。

代码：
- KV cache 重建: [src/openpi/models/pi05.py](src/openpi/models/pi05.py#L334)
- 作为动作采样条件输入: [src/openpi/models/pi05.py](src/openpi/models/pi05.py#L404)

最终返回：
- (actions, output_tokens)
- 返回点: [src/openpi/models/pi05.py](src/openpi/models/pi05.py#L419)


## 4. 策略层：如何把 token 变成 subtask_text

策略推理会优先判断模型输出是否为 tuple。

关键代码：
- tuple 检测: [src/openpi/policies/policy.py](src/openpi/policies/policy.py#L98)
- 提取 subtask_tokens: [src/openpi/policies/policy.py](src/openpi/policies/policy.py#L101)

token 清洗与解码流程：
1. 转 int32 并拉平。
2. 去掉 padding token（0）。
3. EOS（1）截断。
4. 过滤非法词表 id。
5. 用 PaligemmaTokenizer.detokenize 解码。

对应代码：
- 去 padding: [src/openpi/policies/policy.py](src/openpi/policies/policy.py#L133)
- EOS 截断: [src/openpi/policies/policy.py](src/openpi/policies/policy.py#L135)
- 词表过滤: [src/openpi/policies/policy.py](src/openpi/policies/policy.py#L139)
- 解码: [src/openpi/policies/policy.py](src/openpi/policies/policy.py#L141)

最终字段注入：
- subtask_tokens
- subtask_text

注入位置：
- [src/openpi/policies/policy.py](src/openpi/policies/policy.py#L155)


## 5. 传输层：如何返回给客户端

websocket server 直接把 policy.infer 的完整输出字典打包发送。

关键代码：
- 推理调用: [src/openpi/serving/websocket_policy_server.py](src/openpi/serving/websocket_policy_server.py#L61)
- msgpack 发送: [src/openpi/serving/websocket_policy_server.py](src/openpi/serving/websocket_policy_server.py#L69)

因此，只要策略输出含 subtask_text，客户端就可以收到。


## 6. 实际排查中最常见的失效模式

### 6.1 输出只有 yes/no 或极短词

高概率原因：
1. 训练 checkpoint 与评测 config 路由不一致。
2. LoRA checkpoint 与非 LoRA config 混用（或反过来）。
3. 推理 prompt 模板与训练模板不一致（尤其 State 段）。
4. token 被 max_token_len 截断。

### 6.2 动作能力明显下降（原地抖动）

高概率原因：
1. 子任务路径丢失 state 条件（已在当前实现补回）。
2. checkpoint 参数与模型结构部分错配，导致“可运行但半随机”。
3. 子任务 token 不稳定，反向影响动作条件。


## 7. 快速核对清单（建议每次上线前执行）

1. 配置是否确实路由到 TokenizeSubtaskPrompt。
2. observation 内 token_ar_mask 是否为 int32。
3. 生成 prompt 中是否包含 State 段。
4. sample_actions 日志是否显示 tuple 输出。
5. subtask token 清洗后是否仍有有效 token。
6. checkpoint 与 config 是否同构（含 LoRA 维度）。


## 8. 建议的最小调试顺序

1. 先固定一个 checkpoint + 同构 config，不做跨配置对比。
2. 只看 tuple 输出是否稳定。
3. 再看 subtask_text 长度和去重词数。
4. 最后看动作成功率和抖动指标。

这样能避免“同时改多处导致根因不可归因”。


## 9. 附：当前链路上的关键文件

- 路由与配置：
  - [src/openpi/training/config.py](src/openpi/training/config.py)
- 输入 transform：
  - [src/openpi/transforms.py](src/openpi/transforms.py)
- tokenizer：
  - [src/openpi/models/tokenizer.py](src/openpi/models/tokenizer.py)
- PI05 模型实现：
  - [src/openpi/models/pi05.py](src/openpi/models/pi05.py)
- 策略输出解析：
  - [src/openpi/policies/policy.py](src/openpi/policies/policy.py)
- websocket 服务：
  - [src/openpi/serving/websocket_policy_server.py](src/openpi/serving/websocket_policy_server.py)
