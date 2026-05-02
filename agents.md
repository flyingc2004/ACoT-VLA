# ACoT-VLA Subtask Generation Reference Notes

## Current Reference

以本仓库内的 `openpi/` 子仓库为准，且必须切到 `my_behavior` 分支再看实现。

当前 `my_behavior` 上的关键提交是：

- `6a0b10b feat(subtask generation): add hierarchical high/low-level task processing pipeline`
  - 推理侧 subtask generation 链路。
  - 增加 high/low prompt tokenizer、stage-1 high-level transform、`Pi0.sample_low_level_task()`、Gemma `deembed` 和 KV cache 更新。
- `7d95aa4 feat(training): add two-stage joint training for subtask generation and action diffusion`
  - 训练侧 two-stage joint training 链路。
  - 增加 `(obs_stage1, obs_stage2)` dataloader、CE loss 接入、`scripts/train_val.py` 的 two-stage train/val step。
- `432e277 change log path`
  - 只调整日志路径，不是 subtask 主链路。

之前提到的 `openpi_subtask_generation/` 不再作为主要参考。后续移植 ACoT-VLA 时应优先对齐：

- `openpi/src/openpi/models/tokenizer.py`
- `openpi/src/openpi/transforms.py`
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/gemma.py`
- `openpi/src/openpi/policies/policy.py`
- `openpi/src/openpi/training/config.py`
- `openpi/src/openpi/training/data_loader.py`
- `openpi/scripts/train_val.py`

## What my_behavior Implements

### 1. Tokenizer

文件：`openpi/src/openpi/models/tokenizer.py`

新增到 `PaligemmaTokenizer`：

- `tokenize_high_low_prompt(high_prompt, low_prompt)`
- `tokenize_high_low_prompt_with_state(high_prompt, low_prompt, state)`
- `detokenize(tokens)`

核心模板：

- no-state: `Task: {high}. Subtask: {low}.`
- with-state: `Task: {high}, State: {state}; Subtask: {low}.`

输出包括：

- `tokenized_prompt`
- `tokenized_prompt_mask`
- `token_ar_mask`
- `token_loss_mask`

`token_loss_mask` 只覆盖 low-level subtask token，用于 stage-1 CE loss。

### 2. Transform / Config Routing

文件：

- `openpi/src/openpi/transforms.py`
- `openpi/src/openpi/training/config.py`

`transforms.Group` 新增 `high_level_inputs`，用于 stage-1 subtask generation pipeline。

`TokenizeHighLowPrompt` 消费：

- `prompt`: high-level task
- `subtask`: low-level teacher text
- optional `state`

在 `ModelTransformFactory` 的 `ModelType.PI05` 分支中：

- `inputs`: 保持原 action diffusion 路径，使用 `TokenizePrompt`
- `high_level_inputs`: 使用 `TokenizeHighLowPrompt`

因此训练时同一个原始 sample 会被拆成两套 observation：

- `obs_stage1`: high-level prompt + low-level subtask label，用于 CE
- `obs_stage2`: 原动作 diffusion 输入，用于 action loss

### 3. Model Inference

文件：

- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/gemma.py`

`Pi0` 新增：

- `embed_high_level_prefix()`
- `_compute_CE_loss()`
- `sample_low_level_task()`

推理侧 `sample_low_level_task()` 的流程：

1. 用 image + tokenized high-level prompt 做 prefix prefill。
2. 通过 `PaliGemma.llm(..., method="deembed")` 得到 next-token logits。
3. 用 `jax.lax.while_loop` 自回归 decode low-level subtask tokens。
4. 返回 `output_tokens, kv_cache, mask, ar_mask`。

`gemma.py` 为这条链路增加：

- `Embedder.encode()` 改为 `jnp.take`，支持 JAX trace 下的 token lookup。
- `Module.deembed()`
- attention KV cache 初始化和单 token 更新。

### 4. Policy Inference

文件：`openpi/src/openpi/policies/policy.py`

`Policy.__init__()` 新增：

- `high_level_transforms`
- `jit_sample_low_level_task`

`Policy.infer()` 的实际链路：

1. 复制原始 obs。
2. 塞入 placeholder `subtask = "ABCDEFG"`，让 `TokenizeHighLowPrompt` 能构造 high/low 格式。
3. 根据 `token_loss_mask` 把 placeholder low-level token 置零并 mask 掉。
4. 调用 `sample_low_level_task()` decode predicted low-level tokens。
5. 用 `PaligemmaTokenizer.detokenize()` 得到 `predicted_texts`。
6. 当前实现只 `logging.info` 和 `print` 这些文本。
7. Stage 2 仍使用原 action diffusion path 产生 `actions`。

重要：`my_behavior` 当前推理侧已经生成了 subtask text，但没有把 `subtask_text` 写回 `outputs` dict。也就是说 websocket client 默认收不到 `subtask_text`，只能在 server log 中看到。ACoT-VLA 如果目标是“推理侧输出 subtask”，移植时必须额外加：

```python
outputs["subtask_tokens"] = predicted_token_np[0]
outputs["subtask_text"] = predicted_texts[0]
```

### 5. Training

文件：

- `openpi/src/openpi/training/data_loader.py`
- `openpi/scripts/train_val.py`
- `openpi/src/openpi/models/pi0_config.py`

`Pi0Config` 新增：

- `ce_loss_weight: float = 0.1`

`TwoStageTransformedDataset` 会对同一个 sample 分别应用：

- stage 2 pipeline: repack -> data transforms -> normalize -> `model_transforms.inputs`
- stage 1 pipeline: repack -> data transforms -> normalize -> `model_transforms.high_level_inputs`

`DataLoaderImpl.__iter__()` 在 two-stage 模式下 yield：

```python
((obs_stage1, obs_stage2), actions)
```

`scripts/train_val.py` 检测到 tuple observation 后调用：

```python
model.compute_loss(rng, obs_stage2, actions, train=True, obs_stage1=obs_stage1)
```

`Pi0.compute_loss()` 内部：

- 先计算 action diffusion loss。
- 如果传入 `obs_stage1`，再计算 subtask CE loss。
- 返回 `mean(diffusion_loss) + ce_loss_weight * ce_loss`。

## ACoT-VLA Porting Plan

### A. Tokenizer / Transform

在当前 ACoT-VLA 主工程中移植：

- `PaligemmaTokenizer.tokenize_high_low_prompt*`
- `PaligemmaTokenizer.detokenize`
- `transforms.Group.high_level_inputs`
- `TokenizeHighLowPrompt`
- `ModelTransformFactory` 中 PI05 / ACoT config 的 `high_level_inputs` 路由

如果 B1K 使用 `prompt` + `subtask` 字段，repack 必须保留两者。

### B. Model Stage 1

在 `ACOT_VLA` 中移植：

- high-level prefix embedding
- token CE loss
- autoregressive low-level token decode
- Gemma `deembed` / KV cache 支持

注意 ACoT-VLA 是三流结构 `[paligemma, coarse_action_expert, action_expert]`，stage-1 subtask generation 应只走 PaliGemma language stream，不要直接污染 coarse/action expert 的动作路径。

### C. Policy Output

推理侧必须补上 websocket-visible 字段：

- `subtask_tokens`
- `subtask_text`

`openpi/my_behavior` 目前只打印 predicted text，不返回给 client。ACoT-VLA 目标是输出 subtask，所以这一步不能省。

### D. Training Stage

移植 two-stage batch 结构：

- `TwoStageTransformedDataset`
- `transform_dataset()` 的 high-level branch
- dataloader yield `((obs_stage1, obs_stage2), actions)`
- train step 调用 `compute_loss(..., obs_stage1=obs_stage1)`

训练目标：

```text
total_loss = action_diffusion_loss + ce_loss_weight * subtask_ce_loss
```

### E. Optional Action Conditioning

`my_behavior` 目前 stage 2 action diffusion 暂时不使用生成的 subtask。ACoT-VLA 第一版也建议保持这个设计：

- 先验证 subtask 文本可生成、可训练、可通过 websocket 输出。
- 等 subtask 稳定后，再考虑把 generated subtask KV cache 接入动作 denoise prefix。

## Validation Harness

### H0 Branch / Reference

- `cd openpi`
- `git branch --show-current` 应为 `my_behavior`
- `git log --oneline --max-count=3` 应看到 `7d95aa4`、`432e277`、`6a0b10b`

### H1 Transform

输入 fixture：

```python
{
    "prompt": "turn on the radio",
    "subtask": "move the hand to the radio power button",
    "state": np.zeros([32], dtype=np.float32),
    "image": {...},
}
```

断言：

- `tokenized_prompt.shape == (max_token_len,)`
- `tokenized_prompt_mask.dtype == bool`
- `token_loss_mask.sum() > 0`
- loss mask 只覆盖 low-level subtask token

### H2 Model Decode

断言：

- `sample_low_level_task()` 返回 token shape `[batch, max_decoding_steps]`
- `detokenize()` 能把 tokens 转成字符串
- JIT 下不报 shape / KV cache 错误

### H3 Policy Response

ACoT-VLA 移植后必须断言：

- `policy.infer()` 返回 `actions`
- `policy.infer()` 返回 `subtask_tokens`
- `policy.infer()` 返回 `subtask_text`

这是 `my_behavior` 参考实现和 ACoT-VLA 目标之间的一个明确差异。

### H4 Training Microbatch

断言：

- dataloader yield `((obs_stage1, obs_stage2), actions)`
- `obs_stage1.token_loss_mask is not None`
- `model.compute_loss(..., obs_stage1=obs_stage1)` finite
- gradients finite

### H5 Legacy Regression

断言：

- 不带 `high_level_inputs` 的旧 config 仍走 single-stage dataloader。
- 不传 `obs_stage1` 时 `compute_loss()` 返回旧 action diffusion loss。
- 旧 policy action inference 不因 subtask branch 改动而断。

## Known Implementation Caveats

这些是检查 `openpi/my_behavior` 时看到的细节，移植 ACoT-VLA 时建议顺手处理：

- `Policy.infer()` 当前只打印 `predicted_texts`，没有返回 `subtask_text`。
- `tokenize_high_low_prompt()` 的 no-state 截断分支需要同步截断 `ar_mask` 和 `loss_mask`，否则 prompt 超长时 mask 长度可能和 tokens 不一致。
- `sample_low_level_task()` 的 `output_tokens` 默认是 float array，建议显式设为 `jnp.int32`。
- `sample_low_level_task()` 类型标注写的是 `-> str`，实际返回 tuple。
- `embed_high_level_prefix()` 当前没有使用 `observation.token_ar_mask`，而是把语言 token 全部设为 causal mask。移植时要么保持一致，要么显式改成使用 tokenizer 产出的 `token_ar_mask`。
- Gemma KV cache update 用 `idx[0]`，更适合 batch size 1 的推理；如果要 batch decode，需要重新检查不同 prefix length 的情况。
- `Policy.__init__()` 无条件构造 `jit_sample_low_level_task`，可能影响 PyTorch policy path。ACoT-VLA 若只用 JAX 可先不处理。
- 生成 subtask 本身不需要 GenieSim；GenieSim 只用于闭环环境评测。

## Definition Of Done For ACoT-VLA

最小可用版本应满足：

- server-side `policy.infer()` 返回 `actions`、`subtask_tokens`、`subtask_text`。
- websocket client 能收到 `subtask_text`。
- two-stage dataloader 能提供 `obs_stage1` 和 `obs_stage2`。
- `subtask_ce_loss` 和 action diffusion loss 能一起反传。
- 旧 ACoT-VLA config 在不开 subtask 分支时行为不变。
- 不依赖 GenieSim 也能完成 H1-H4；闭环成功率评测再安装 GenieSim。
