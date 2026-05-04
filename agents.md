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

## Challenge Branch Merge Strategy

### Current Branch Facts

当前仓库工作基线：

- 当前分支：`subtask`
- subtask HEAD：`930f872 feat: subtask generation`
- challenge 分支：`origin/challenge`
- challenge HEAD：`0775b0c feat:final challenging`
- `git merge-base subtask origin/challenge` 没有结果，说明两边是 unrelated histories。

因此不要执行：

```bash
git merge --allow-unrelated-histories origin/challenge
```

也不要把 `origin/challenge` 的核心文件整文件覆盖到 `subtask`。两棵树直接 diff 约为：

```text
1027 files changed, 7520 insertions(+), 244700 deletions(-)
```

这个巨大差异主要来自 `challenge` 删除/替换了 `lerobot/`、`third_party/libero` 等依赖树，并不是实际业务逻辑都需要合进来。最佳手段是：

```text
以 subtask 为唯一主线，创建集成分支；把 challenge 当成供体树，按功能手动移植。
```

建议集成分支：

```bash
git switch subtask
git switch -c integrate/challenge-on-subtask
```

### Non-Negotiable Goal

合并后必须同时保留两条能力：

- 当前 `subtask` 分支已经实现的 stage-1 subtask semantic output。
- `challenge` 分支为了推理仿真环境引入的多任务、checkpoint routing、sorting continuous、训练数据构建和权重加载修复。

如果二者冲突，优先保留 `subtask` 的语义输出链路，再把 `challenge` 的功能以局部 patch 接进去。

### Do Not Take From Challenge As-Is

这些来自 `challenge` 的变化不要直接照收：

- 删除 `agents.md`。
- 删除或替换 `lerobot/`、`third_party/libero` 的整棵目录。
- 新增 `.gitmodules` 并把 `third_party/aloha`、`third_party/libero` 改成 submodule，除非明确决定重构依赖管理。
- 把 `pyproject.toml` 的 `lerobot = { path = "lerobot" }` 改成远端 git dependency，除非本仓库不再 vendor `lerobot/`。
- 提交 `outputs/sorting_phase_classifier*.pt`、`training_curves.png`、`training_metrics.json` 等训练产物。
- 提交调试图片 `top_head.png`。
- 直接采用硬编码绝对路径，例如 `/home/xhz/...`、`/mnt/sdc/...`、`/data/...`，必须改成 env var 或文档化默认值。
- 直接采用 `scripts/train.sh` 的本机 GPU / wandb 设置。

`.gitignore` 中唯一可以直接吸收的小改动是：

```text
sorting_phase_dataset/
```

但要保留当前 `.gitignore` 中的 local clone / Codex 忽略项：

```text
/openpi/
/openpi_subtask_generation/
.codex
```

### Protected Subtask Files

以下文件是当前 `subtask` 主链路的保护区，不能从 `challenge` 整文件覆盖：

- `src/openpi/models/tokenizer.py`
- `src/openpi/transforms.py`
- `src/openpi/models/acot_vla.py`
- `src/openpi/models/gemma.py`
- `src/openpi/policies/policy.py`
- `src/openpi/training/config.py`
- `src/openpi/training/data_loader.py`
- `scripts/train.py`

原因：`challenge` 在这些文件里会删除或回退当前 subtask generation：

- 删除 `PaligemmaTokenizer.tokenize_high_low_prompt*()` 和 `detokenize()`。
- 删除 `transforms.Group.high_level_inputs`。
- 删除 `TokenizeHighLowPrompt`。
- 删除 `ACOTConfig.enable_subtask_generation`、`subtask_max_decoding_steps`、`subtask_temperature`、`subtask_ce_loss_weight`、`subtask_use_state_input`。
- 删除 `ACOT_VLA.embed_high_level_prefix()`、`_compute_subtask_ce_loss()`、`sample_low_level_task()`。
- 删除 `Policy` 的 high-level transform / low-level decode / `subtask_text` 输出。
- 删除 `TwoStageTransformedDataset` 和 `DataLoaderACOTImpl` 的 two-stage yield。
- 删除 `scripts/train.py` 里对 `(obs_stage1, obs_stage2)` 的训练分支。
- 在 `gemma.py` 中把 `Embedder.encode()` 从 `jnp.take(...)` 改回数组索引，并删除 `Module.deembed()`，这会破坏 JAX trace 下的 subtask decode。

### Safe-To-Import New Files

这些是 `challenge` 中新增且相对独立的功能文件，适合先移入集成分支：

- `src/openpi/policies/checkpoint_switcher.py`
- `src/openpi/policies/routing_policy.py`
- `src/openpi/policies/sorting_phase_state_machine.py`
- `scripts/build_data.sh`
- `scripts/build_sorting_phase_dataset.py`
- `scripts/make_sorting_continuous_dataset.py`
- `scripts/plot_sorting_curves.py`
- `scripts/train_sorting_phase_classifier.py`
- `scripts/server_5_tasks.sh`
- `scripts/server_multiple_task.sh`
- `yrm/checkpoint_routing.example.json`
- `yrm/train.py`
- `yrm/train.sh`
- `yrm/train_5tasks.py`
- `yrm/train_5tasks.sh`
- `yrm/train_four_tasks.py`
- `yrm/train_four_tasks.sh`
- `yrm/train_original_style.py`
- `yrm/train_original_style.sh`

导入方式：

```bash
git restore --source=origin/challenge -- \
  src/openpi/policies/checkpoint_switcher.py \
  src/openpi/policies/routing_policy.py \
  src/openpi/policies/sorting_phase_state_machine.py
```

然后手动 review，不要无脑批量 restore 所有新增文件。尤其 `yrm/ckpt_routing.json` 和 `yrm/checkpoint_routing_five_vs_baseline.json` 含本机 checkpoint 绝对路径，应优先保留 example 文件，把真实路径改成用户本地不入库配置。

### Manual Port Map

#### 1. Serving / Checkpoint Routing

目标：让 `scripts/serve_policy.py` 支持 `--checkpoint_routing`，但不破坏现有默认 server path。

从 `challenge` 移植：

- `CheckpointRoutingSwitcher`
- `RoutingPolicy`
- `Args.checkpoint_routing`
- `Args.routing_preload_default`
- `Args.routing_strict_load`
- `EnvMode.G2SIM`
- `DEFAULT_CHECKPOINT[EnvMode.G2SIM]`

需要改造：

- `DEFAULT_CHECKPOINT[EnvMode.G2SIM].dir` 不要写死 `/mnt/sdc/...`，改成相对路径、env var 或文档示例。
- `logging.basicConfig(filename="/home/xhz/logging/icra/...")` 不要照搬。日志路径应沿用当前仓库习惯，或使用 `LOG_DIR` 环境变量。
- `RoutingPolicy.infer()` 只是透传底层 policy，因此底层 `Policy` 的 `subtask_text` 输出会自然随 response 返回；这个行为要在验证里明确检查。

#### 2. Policy Runtime Prompt Logic

目标：吸收 challenge 的仿真任务 prompt mapping、sorting continuous prompt controller、可选 noise 输入，同时保留 subtask decode。

从 `challenge` 移植到 `src/openpi/policies/policy.py`：

- `SortingContinuousPromptController.from_env()` 初始化。
- sorting continuous 的 prompt 自动更新逻辑。
- `prompt_mapping` 中对 `pour_workpiece`、`open_door`、`scoop_popcorn`、`hold_pot`、`place_block_into_box`、`take_wrong_item_shelf`、`stock_and_straighten_shelf`、`clean_the_desktop` 的 prompt 注入。
- `infer(self, obs, *, noise=None)` 对 `sample_actions(..., noise=...)` 的兼容透传。

必须保留：

- `high_level_transforms`
- `_sample_low_level_task`
- `_subtask_detokenizer`
- `subtask_tokens`
- `subtask_text`

需要谨慎处理：

- `challenge` 每次 inference 都保存 `top_head.png`，这只适合 debug。默认不要启用；若需要，放到 env flag 之后。
- prompt mapping 发生在 stage-1 subtask decode 之前，否则 subtask generation 看到的还是旧 prompt。
- sorting controller 修改后的 prompt 也必须同时进入 high-level subtask transform 和 action transform。

推荐顺序：

```text
obs copy
-> sorting controller / task prompt mapping
-> stage-1 subtask decode
-> stage-2 action sample
-> output_transforms
-> attach subtask_tokens/subtask_text
-> post_process
```

#### 3. Model Noise Injection

目标：支持 `sample_actions(..., noise=...)`，为 temporal ensemble、warm-start noise、receding horizon 实验留接口。

从 `challenge` 移植：

- `src/openpi/models/pi0.py` 的 `sample_actions(..., noise=None)` 参数和 shape 校验。
- `src/openpi/models/acot_vla.py` 的 `sample_actions(..., noise=None)` 参数和 shape 校验。
- `Policy.infer(..., noise=None)` 的 sample kwargs 透传。

必须保留：

- `ACOT_VLA.sample_low_level_task()`
- `ACOT_VLA._compute_subtask_ce_loss()`
- `ACOT_VLA.compute_loss(..., obs_stage1=None, subtask_ce_loss_weight=None)`
- `gemma.Module.deembed()`

不要把 `challenge` 的 `ACOT_VLA.compute_loss()` 整段拿来覆盖，因为它会删掉 subtask CE loss。

#### 4. Training Data / Sampler

目标：吸收 challenge 对 ICRA 仿真数据的容错、segment instruction、continuous sorting 采样支持，同时保留 two-stage dataloader。

从 `challenge` 移植：

- `DataConfig.subtask_reset_truncation_mode`
- `MultiLeRobotDataset(..., tolerances_s={...})`
- `_find_segment_instruction()`
- `_join_episode_instructions()`
- `SegmentInstructionFromHighlevelInstruction`
- `PromptFromHighlevelInstruction` 对 segment 缺失的更稳健处理。
- `FrameSampler(..., reset_truncation_mode=...)` 的配置入口。

必须保留：

- `try/except ModuleNotFoundError` 的 `lerobot` import 兼容。
- `RepackTransform` 对 `subtask` 字段的保留逻辑。
- `TwoStageTransformedDataset`
- `transform_dataset()` 的 high-level branch。
- `DataLoaderACOTImpl.__iter__()` 对 tuple batch 的处理。

需要修正 challenge 里的一个半成品点：

- `FrameSampler.reset_truncation_mode` 在 `challenge` 中保存了但没有真正传入 `sample_subtask()` 使用。移植时应把 `sample_subtask(dataset, reset_truncation_mode)` 接起来。
- `auto` 模式建议：检测到 sorting/continuous task 时禁用 reset truncate；普通 subtask sampler 保持旧的 reset truncate。
- `always` 模式：保持旧行为，reset-like interval 超过阈值就截断。
- `never` 模式：完全不截断 reset-like interval。

#### 5. Training Script

目标：吸收 challenge 对训练入口的局部修正，但保留 two-stage subtask training。

可以参考：

- `scripts/train.py` 中 ACOT train step 的类型标注。
- 首 batch wandb 图像记录可改得更稳健，但要兼容 tuple observation。

必须保留：

- `acot_train_step()` 中 `isinstance(observation, tuple)` 的分支。
- `model.compute_loss(rng, obs_stage2, actions, coarse_actions, train=True, obs_stage1=obs_stage1)`。
- wandb image logging 在 tuple batch 时使用 `obs_stage2`。

不要照搬：

- `challenge` 删除 tuple observation 的版本。
- `scripts/train.sh` 的固定 config、固定 `CUDA_VISIBLE_DEVICES`、固定 `WANDB_MODE=online`。

#### 6. Weight Loading

目标：吸收 checkpoint key normalize 修复。

从 `challenge` 移植到 `src/openpi/training/weight_loaders.py`：

- `_normalize_flat_key_tuple()`
- `_merge_params()` 开头调用 `_model.convert_str_keys_to_int(loaded_params)`
- clone missing param 时用 normalize 后的 key 查 `flat_loaded`

这是低风险且有价值的修复，适合早合。

#### 7. ICRA / G2SIM Config

目标：把 challenge 的多任务 ICRA config 信息迁移到当前 `acot_icra_simulation_challenge_reasoning_to_action`，但不要写死个人机器路径。

从 `challenge` 参考：

- 任务列表扩展到 pour/open/scoop/hold/stock/place/sorting/clean 等多任务数据。
- `prompt_map_inject_to_training` 的新任务名和 prompt。
- `assets=AssetsConfig(assets_dir=..., asset_id=".")` 的设计意图：从指定 assets 目录直接加载 norm stats。
- `base_config=DataConfig(dataloader_sampler="subtask", prompt_from_task=True)`。
- `joint_action_shifts=(2, 1)`。
- `extra_delta_transform=(True, True)`。
- `delta_action_mask=_transforms.make_bool_mask(14, -18)`。
- LoRA config：`paligemma_variant="gemma_2b_lora"`。
- `warmup_steps=0`、`batch_size=64`、`num_workers=32` 等实验参数可作为 challenge preset，而不是唯一默认。

需要本地化：

- 数据路径用环境变量或注释示例，不要硬编码 `/data/Dataset/...`。
- baseline checkpoint path 用环境变量，例如 `BASELINE_PARAMS`、`BASELINE_NORM_ASSETS_DIR`。
- 可以新增一个独立 config，例如 `acot_icra_simulation_challenge_reasoning_to_action_challenge_runtime`，避免覆盖当前能工作的本地 5-task config。

#### 8. Norm Stats

目标：吸收 challenge 让 norm stats 写入 config assets 目录的思路。

从 `challenge` 参考：

- `scripts/compute_norm_stats.py` 根据 `config.data.assets.asset_id` 写到 `config.assets_dirs / asset_id`。

必须保留或补强：

- 当前脚本中 `valid_batches == 0` 时抛错的 guard 不要删除。
- `max_batches = max(1, int(num_batches * sample_ratio))` 不要退回可能为 0 的版本。
- 如果 `asset_id == "."`，确保 output path 是 `config.assets_dirs` 本身，而不是拼出奇怪路径。

### Recommended Merge Order

#### Phase 0: Branch and Baseline

```bash
git switch subtask
git switch -c integrate/challenge-on-subtask
git status --short --branch
```

记录当前 subtask baseline：

```bash
git log --oneline --max-count=3
python -m compileall src scripts
```

#### Phase 1: Import Independent Files

先导入独立新增文件：

- checkpoint routing policy files
- sorting phase state machine
- dataset building / classifier scripts
- example routing JSON

提交建议：

```text
feat(challenge): add routing and sorting support files
```

#### Phase 2: Low-Risk Core Fixes

手工移植：

- `weight_loaders.py` checkpoint key normalize。
- `.gitignore` 的 `sorting_phase_dataset/`。
- `serve_policy.py` 的 `--checkpoint_routing` 参数。

提交建议：

```text
fix(training): normalize checkpoint keys during weight loading
feat(serving): add checkpoint routing entrypoint
```

#### Phase 3: Runtime Policy Integration

手工移植：

- sorting prompt controller 初始化和 step。
- task prompt mapping。
- optional `noise` passthrough。

同时保持：

- stage-1 subtask decode。
- output `subtask_tokens` / `subtask_text`。

提交建议：

```text
feat(policy): combine challenge prompt routing with subtask output
```

#### Phase 4: Model Sampling Interface

手工移植：

- `pi0.sample_actions(..., noise=None)`
- `ACOT_VLA.sample_actions(..., noise=None)`

提交建议：

```text
feat(models): support externally supplied sampling noise
```

#### Phase 5: Data Pipeline Integration

手工移植：

- `DataConfig.subtask_reset_truncation_mode`
- `tolerances_s`
- segment instruction helpers
- fixed sampler truncation mode

同时保持:

- `TwoStageTransformedDataset`
- high-level `subtask` label flow

提交建议：

```text
feat(data): add challenge dataset tolerance and sampler modes
```

#### Phase 6: Challenge Config Preset

新增或更新 challenge-specific config。推荐新增 config，避免覆盖当前可运行配置。

提交建议：

```text
feat(config): add challenge simulation training preset
```

#### Phase 7: Verification and Cleanup

清理：

- 不提交 outputs/checkpoints。
- 不提交 top_head.png。
- 不提交个人绝对路径。
- 不改依赖结构，除非专门做依赖迁移 PR。

最后再决定是否吸收 `uv.lock`。只有当 `pyproject.toml` 的依赖确实变化后，才更新 lock。

### Validation Checklist After Merge

#### Static Checks

```bash
python -m compileall src scripts
python -m compileall yrm
```

#### Git Checks

确认没有误删核心资产：

```bash
git status --short
git diff --name-status subtask...HEAD
```

重点检查不要出现：

```text
D agents.md
D lerobot/...
D third_party/libero/...
A outputs/...
A top_head.png
```

#### Subtask Checks

必须继续满足：

- `PaligemmaTokenizer.detokenize()` 存在。
- `TokenizeHighLowPrompt` 存在。
- `transforms.Group.high_level_inputs` 存在。
- `ACOTConfig.enable_subtask_generation` 存在。
- `ACOT_VLA.sample_low_level_task()` 存在。
- `Policy.infer()` 返回 `subtask_text`。
- two-stage dataloader 能 yield `((obs_stage1, obs_stage2), actions, coarse_actions)`。

#### Challenge Runtime Checks

必须新增满足：

- `scripts/serve_policy.py --checkpoint_routing yrm/checkpoint_routing.example.json` 能解析参数。
- `RoutingPolicy.infer()` 会把 obs 路由到底层 policy。
- sorting continuous controller 在没有 classifier checkpoint 时不会让普通 policy 启动失败。
- task prompt mapping 在 action sampling 和 subtask generation 前生效。
- `sample_actions(..., noise=...)` shape 正确时报通，shape 错误时报清晰错误。

#### Training Checks

必须验证：

- 单阶段旧 config 仍能创建 dataloader。
- 开启 subtask generation 的 ACOT config 仍能创建 two-stage batch。
- `acot_train_step()` 对 tuple observation 和普通 observation 都能工作。
- norm stats 在没有有效 batch 时会报错，而不是写出空 stats。

### Final Recommendation

这个合并不要追求保留 `challenge` 的 commit history。由于 unrelated histories 和大规模依赖树差异，保留历史会让冲突成本远高于收益。

最好的工程形态是：

```text
subtask branch history
  + small commits manually ported from challenge by feature
  + docs describing which challenge commit/feature each commit came from
```

如果后续需要追溯，可以在每个移植 commit message 里加来源，例如：

```text
Ported from origin/challenge commit 5126dfe (checkpoint switcher)
Ported from origin/challenge commit c09498f (sorting classifier)
Ported from origin/challenge commit ac05a3d/ab1efa2 (noise interface)
```
