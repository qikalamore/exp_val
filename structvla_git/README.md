# StructVLA

`StructVLA` 是一个两阶段的 `vision-language-action` 管线：先学习预测稀疏的 `structured frames`，再把这种规划能力迁移到动作生成，并在共享的 token 空间里完成对齐。

当前仓库主要包含：

- `StructVLA` 的 `planner` 与 `action policy` 训练代码。
- 面向 `ManipArena` 的数据预处理、`pickle` 生成、`structured frames` 提取与部署胶水代码。
- 一个内置的 `maniparena-repo/` 子项目，用来把策略封装成 `ManipArena` 平台可调用的服务。
- `LIBERO`、`SimplerEnv` 以及另一条 `Franka real-world` 流程的辅助代码。

<img src="docs/imgs/structvla.png" alt="StructVLA" height="300">

相关链接：
- Paper: https://arxiv.org/abs/2603.12553
- Project page: https://wm-planner.github.io/structvla/

## 这个仓库现在解决什么问题

从当前代码结构看，仓库里实际上并存了三条相关但不同的工作流：

1. `ManipArena` 双臂工作流：构建多视角 token episode，生成归一化后的 `pickle`，提取 `structured frames`，训练 `planner`，最后通过 `maniparena-repo/` 部署。
2. `Franka real-world` 工作流：用 `tools/real_world/teledata_process.py` 处理基于 `CSV` 的遥操作数据，生成 `real_all_norm.pkl`，再训练动作模型。
3. `LIBERO / SimplerEnv` 工作流：仓库中仍然保留并可继续使用，但这份 `README` 不作为重点展开。

最容易混淆的一点是：仓库同时包含了

- 一层以 `pickle` 和 `structured-frame manifest` 为核心的 `training pipeline`
- 一层位于 `maniparena-repo/` 下的 `deployment / evaluation wrapper`

这两层是衔接关系，但不是同一层东西。

## 环境准备

```bash
conda create -n structvla python=3.10
conda activate structvla
pip install -r requirements.txt
```

代码默认还依赖这些预训练资源：

- `Emu3` base / tokenizer 相关资源
- `Emu3` vision tokenizer
- `WORLD_MODEL_POSTTRAIN`
- 位于 `pretrain/fast/` 下的 `FAST action tokenizer`

本地资源目录约定可参考 `pretrain/README.md`。

## 仓库结构

```text
structvla/
+-- configs/                         # normalizer、model config、示例 structured-frame 输出
+-- docs/                            # 各 benchmark 的补充说明
+-- maniparena-repo/                 # ManipArena 服务与评测封装
+-- models/                          # vision tokenizer 与推理相关代码
+-- pretrain/                        # 本地 tokenizer 资源
+-- reference/                       # 外部参考代码（Emu3、RoboVLMs、LIBERO、SimplerEnv）
+-- scripts/
|   +-- planner/                     # structured planner 训练脚本
|   +-- simulator/maniparena/        # ManipArena action model 训练脚本
|   +-- real_world/                  # Franka real-world action training
|   `-- tokenizer/                   # VQ token 提取脚本
+-- tools/
|   +-- pickle_gen/                  # 构建归一化后的 pickle 数据集
|   +-- real_world/                  # 基于 CSV 的真实数据预处理
|   `-- structured_frames/           # structured-frame / keystep 提取
+-- train/                           # dataset class 与训练入口
+-- pack_codes_to_parquet.py
+-- pack_real_all_actions_instruction_to_parquet.py
`-- README.md
```

## ManipArena 全流程

这一节按照当前仓库里的真实代码路径，说明 `ManipArena` 数据是怎么被处理、训练以及部署的。

### 1. 分层理解

可以把当前 `ManipArena` 流程理解成三层：

1. `Episode storage layer`
   - 原始或预处理后的 episode
   - `action .npy`
   - `instruction.txt`
   - 多视角视觉 token `.npy`
2. `Training data layer`
   - 归一化后的 `pickle`，例如 `real_all_dualarm_norm.pkl`、`sim_all_dualarm_norm.pkl`
   - `structured-frame manifest`，例如 `triplets_manifest.csv`
3. `Serving / benchmark layer`
   - `maniparena-repo/`
   - `WebSocket server`
   - 本地 mock 检查与 `open-loop evaluation`

当前 `planner` 和 `policy` 训练读取的是 `pickle + structured-frame manifest` 这一层，而不是直接读取 `parquet`。

### 2. ManipArena episode 期望目录

对于双臂 `ManipArena` 流程，代码期望的 `processed_data` 根目录大致如下：

```text
processed_data/
+-- real_all/
|   `-- <episode_name>/
|       +-- instruction.txt
|       `-- actions/
|           +-- 0.npy
|           +-- 1.npy
|           `-- ...
+-- maniparena_dualarm_codes_160_140/          # 也可能是其他分辨率后缀
|   `-- <episode_name>/
|       +-- cam_high/
|       |   +-- 0.npy
|       |   `-- ...
|       +-- cam_left_wrist/
|       |   +-- 0.npy
|       |   `-- ...
|       `-- cam_right_wrist/
|           +-- 0.npy
|           `-- ...
`-- meta/
```

这正是 `tools/pickle_gen/pickle_generation_maniparena.py` 所期待的布局。

其中最关键的是：

- `real_all/` 保存语言与动作。
- `maniparena_dualarm_codes_*` 保存 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三个视角的视觉 token。
- 双臂数据对应的训练 `dataset class` 也是按 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 这些 key 读取的。

### 3. 按步骤走完整个 ManipArena 双臂流程

#### Step A. 准备 `real_all/` 和 token 目录

当前仓库默认你已经有下面这些产物：

- `real_all/<episode>/instruction.txt`
- `real_all/<episode>/actions/*.npy`
- `maniparena_dualarm_codes_* / <episode> / cam_high|cam_left_wrist|cam_right_wrist/*.npy`

如果是 `ManipArena simulation` 的视觉 token 提取，仓库内现成的启动脚本是：

```bash
bash scripts/tokenizer/extract_vq_emu3_maniparena.sh
```

需要注意：

- `scripts/tokenizer/extract_vq_emu3_maniparena.sh` 实际调用的是 `models/tokenizer/new_tokenizer.py`
- 当前 `process_data='maniparena'` 的路径配置是直接写在 `models/tokenizer/new_tokenizer.py` 里的
- 如果你想把这套提 token 逻辑复用到别的 `ManipArena` 数据目录，需要先改 `models/tokenizer/new_tokenizer.py` 里的路径

#### Step A.1 仓库内这两个 parquet 样本对应的实际目录结构

这次仓库里实际放入并读取到的两个样本文件是：

- `real_classify_items_as_shape_chunk-000_episode_000000.parquet`
- `real_classify_items_as_shape_chunk-000_episode_000000_action.parquet`

从这两个 `parquet` 的内容里，能够确认出以下事实：

- `episode` 名为 `real_classify_items_as_shape_chunk-000_episode_000000`
- `task` 名为 `real_classify_items_as_shape`
- `instruction` 为 `Classify by object shape`
- 视觉 `parquet` 中明确保留了 `camera` 维度，样本里可确认到 `cam_high`、`cam_right_wrist`，并且还能看到 `left_wrist` 相关字符串

因此，这两个 `parquet` 对应的原始数据目录，至少可以还原为下面这个 episode 级结构：

```text
<codes_root>/
`-- real_classify_items_as_shape_chunk-000_episode_000000/
    +-- cam_high/
    |   +-- 0.npy
    |   `-- ...
    +-- left_wrist/ or cam_left_wrist/
    |   +-- 0.npy
    |   `-- ...
    `-- cam_right_wrist/
        +-- 0.npy
        `-- ...

<real_all_root>/
`-- real_classify_items_as_shape_chunk-000_episode_000000/
    +-- instruction.txt              # 内容为: Classify by object shape
    `-- actions/
        +-- 0.npy
        `-- ...
```

这里需要说明两点：

- 从 `parquet` 可以稳定恢复 `episode`、`task`、`instruction` 和 `camera` 层级；对于当前这两个 case，可以按 `0.npy`、`1.npy` 这样的数字文件名恢复
- 左腕相机目录名在当前样本的二进制文本里能看到 `left_wrist`，但没有完整读出带 `cam_` 前缀的形式；结合当前仓库训练代码，实际使用时通常应当与 `cam_high`、`cam_right_wrist` 对齐为 `cam_left_wrist`

#### Step B. 可选的 parquet 打包

仓库里提供了两个 `parquet` 打包脚本，适合做存储、传输或下游索引，但它们不是当前训练的主入口。

1. `pack_codes_to_parquet.py`

输入目录结构：

```text
<input_dir>/<episode>/<camera>/*.npy
```

输出特点：

- 每个 `episode` 产出一个 `parquet`
- 每一行对应一个视觉 token frame
- 字段包括 `task`、`episode`、`camera`、`frame_index`、`code_shape`、`code`

示例：

```bash
python pack_codes_to_parquet.py \
  /path/to/maniparena_dualarm_codes_160_140 \
  /path/to/maniparena_dualarm_codes_160_140_parquet_group
```

2. `pack_real_all_actions_instruction_to_parquet.py`

输入目录结构：

```text
<input_dir>/<episode>/instruction.txt
<input_dir>/<episode>/actions/*.npy
```

输出特点：

- 每个 `episode` 产出一个 `parquet`
- 以 episode 为单位保存 `instruction` 和整段 `actions`
- 字段包括 `task`、`episode`、`instruction`、`num_actions`、`action_shape`、`actions`

示例：

```bash
python pack_real_all_actions_instruction_to_parquet.py \
  /path/to/real_all \
  /path/to/real_all_actions_instruction_parquet_group
```

要特别说明的是：当前仓库里的训练脚本依旧以 `pickle` 为主输入，而不是直接读这些 `parquet`。

#### Step C. 生成归一化后的 ManipArena pickle

使用：

```bash
python tools/pickle_gen/pickle_generation_maniparena.py \
  --dataset_path /path/to/maniparena_dataset_real/processed_data \
  --output_path /path/to/maniparena_dataset_real/processed_data/meta \
  --normalizer_path configs/normalizer_real \
  --output_filename real_all_dualarm_norm.pkl
```

这个脚本会做的事情是：

- 读取 `real_all/<episode>/instruction.txt`
- 读取 `real_all/<episode>/actions/*.npy`
- 查找 `maniparena_dualarm_codes_*` 或 `maniparena_dualarm_recodes_*` 下的多视角 token
- 在 `action` 和三个视角之间做长度对齐
- 生成如下结构的 `pickle`

```python
{
    "text": ...,
    "cam_high": [...],
    "cam_left_wrist": [...],
    "cam_right_wrist": [...],
    "action": np.ndarray,
}
```

- 计算动作归一化统计量并保存到 `normalizer_path`

产物包括：

- `meta/real_all_dualarm_norm.pkl`
- 位于 `configs/normalizer_real/` 或你自定义目录下的 `normalizer stats`

#### Step D. 提取 structured frames / keysteps

当前相关的提取器有两个。

对于 `planner training`，更推荐使用统一版本：

```bash
python tools/structured_frames/structured_frames_extract_maniparena.py \
  --datasets real \
  --dataset /path/to/meta/real_all_dualarm_norm.pkl \
  --out_dir /path/to/structout_testset
```

它会输出：

- `triplets_manifest.csv`
- `summary.json`

#### Step E. 训练 structured planner

`ManipArena real planner`：

```bash
bash scripts/planner/train_video_1node_maniparena_real.sh
```

`ManipArena sim planner`：

```bash
bash scripts/planner/train_video_1node_maniparena_sim.sh
```

这两个脚本最终调用的都是 `train/train_moe_planner.py`，并使用：

- `--data_path` 指向 `*_dualarm_norm.pkl`
- `--keystep_path` 指向 `triplets_manifest.csv`

在运行之前，需要先把脚本里面的绝对路径改成你本机的路径。

#### Step F. 训练 action model

`ManipArena simulation` 的动作训练脚本是：

```bash
bash scripts/simulator/maniparena/train.sh
```

另一个变体是：

```bash
bash scripts/simulator/maniparena/train_aloha.sh
```

这两个脚本最终调用 `train/train_moe.py`。

基于当前仓库状态，可以这样理解：

- `ManipArena` 的 `planner`，真实和仿真都有专门脚本
- `ManipArena` 的 `action training`，当前仓库里明确提供的是仿真侧脚本
- 仓库里的 `scripts/real_world/train_real_world_robot.sh` 属于下面会讲的 `Franka` 真实机器人分支，并不是双臂 `ManipArena real pickle` 的动作训练脚本

### 4. `maniparena-repo/` 在整条链里的作用

`maniparena-repo/` 是部署和评测这一层。

它负责的事情是：

- 接收 `ManipArena evaluation platform` 发来的 observation
- 把 observation 转成你的模型输入
- 执行策略推理
- 把动作 chunk 按平台需要的格式返回出去

最关键的文件有：

- `maniparena-repo/serve.py`：方便启动服务的入口
- `maniparena-repo/maniparena/`：协议、server、转换工具
- `maniparena-repo/examples/my_policy.py`：策略适配层
- `maniparena-repo/examples/model_wrapper_emu.py`：面向 3 视角、14 维双臂动作的 `StructVLA-style Emu wrapper`

典型启动方式：

```bash
cd maniparena-repo
python serve.py \
  --checkpoint /path/to/your/checkpoint \
  --control-mode end_pose \
  --port 5901
```

本地检查：

```bash
python scripts/mock_ping.py --uri ws://127.0.0.1:5901
python scripts/mock_schema_check.py --uri ws://127.0.0.1:5901
```

`open-loop evaluation`：

```bash
python scripts/eval_openloop.py \
  --server ws://127.0.0.1:5901 \
  --dataset /path/to/maniparena_sim_task \
  --episode 0 \
  --save-dir openloop_plots \
  --action-chunk 10
```

如果你的目标是把 `StructVLA checkpoint` 真正接到 `ManipArena benchmark` 上，那么 `maniparena-repo/` 就是最后那层对外暴露服务的封装。

## 当前仓库里的 Franka real-world 流程

仓库里还保留了另一条真实机器人流程，用于处理基于 `CSV` 的 `Franka teleoperation data`。它和双臂 `ManipArena` 相关，但目录结构和数据形态并不相同。

使用：

```bash
python tools/real_world/teledata_process.py \
  --raw_root /path/to/raw_root_a /path/to/raw_root_b \
  --out_root /path/to/processed_out \
  --run_encode \
  --build_pkl
```

这个脚本可以：

- 遍历 `RAW_ROOT/<timestamp>/episodes_5hz/*.csv`
- 把语言和逐步动作写入 `real_all/`
- 把 resize 后的 RGB 图像写入 `real_all_raw_*`
- 调用视觉 tokenizer，生成：
  - `real_all_codes_main_<W>x<H>`
  - `real_all_gripper_codes_wrist_<W>x<H>`
- 构建：
  - `meta/real_all_norm.pkl`
  - `normalizer stats`

这一条 `Franka` 流程，正是 `scripts/real_world/train_real_world_robot.sh` 所使用的输入。

训练方式：

```bash
bash scripts/real_world/train_real_world_robot.sh
```

仓库里还提供了一个基于 `Flask` 的推理服务：

- `tools/real_world/real_experiments_server.py`

所以，按当前代码状态区分：

- `real_all_norm.pkl` 属于 `Franka real-world` 这条分支
- `real_all_dualarm_norm.pkl` 属于双臂 `ManipArena` 这条分支

## 其他仍可使用的流程

仓库中仍然保留了：

- `tools/process/libero_process.py`
- `tools/process/simplerenv_bridge.py`
- `tools/pickle_gen/pickle_generation_libero.py`
- `tools/pickle_gen/pickle_generation_simplerenv_bridge.py`
- `scripts/simulator/libero/train_libero_video.sh`
- `scripts/simulator/simplerenv/train_simplerenv_bridge_video.sh`

如果你要看这些 benchmark 的补充说明，可参考：

- `docs/libero.md`
- `docs/simpler.md`
- `docs/real-world.md`

## 实用说明

- 很多 `shell script` 里写的是绝对路径，运行前需要按你的机器环境修改。
- `ManipArena` 的视觉 token 提取脚本当前默认仍然指向仿真路径。
- 两个 `parquet packer` 适合做数据打包，但当前训练代码主入口仍然读取 `pickle`。
- 对于双臂 `ManipArena` 数据，最关键的字段是 `text`、`cam_high`、`cam_left_wrist`、`cam_right_wrist` 和 `action`。
