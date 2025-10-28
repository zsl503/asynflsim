# AsynFL — 异步联邦学习仿真框架

这是一个用于研究和仿真异步联邦学习（Asynchronous Federated Learning，AsynFL）的 Python 框架。它提供多种联邦学习算法实现、数据分割/生成工具、可视化与记录器，方便开展算法对比、非IID 分布生成及实验复现。

## 主要功能
- 多种联邦学习算法实现（详见 `src/fl/`）：如 FedAvg、FedAsync、FedBuff、FedProx、FedDyn 及若干变体/改进算法。
- 支持客户端异步行为与仿真环境（基于 SimPy），可模拟客户端延迟、速度因子和中断场景。
- 数据生成与分割工具（`generate_data.py`）：支持 iid、Dirichlet、shards、随机分配类别、语义分区等多种非IID 分布策略。
- 数据集接口与处理（`src/utils/data`）：包含多个数据集适配器（MNIST、CIFAR、FEMNIST、CelebA、Sent140 等）以及分区脚本。
- 训练过程记录与可视化（`src/utils/record.py`）：记录聚合时间线、客户端时间、动画生成（可选）及 TensorBoard 支持。

## 仓库结构（关键目录）

- `src/fl/`：联邦学习算法实现（每个算法以模块形式存在，如 `fedavg.py`、`fedasync.py`、`fedbuff.py` 等）。
- `src/utils/`：数据加载、预处理、工具函数与记录器。
- `src/models/`：模型注册与定义。
- `generate_data.py`：数据分割与生成脚本，用于构造 iid 或多种 non-iid 分布数据集目录（输出到 `src/data/` 下的子目录）。
- `main.py`：实验主入口，负责加载参数、构建仿真环境、初始化服务端与客户端并运行仿真。

## 实现的核心算法

该项目包含并实验化了多种联邦学习与异步机制，主要包括（但不限于）：

- FedAvg（同步平均）
- FedAsync / FedAC（异步联邦学习变体）
- FedBuff（带缓冲/队列的异步聚合策略）
- FedProx（带有正则化项的联邦算法）
- FedDyn（动态正则化/自适应方法）
- MIME, SCAFFOLD 等优化与校正方法

每个算法模块遵循统一接口：`Server` 与 `Client` 类，利用 SimPy 驱动 `server_process` 与 `client_process` 来模拟异步时间与事件。

## 算法与代表性论文

下面列出仓库中已确认或常用的算法及其代表性论文（优先选择原始/权威的 arXiv/会议版本），找不到或不确定的实现未列出：

| 算法 | 代表性论文 | 链接 | 备注 |
|---|---|---|---|
| FedAvg | Communication-Efficient Learning of Deep Networks from Decentralized Data — H. B. McMahan et al. (AISTATS 2017) | https://arxiv.org/abs/1602.05629 | 已实现 | 
| FedProx | Federated Optimization in Heterogeneous Networks — Tian Li et al. (MLSys 2020) | https://arxiv.org/abs/1812.06127 | 已实现 |
| FedBuff | Federated Learning with Buffered Asynchronous Aggregation — Yousefpour et al. | https://arxiv.org/abs/2106.06639 | 已实现  |
| SCAFFOLD | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning — Karimireddy et al. | https://arxiv.org/abs/1910.06378 | 已实现  |
| FedFa | FedFa: A Fully Asynchronous Training Paradigm for Federated Learning — Xu et al. | https://arxiv.org/abs/2404.11015 | 已实现  |
| CA2FL | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization — Wang et al. | https://arxiv.org/abs/2007.07481 | 已实现  |
| MIME | Mime: Mimicking Centralized Stochastic Algorithms in Federated Learning — Karimireddy et al. | https://arxiv.org/abs/2008.03606 | 待测试 |
| FedDyn | Federated Learning Based on Dynamic Regularization — Acar et al. | https://arxiv.org/abs/2111.04263 | 待测试 |
| FADAS | FADAS: Towards Federated Adaptive Asynchronous Optimization — Wang et al. (ICML 2024) | https://arxiv.org/abs/2407.18365 | 待实现 |
| FedAC | Efficient Asynchronous Federated Learning with Prospective Momentum Aggregation and Fine-Grained Correction — Zang et al. (ICML 2024) | https://ojs.aaai.org/index.php/AAAI/article/view/29603 | 待实现 |


## 快速开始

先安装依赖（项目未包含完整依赖清单，请在虚拟环境中安装常见包）：

```bash
pip install torch numpy matplotlib simpy tensorboard
```

生成数据（以 CIFAR10 的 Dirichlet 分布为例）：

```bash
python generate_data.py -d cifar10 -cn 50 -a 0.5
```

常用参数解释（部分）:
- `-d/--dataset`：数据集名称（见 `src/utils/data/datasets.py` 中 `DATASETS`）。
- `-cn/--client_num`：客户端数量。
- `-a/--alpha`：Dirichlet 非IID 系数（alpha > 0 时生效）。
- `--iid`：使用 IID 分区（覆盖 Dirichlet）。

运行仿真实验（示例）：

```bash
python main.py --algo FedBuff --output_dir output/ --param_file src.config.default --a 0.1 --c 100--post_str c100p20 --device=cpu --f 
```

示例参数说明：

- `--algo FedBuff`：选择联邦学习算法为 FedBuff（缓冲/队列式异步聚合）。脚本内部会将算法名转换为小写用于加载对应模块（例如 `src.fl.fedbuff`）。
- `--output_dir output/`：指定输出根目录为 `output/`（脚本会在该目录下再根据 `model_name` 与算法名创建子目录，或直接使用提供的完整路径）。
- `--param_file src.config.default`：指定参数类所在模块为 `src.config.default`，脚本将从该模块动态加载名为 `{Algo}Params` 的参数类（例如 `FedBuffParams` 或 `fedBuffParams`，具体取决于模块中类名）。
- `--a 0.1`：Dirichlet 分布的 alpha 值（非IID 强度），当生成数据或选择非IID 分区时生效。值越小，数据越不均衡/异质。
- `--c 100`：命令行短形式 `-c` 或 `--client_num`（在 `main.py` 中也可能通过 `--num_clients` 指定），表示客户端数量为 100。
- `--f`：短形式 `--force`（布尔开关），用于强制将命令行参数覆盖参数类中已有的值（避免因冲突导致脚本抛错）。
- `--device=cpu`: 将device指定为cpu，而不采用配置文件设置，仅当`--f`时起作用。
- `--post_str c100p20`：用于在输出目录名或分布字符串中附加后缀 `_c100p20`（便于区分不同实验的输出）。

注意事项：

- `main.py` 中存在多种参数别名（例如 `--num_clients`, `--client_num`, `-c` 等），脚本会合并命令行显式参数与 `param_file` 中的默认参数；当发生不一致且未加 `--force` 时会抛出错误以避免无意覆盖。
- 算法模块名与参数类名的大小写与命名需在你的 `src/config` 与 `src/fl` 中对应，否则脚本将在导入时失败。脚本会尝试把 `--algo` 转为小写后导入 `src.fl.{algo}`。

说明：`main.py` 支持通过 `--param_file` 指定参数类（参数类在 `src/config` 中），默认从 `src.config.default` 加载。你也可以通过命令行覆盖参数（脚本会检测冲突并提示）。

生成速度因子（若参数中未提供）
- 当 `params.speed_factors` 为 None 时，`main.py` 会调用 `gen_speed_factor`，读取数据目录下的 `all_stats.json` 并基于指数分布生成客户端速度因子。可通过 `--speed_lda` 调整 lambda 值。

## 参数配置

参数类位于 `src/config/` 下（例如 `default.py`, `params.py`），每个算法有对应的 `*Params` 类（例如 `fedBuffAdvParams`）。`main.py` 将根据命令行选择 `algo` 并加载相应的参数类。

如果想快速创建实验配置：
1. 复制 `src/config/default.py` 并修改为新的参数类，或在命令行中通过 `--param_file` 指定含有你自定义参数类的模块。
2. 可通过命令行直接覆盖任意参数（注意：若冲突且未使用 `--force`，脚本会报错）。

## 输出与可视化

- 输出目录结构（默认）：

- `output/{model_name}/{algo}_{dist}`：实验输出主目录，包含日志、模型与 recorder 数据。
- `final_global_model.pth`：训练结束后保存的全局模型权重。
- `recorder/`：记录器导出的事件数据，可用于分析或动画重建。
- `aggregation_timeline.png`, `client_times.png`：部分自动生成的可视化文件。

可选：根据 `src/utils/record.py`，可以生成基于 Matplotlib/FFmpeg 的动画（若系统环境支持 FFmpeg）。

## 开发与扩展建议

- 若要加入新算法：在 `src/fl/` 中新增模块，提供 `Server` 与 `Client` 类并保持接口兼容（参照 `base.py`）。
- 若要支持新数据集：在 `src/utils/data/datasets.py` 中添加适配器并在 `DATASETS` 注册。

## 已知问题

- 数据划分默认采用含一个全局测试集的形式，服务端通过全局测试集进行测试。如若不使用全局测试集，可能会出现未知问题。

- 同步模式（self.select_method = random_sync）选择客户端阶段会出现问题
