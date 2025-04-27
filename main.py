import datetime
import json
import os

from src.models.models import MODELS
os.environ['NUMEXPR_MAX_THREADS'] = "16"

import random
from matplotlib import pyplot as plt
import numpy as np
import simpy
import importlib
from torch.utils.data import DataLoader

import torch
from src.fl.base import BaseClient, BaseServer
from src.config.params import BaseExperimentParams
from src.utils.data_loader import DataHandler
from src.utils.record import AdvFLAnimator, SimulationRecorder, FLAnimator
import logging


# 配置随机种子保证可重复性
def fix_random_seed(seed: int) -> None:
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def interrupt_handler(env, stop_event, process):
    yield stop_event
    process.interrupt()

def main(params:BaseExperimentParams, output_dir:str):
    logging.info("\nParams:\n"+str(params))
    recorder = SimulationRecorder(num_clients = params.num_clients, use_tensorboard=True, tensorboard_dir=output_dir)
    
    if params.device != "cpu" and torch.cuda.is_available():
        gpu_id = int(params.device.split(":")[-1])
        if gpu_id >= torch.cuda.device_count():
            logging.info(f"GPU ID {gpu_id} is out of range, using CPU instead.")
            params.device = "cpu"
    else:
        logging.info("GPU is not available, using CPU instead.")
        params.device = "cpu"
    
    # 加载数据
    (train_client_datasets, 
     val_client_datasets, 
     test_client_datasets) = \
        DataHandler.load_data(
        dataset_name = params.dataset_name, 
        file_dir = params.dataset_dir,
        args = params.dataset_args,
        center_test = False
    )

    # 加载测试数据集
    test_dataset = DataHandler.load_data(
        dataset_name = params.dataset_name, 
        file_dir = params.dataset_dir,
        args = params.dataset_args,
        center_test=True
    )

    # 初始化模型
    global_model = MODELS[params.model_name](dataset = params.dataset_name).to(params.device)
    # 输出模型所有层名称、参数数据类型和参数的形状
    logging.info(f"Model {params.model_name} initialized")
    for name, param in global_model.named_parameters():
        logging.info(f"{name}: {param.data.dtype} {param.data.shape}")
    # 初始化算法
    algorithm_module = importlib.import_module(f"src.fl.{params.algorithm.lower()}")
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size)
    # logging.info(len(test_dataset))
    server:BaseServer = algorithm_module.Server(global_model, test_loader, recorder, params)
    
    # 创建仿真环境
    env = simpy.Environment()
    server.init_env(env)
    # 创建客户端进程
    clients = []
    for i in range(params.num_clients):
        # 初始化客户端
        # speed_factor = np.random.choice(params.speed_factors)
        # print(f"Client {i} train num_samples: {len(train_client_datasets[i])}, val num_samples: {len(val_client_datasets[i])}, test num_samples: {len(test_client_datasets[i])}")
        client:BaseClient = algorithm_module.Client(
            client_id=i,
            base_model=global_model,
            data_loaders = (
                DataLoader(train_client_datasets[i], batch_size=params.batch_size, shuffle=True, drop_last=True),
                DataLoader(val_client_datasets[i], batch_size=params.batch_size, shuffle=True, drop_last=True) if len(val_client_datasets[i]) else None,
                DataLoader(test_client_datasets[i], batch_size=params.batch_size, shuffle=True, drop_last=True) if len(test_client_datasets[i]) else None,
                ),
            recorder=recorder, 
            params=params,
            speed_factor=params.speed_factors[i%len(params.speed_factors)],
        )
        client.registration(server)
        # 包装为SimPy进程
        client_process = env.process(client.client_process(server))     
        # 设置中断处理
        env.process(
            interrupt_handler(env, server.stop_event, client_process))
        
        clients.append(client)

    server_process = env.process(server.server_process())
    env.process(
        interrupt_handler(env, server.stop_event, server_process))
    
    # 运行仿真
    logging.info("Simulation started...")

    try:
        env.run(until=server.stop_event)
    except simpy.Interrupt:
        logging.info("Simulation interrupted by user")
    # except Exception as e:
    #     logging.info(f"Simulation failed: {e}")

    # 保存最终模型
    torch.save(server.global_model.state_dict(), f"{output_dir}/final_global_model.pth")
    logging.info(f"Completed {server.aggregation_count} aggregation rounds")
    
    recorder.save(f"{output_dir}/recorder.json")
    logging.info("Recorder events saved to recorder.json")


    # # 画图，每个client一张
    # os.makedirs(f"{output_dir}/loss_time", exist_ok=True)
    # # 画出clients的loss_list和time_list
    # for i in range(params.num_clients):
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(clients[i].time_list, clients[i].loss_list, label=f"Client {i}")
    #     plt.legend()
    #     plt.xlabel('Time')
    #     plt.ylabel('Loss')
    #     plt.title(f'Client {i} Loss-Time Curve')
    #     plt.savefig(f"{output_dir}/loss_time/client_{i}_loss_time.png")
    #     plt.close()

    # 最终可视化
    recorder.visualize_client_times(file_path=f"{output_dir}/client_times.png")
    logging.info("Client time visualization saved to client_times.png")
    
    # # 绘制聚合时间线
    plt.figure(figsize=(14, 6))
    plt.plot(recorder.aggregation_times, marker='o')
    plt.xlabel('Aggregation Round')
    plt.ylabel('Simulation Time (s)')
    plt.title('Aggregation Timeline')
    plt.grid(True)
    plt.savefig(f"{output_dir}/aggregation_timeline.png")

    # 生成动画
    logging.info("Generating animation...")
    if 'test' not in params.algorithm:
        animator = FLAnimator(recorder, params.num_clients, params.buffer_size, time_scale=1)
    else:
        animator = AdvFLAnimator(recorder, params.num_clients, params.buffer_size, time_scale=1, max_window_size=4)
    ani = animator.animate()
    try:
        # 判断格式MP4是否可用
        ani.save(f"{output_dir}/fl_simulation.{params.video_format}", writer="ffmpeg", fps=animator.fps)
        logging.info(f"Animation saved to fl_simulation.{params.video_format}")
    except Exception as e:
        logging.info(f"Animation failed: {e}")
    
def gen_speed_factor(data_dir, lda=0.01, output_dir=None):
    """
    按照fedbuff的指数分布，生成客户端速度因子，使用指数分布生成速度因子，平均值为lda * x，其中x为每个客户端的样本数
    """
    json_file = os.path.join(data_dir, 'all_stats.json')
    stats = json.load(open(json_file, 'r'))
    stats.pop('sample per client')

    speeds = []
    for k, v in stats.items():
        speeds.append(np.random.exponential(scale=lda * v['x']))

    if output_dir:
        x_values = [v['x'] for v in stats.values()]

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, speeds, alpha=0.7)
        plt.xlabel('Data Number')
        plt.ylabel('Speeds')
        plt.title('Scatter Plot of Data Number vs Speeds')
        plt.savefig(os.path.join(output_dir, 'speed_factor.png'))
        
        # Save the speed factors to a file
        plt.figure(figsize=(10, 6))
        plt.hist(speeds, bins=30, edgecolor='black')
        plt.xlabel('Speed Range')
        plt.ylabel('Number of Clients')
        plt.title('Distribution of Speeds')
        plt.savefig(os.path.join(output_dir, 'speed_factor_distribution.png'))
    return speeds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iid', action='store_true', help='使用IID数据分布')
    parser.add_argument('--algo', type=str, default='fedBuffAdv', help='选择联邦学习算法')
    parser.add_argument('--a', type=float, default=0.5, help='数据集Dirichlet 分布系数')
    parser.add_argument('--client_num', type=int, default=50, help='客户端数量')
    parser.add_argument('--force', action='store_true', help='强制应用参数') 
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--device', type=str, default=None, help='device')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--param_file', type=str, default=None, help='参数文件')
    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    parser.add_argument('--post_str', type=str, default='', help='后缀')
    args = parser.parse_args()
    iid = args.iid
    algo = args.algo
    
    args.post_str = f"_{args.post_str}" if args.post_str else ""
    if not iid:
        dataset_dir, dist_str = f'data/noniid-{args.a}_{args.client_num}/cifar10', f"noniid-{args.a}{args.post_str}"
    else:
        dataset_dir, dist_str = f'data/iid_{args.client_num}/cifar10', f"iid{args.post_str}"
    
    params_class_name = f"{algo}Params"
    if args.param_file:
        # 自动导入文件param_file.py，里面会有params_class_name = f"{algo}Params"
        param_file = importlib.import_module(args.param_file)
    else:
        param_file = importlib.import_module("src.config.default")

    # 动态加载名为params_class_name的类
    params = getattr(param_file, params_class_name)(dataset_dir)
    fix_random_seed(params.seed)
    params.algorithm = algo.lower()
    if args.force:
        params.num_clients = args.client_num
        params.seed = args.seed if args.seed is not None else params.seed
        params.device = args.device if args.device is not None else params.device
        params.model_name = args.model_name if args.model_name is not None else params.model_name

    if args.client_num and params.num_clients != args.client_num:
        raise ValueError(f"客户端数量不一致: {params.num_clients} != {args.client_num}, 请检查参数")
    if args.seed and params.seed != args.seed:
        raise ValueError(f"随机种子不一致: {params.seed} != {args.seed}, 请检查参数")
    if args.device and params.device != "cuda:" + args.gpu and args.gpu != "cpu":
        raise ValueError(f"GPU ID 不一致: {params.device} != {args.gpu}, 请检查参数")
    if args.model_name and params.model_name != args.model_name:
        raise ValueError(f"模型名称不一致: {params.model_name} != {args.model_name}, 请检查参数")


    output_dir = f"output/{params.speed_mode}/{params.model_name}/{algo}_{dist_str}" if args.output_dir is None else \
        os.path.join(args.output_dir,f"{params.model_name}/{algo}_{dist_str}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if params.speed_factors is None:
        # 生成速度因子
        params.speed_factors = gen_speed_factor(dataset_dir, lda=0.01, output_dir=output_dir)

    print(
f'''Time:{datetime.datetime.now()}
output_dir:{output_dir}
dataset_dir:{dataset_dir}
Params:
{params}
''')

    logging.basicConfig(
        filename=os.path.join(output_dir, "output.log"), 
        filemode='w',
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y%m%d %H:%M:%S')
    main(params, output_dir)
