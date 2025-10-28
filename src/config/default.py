from .params import BaseExperimentParams, DatasetArgs

class ExperimentParams(BaseExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        # 基础参数
        self.num_clients = 50 # 客户端数量，和选择对应的数据集文件有关
        self.local_rounds = 1   # 每个客户端本地训练轮数
        self.server_lr = 0.1  # 通常设为1.0直接使用客户端更新
        self.buffer_size = 10 # 服务器缓冲区大小
        self.clients_per_round = None   # 每轮选择的客户端数量，None表示使用所有客户端
        self.device = "cuda:0"  # 设备选择，cpu或cuda:0等
        self.train_method = "minibatch"  # minibatch, fullbatch
        self.select_method = "random_async"  # random_async, random_sync
        self.stop_type = 'update'  # 'time', 'round', 'update'，仿真停止类型，其中update表示全局更新次数，round表示服务器轮数
        self.max_updates = 60000    # 全局最大更新次数，当stop_type为'update'时使用
        self.sim_time = 600000  # 仿真最大时间，当stop_type为'time'时使用

        # 设备参数
        self.optimizer = "sgd" # sgd, adam
        self.batch_size = 32     # 本地训练批次大小
        self.learning_rate = 0.01   # 本地学习率
        self.momentum = 0   

        self.speed_mode = "assign" # assign(time=指定因子), multi(time=指定因子*实际执行), add(time=实际执行+指定因子)
        
        self.speed_factors = None # 客户端速度因子列表，None表示遵循输入参数自动生成速度因子

        self.use_random_delay = False   # 是否使用随机延迟

        # 数据集参数
        self.dataset_name = "mnist"   # 数据集名称
        self.dataset_dir = dataset_dir  # 数据集路径，如为变量，则从外部传入
        self.dataset_args = DatasetArgs()

        # 模型参数
        self.model_name = "lenet5"  # 模型名称，可用列表参见 models.py
        
        # 算法选择
        self.algorithm = None # 忽略

        self.seed = 42 # 随机种子
        self.video_format = "gif" # 视频格式，gif或mp4，将在训练结束后生成训练过程可视化视频（需要取消main中对应注释）
        self.validation_interval = 1  # 验证间隔，单位为服务器轮数

class FedBuffParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)

class FedAsyncParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.buffer_size = 1
        self.alpha = 0.9
        self.validation_interval = 10

class CA2FLParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.use_stale = False

class FedFAParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        # self.server_lr = 1
        self.validation_interval = 10

class FedProxParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.mu = 0.01

class MimeParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.beta = 0.5
        self.select_method = "random_sync"  # random_async, random_sync 
        self.clients_per_round = 10

class FedAvgParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.select_method = "random_sync"  # random_async, random_sync 
        self.clients_per_round = 10

class FedDynParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.select_method = "random_sync"  # random_async, random_sync 
        self.clients_per_round = 10
        self.mu = 0.1 # 动态正则化参数，可以用类似形式自定义参数

class FedAsyncParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.buffer_size = 1
        self.alpha = 0.9
        self.validation_interval = 10
