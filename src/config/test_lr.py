from .params import BaseExperimentParams

class ExperimentParams(BaseExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        # 基础参数
        self.num_clients = 50
        self.local_rounds = 1
        self.sim_time = 600000
        self.server_lr = 1.0  # 通常设为1.0直接使用客户端更新
        self.buffer_size = 10
        self.clients_per_round = None
        self.device = "cuda:1"
        self.train_method = "minibatch"  # minibatch, fullbatch
        self.select_method = "random_async"  # random_async, random_sync 

        # 设备参数
        self.optimizer = "sgd"
        self.batch_size = 32
        self.learning_rate = 0.01
        self.momentum = 0

        self.speed_mode = "assign" # assign, multi, add
        
        self.speed_factors = None

        self.use_random_delay = False

        # 数据集参数
        self.dataset_name = "cifar10"
        self.dataset_dir = dataset_dir
        self.dataset_args = None

        # 模型参数
        self.model_name = "res18"
        # self.model_name = "avgcnn"
        
        # 算法选择
        self.algorithm = None

        self.seed = 42
        self.video_format = "gif"
        self.validation_interval = 1

        self.use_sample_weight = False
        
        self.num_rounds = 100

class FedBuffParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)

class FedAsyncParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.buffer_size = 1
        if self.model_name == "simplecnn":
            self.num_rounds = 10000
        elif self.model_name == "res18":
            self.num_rounds = 2000
        self.alpha = 0.9
        self.validation_interval = 10

class FedGCRParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.gamma = 0.2
        self.mu = 0.1
        
class CA2FLParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.use_stale = False

class FedFAParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.server_lr = 1
        self.validation_interval = 10
        
        if self.model_name == "simplecnn":
            self.num_rounds = 10000
        elif self.model_name == "res18":
            self.num_rounds = 6000


class CA2FLCRParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.mu = 0.1


class FedBuffCRParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.mu = 0.1
