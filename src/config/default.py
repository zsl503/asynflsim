from .params import BaseExperimentParams

class ExperimentParams(BaseExperimentParams):
    def __init__(self):
        # 基础参数
        self.num_clients = 50
        self.num_rounds = None
        self.local_rounds = 1
        self.sim_time = 600000
        self.server_lr = 1.0  # 通常设为1.0直接使用客户端更新
        self.buffer_size = None
        self.clients_per_round = 20
        self.device = "cuda:0"

        # 设备参数
        self.batch_size = 32
        self.learning_rate = 0.01
        self.momentum = 0.9

        self.speed_mode = "assign" # assign, multi, add
        # self.speed_factors = [1.0]
        self.speed_factors = [1.0] * 10 + [3.0] * 40        
        # 异构性参数
        # self.speed_factors=[2.22154743, 2.99225874, 2.56014311, 2.28116789, 1.68086805, # 均值3
        #                     2.72018576, 3.22160386, 3.12498416, 3.28393421, 3.12426597,
        #                     2.4651326 , 2.72583368, 4.50757313, 3.163874  , 3.45565972,
        #                     2.42281847, 2.8569426 , 2.094492  , 3.41230004, 3.59521698,
        #                     2.82390734, 3.30307675, 3.85890725, 2.15136005, 3.04735932,
        #                     2.12399591, 4.09377067, 3.68923684, 1.81302263, 2.7655528 ,
        #                     3.01695142, 1.21647192, 4.05221233, 2.70117355, 2.09826563,
        #                     4.39108626, 2.58213705, 4.10158726, 5.20221216, 3.78843891,
        #                     2.22464031, 4.2828837 , 2.73704616, 1.35857412, 1.12687816,
        #                     2.5600062 , 3.11502641, 1.3670042 , 3.11939898, 3.35920345]

        # 均值5的异构性参数
        # self.speed_factors=[5.33806256, 6.78163249, 8.52968966, 4.88051869, 3.14654608,
        #                     4.823028  , 3.00429364, 5.49920882, 3.19710493, 5.83224037,
        #                     2.27320093, 5.23144287, 4.29175414, 8.19244236, 3.49180549,
        #                     2.26117675, 0.24128012, 0.08882558, 6.01587755, 5.84608239,
        #                     6.01387751, 6.16857685, 2.96576352, 8.52386047, 2.19848387,
        #                     0.60039544, 1.36554745, 3.97410425, 4.38379958, 4.46232702,
        #                     5.0531861 , 5.01553039, 5.68354896, 5.66300858, 7.70466424,
        #                     6.27728997, 5.9536948 , 4.46312958, 1.41441875, 3.19587747,
        #                     6.64279606, 4.63128938, 5.56483596, 9.38136693, 3.68467164,
        #                     7.46536746, 8.24959875, 5.89712931, 4.35031017, 3.1400713 ]

        # 长尾分布[1-20]，b=2.5
    #     self.speed_factors = \
    #     [5.46265944, 12.91949603,  6.55888291,  3.34577912,  1.51722354,
    #     4.0582963 , 13.12100033,  1.04668563,  3.28848478,  3.92762119,
    #     5.21556061,  1.53524727,  7.85141225,  5.26753533,  4.28998947,
    #     2.28359382,  6.84678854,  2.57683991,  6.43821832, 15.18724675,
    #     3.73092234,  5.74559437,  4.3077141 ,  1.13071328,  3.36051751,
    #     2.86603287, 13.84304777,  1.85339283, 10.42791467, 14.86796715,
    #    19.96339895,  9.27013151,  1.4679713 ,  4.34075914,  1.20064414,
    #     2.70932031,  1.70630094, 16.17324755, 10.1486854 ,  1.20578804,
    #     1.067746  ,  4.74493252,  5.83721018,  8.56628084,  9.78441418,
    #    17.41315845,  6.26679021, 12.25731504,  6.53398904,  3.11782075]

        # 数据集参数
        self.dataset_name = "cifar10"
        self.dataset_dir = None
        self.dataset_args = None

        # 模型参数
        # self.model_name = "res18"
        self.model_name = "simplecnn"
        
        # 算法选择
        self.algorithm = None

        self.seed = 42
        self.video_format = "gif"
        
    def __str__(self):
        # Get the dictionary of attributes (including those from subclasses)
        attrs = vars(self)
        
        # Format each attribute and value as 'key: value' pairs
        formatted_attrs = [f"{key}: {value}" for key, value in attrs.items()]
        
        # Join them into a single string with newlines
        return "\n".join(formatted_attrs)

class FedBuffParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = 5
        self.dataset_dir = dataset_dir
        if self.model_name == "simplecnn":
            self.num_rounds = 1000
        elif self.model_name == "res18":
            self.num_rounds = 300
        self.use_sample_weight = False


class FedAsyncParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = 1
        self.dataset_dir = dataset_dir
        if self.model_name == "simplecnn":
            self.num_rounds = 1000
        elif self.model_name == "res18":
            self.num_rounds = 300
        self.use_sample_weight = False
        self.validation_interval = 10
        self.alpha = 0.9

class FedCRSParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = 5
        self.dataset_dir = dataset_dir
        if self.model_name == "simplecnn":
            self.num_rounds = 1000
        elif self.model_name == "res18":
            self.num_rounds = 300
        self.use_sample_weight = False
        self.history_buffer_size = 15
        self.gamma = 0.5

class FedCRMParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = 10
        self.dataset_dir = dataset_dir
        # self.algorithm = "fedac"
        if self.model_name == "simplecnn":
            self.num_rounds = 1000
        elif self.model_name == "res18":
            self.num_rounds = 300
        self.use_sample_weight = False
        self.history_buffer_size = 10
        self.beta = 0

class FedFaParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        # self.buffer_size = 10
        self.dataset_dir = dataset_dir
        self.algorithm = "fedfa"
        if self.model_name == "simplecnn":
            self.num_rounds = 5000
        elif self.model_name == "res18":
            self.num_rounds = 1000
        self.use_sample_weight = False
        self.clients_per_round = None
        self.buffer_size = 10

class FedACParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = 10
        self.dataset_dir = dataset_dir
        # self.algorithm = "fedac"
        if self.model_name == "simplecnn":
            self.num_rounds = 1000
        elif self.model_name == "res18":
            self.num_rounds = 300
        self.use_sample_weight = False
        self.history_buffer_size = 10

class FedBuffAdvParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = 30
        self.dataset_dir = dataset_dir
        self.algorithm = "fedbuffadv"
        if self.model_name == "simplecnn":
            self.num_rounds = 1000
        elif self.model_name == "res18":
            self.num_rounds = 300
        self.use_sample_weight = False
        self.indicator_threshold = 0.95
        self.class_num = 10
        self.max_candidate = 100
        self.version_tolerance = 10

class FedBuffAdv2Params(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = 10
        self.dataset_dir = dataset_dir
        self.algorithm = "fedbuffadv2"
        if self.model_name == "simplecnn":
            self.num_rounds = 1000
        elif self.model_name == "res18":
            self.num_rounds = 300
        self.use_sample_weight = False
        self.mu = 0.01
        self.history_buffer_size = 10
        # self.hot_round = 10
        self.beta = 0

class CA2FLParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.algorithm = "ca2fl"
        if self.model_name == "simplecnn":
            self.num_rounds = 1000
        elif self.model_name == "res18":
            self.num_rounds = 300
        self.buffer_size = 10
        self.clients_per_round = 20
        self.server_lr = 1

class FedAvgParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = self.num_clients
        self.participation_ratio = 1
        self.dataset_dir = dataset_dir
        self.algorithm = "fedavg"
        self.num_rounds = 100

class Fedtest1Params(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = 10
        self.participation_ratio = 1
        self.dataset_dir = dataset_dir
        self.algorithm = "fedtest1"
        self.num_rounds = 20

class Fedtest2Params(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = self.num_clients
        self.participation_ratio = 1
        self.dataset_dir = dataset_dir
        self.algorithm = "fedtest2"
        self.num_rounds = 100
        self.relax = 0.95
        self.nw_need_client = 5
        self.max_window_size = 5

class Fedtest3Params(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        self.buffer_size = self.num_clients
        self.participation_ratio = 1
        self.dataset_dir = dataset_dir
        self.algorithm = "fedtest3"
        if self.model_name == "simplecnn":
            self.num_rounds = 1000
        elif self.model_name == "res18":
            self.num_rounds = 300
        self.relax = 0.95
        self.nw_need_client = 5
        self.max_window_size = 5
