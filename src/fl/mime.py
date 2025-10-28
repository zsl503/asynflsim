import torch
import torch.nn.functional as F

from .fedavg import FedAvgServer, FedAvgClient

class MimeServer(FedAvgServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)
        self.beta = params.beta
        self.s = {key: torch.zeros_like(value, device=self.device).float() for key, value in model.state_dict().items()}

    def aggregate(self):
        avg_delta = {}
        for key, value in self.global_model.state_dict().items():
            layer_sum = torch.stack(
                [delta[key] for delta, _, _, model_version in self.buffer]
            ).sum(0)
            avg_delta[key] = layer_sum / len(self.buffer)

            self.s[key] = (1 - self.beta) * avg_delta[key] + self.beta * self.s[key]

        # 应用全局更新（含服务器学习率）
        current_weights = self.global_model.state_dict()
        new_weights = {
            key: (current_weights[key] - self.params.server_lr * avg_delta[key].to(self.device))
            for key in current_weights
        }
        self.global_model.load_state_dict(new_weights)

        self.buffer.clear()
        self.aggregation_count += 1
        self.model_version += 1
        
        self.check_and_validate()
        self.recorder.record_aggregation(self.env.now, self.model_version)
        self.recorder.aggregation_times.append(self.env.now)
        self.check_stop_condition()

    def send_msg(self, client_id):
        ''' 默认返回全局模型，此处没有深拷贝，请注意使用 '''
        return (self.model_version, self.global_model.state_dict(), self.s)

class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, beta=0.9):
        super(MyOptimizer, self).__init__(params, dict(lr=lr))
        self.beta = beta

    def step(self, s=None, named_params=None):
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                lr = group['lr']
                
                # 尝试用外部传入的 name 映射
                name = None
                for n, param in named_params.items():
                    if p is param:
                        name = n
                        break

                if name is not None and name in s:
                    p.data -= lr * ((1 - self.beta) * p.grad.data + self.beta * s[name])
                else:
                    p.data -= lr * p.grad.data

class MimeClient(FedAvgClient):
    def __init__(self, client_id, base_model, data_loaders, recorder, params, speed_factor):
        super().__init__(client_id, base_model, data_loaders, recorder, params, speed_factor)
        self.old_delta = None
        self.last_global_params = None
        self.local_train = self.gcr_local_train_minibatch

    def recv_from_server(self, server):
        self.model_version, state_dict, s  = server.send_msg(self.client_id)
        self.model.load_state_dict(state_dict)
        self.s = {key: value.clone() for key, value in s.items()}

    def send_to_server(self, server):
        delta = {}
        for k in self.initial_state:
            delta[k] = self.initial_state[k] - self.model.state_dict()[k]

        self.old_delta = delta

        # 上传前注册
        self.registration(server)
        # 上传参数
        with server.res.request() as req:
            yield req
            server.recv_msg((delta, len(self.train_data_loader.dataset), self.client_id, self.model_version))

    def gcr_local_train_minibatch(self):
        named_params = dict(self.model.named_parameters())
        self.optimizer = MyOptimizer(
            named_params.values(),
            lr=self.params.learning_rate,
            beta=self.params.beta
        )
        self.model.train()

        for _ in range (self.params.local_rounds):
            for images, labels in self.train_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                # self.optimizer.step()
                self.optimizer.step(self.s, named_params=named_params)

Client = MimeClient
Server = MimeServer
