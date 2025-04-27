import copy
import logging
import math
import random
import torch
import torch.nn.functional as F

from .fedbuff import FedBuffClient
from .base import BaseServer, BaseClient

class ScaffoldServer(BaseServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)
        self.buffer_size = params.buffer_size
        # 初始化服务器控制变量
        self.server_controls = {}
        for key in self.global_model.state_dict().keys():
            self.server_controls[key] = torch.zeros_like(self.global_model.state_dict()[key]).float()
        self.selected_clients = set()

    def select_clients(self):
        """客户端选择策略：随机选择Mc个活跃客户端"""
        if not self.params.use_async:
            # 可选池子中未选过的客户端
            all_ids = [client_id for client_id in self.client_pool.keys() if client_id not in self.selected_clients]
            client_per_round = self.params.clients_per_round if self.params.clients_per_round is not None else self.params.num_clients
            if client_per_round <= len(all_ids):
                selected_ids = random.sample(all_ids, client_per_round)
            else:
                selected_ids = all_ids

            # 选择的客户端数量不足时，从可选池中随机选择一些客户端
            if len(selected_ids) < client_per_round:
                rest_set = set(list(self.client_pool.keys())) - set(selected_ids)
                extra_ids = random.sample(rest_set, client_per_round - len(selected_ids))
                selected_ids += extra_ids
                # 更新已选择的客户端集合
                self.selected_clients = set(selected_ids)
            else:
                # 更新已选择的客户端集合
                self.selected_clients |= set(selected_ids)

            logging.info(f"[Round {self.model_version}] Selected clients: {sorted(selected_ids)}")
            return selected_ids
        else:
            return super().select_clients()
    
    def recv_msg(self, msg):
        (delta, client_controls, num_samples, client_id, model_version) = msg
        self.buffer.append(msg)
        logging.info(f"[Client {client_id}] Update uploaded ({len(self.buffer)}/{self.buffer_size}) at {self.env.now:.2f}s")
        self.recorder.record_buffer_update(
            self.env.now,
            [x[3] for x in self.buffer]
        )
        self.client_update_count += 1
        self.total_staleness += self.model_version - model_version
        if not self.aggregation_trigger.triggered:
            self.aggregation_trigger.succeed()
    
    def send_msg(self, client_id):
        ''' 默认返回全局模型，此处没有深拷贝，请注意使用 '''
        return (self.model_version, self.global_model.state_dict(), copy.deepcopy(self.server_controls))

    def aggregate(self):
        def staleness(model_version):
            return 1 / math.sqrt(1 + self.model_version - model_version)
        
        avg_delta = {}
        avg_control = {}
        new_weights = {}
        current_weights = self.global_model.state_dict()
        # 聚合模型更新和控制变量更新
        for key in current_weights.keys():
            # self.server_controls[key] = torch.zeros_like(current_weights[key]).float()
            layer_sum = torch.stack(
                [delta[key].float() * staleness(model_version) for delta, _, _, client_id, model_version in self.buffer]
            ).sum(0)
            avg_delta[key] = layer_sum / len(self.buffer)
            
            control_sum = torch.stack(
                [delta_control[key].float() for _, delta_control, _, _, model_version in self.buffer]
            ).sum(0)
            avg_control[key] = control_sum / self.params.num_clients
            
            # 更新服务器控制变量
            self.server_controls[key] += avg_control[key]
            # 更新模型权重
            new_weights[key] = current_weights[key] - self.params.server_lr * avg_delta[key]

        self.global_model.load_state_dict(new_weights)

        self.buffer.clear()
        self.aggregation_count += 1
        self.model_version += 1

        self.check_and_validate()
        self.recorder.record_aggregation(self.env.now, self.model_version)
        self.recorder.aggregation_times.append(self.env.now)
        self.check_stop_condition()

class ScaffoldClient(FedBuffClient):
    def __init__(self, client_id, base_model, data_loaders, recorder, params, speed_factor):
        super().__init__(client_id, base_model, data_loaders, recorder, params, speed_factor)
        self.local_train = self.fullbatch_local_train
        self.initial_state = None
        self.server = None
        # 初始化客户端控制变量
        self.client_controls = {}
        self.server_controls = None
        for key in self.model.state_dict().keys():
            self.client_controls[key] = torch.zeros_like(self.model.state_dict()[key])
    
    def recv_from_server(self, server):
        self.model_version, state_dict, self.server_controls = server.send_msg(self.client_id)
        self.model.load_state_dict(state_dict)

    def fullbatch_local_train(self):
        # Scaffold本地训练，不适用于minibatch，拼接为一个大的 batch
        self.model.train()
        all_images, all_labels = [], []
        for images, labels in self.train_data_loader:
            all_images.append(images)
            all_labels.append(labels)
        images = torch.cat(all_images).to(self.device)
        labels = torch.cat(all_labels).to(self.device)

        for _ in range(self.params.local_rounds):
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward() 
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                param.grad.data += (- self.client_controls[name] + self.server_controls[name]).data
            self.optimizer.step()

    def send_to_server(self, server):
        delta = {}
        delta_control = {}
        # 计算模型差异和控制变量更新
        with torch.no_grad():
            for k in self.initial_state:
                delta[k] = self.initial_state[k] - self.model.state_dict()[k]
                old_control = self.client_controls[k].clone().detach()
                self.client_controls[k] = self.client_controls[k] - self.server_controls[k] + \
                        delta[k] / (self.params.local_rounds * self.params.learning_rate)
                delta_control[k] = self.client_controls[k] - old_control
        
        # 上传前注册
        self.registration(server)
        with server.res.request() as req:
            yield req
            server.recv_msg((delta, delta_control, len(self.train_data_loader.dataset), self.client_id, self.model_version))

Client = ScaffoldClient
Server = ScaffoldServer
