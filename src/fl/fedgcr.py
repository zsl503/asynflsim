# t2是要33行时间系数不要50行（），t3是不要33行时间系数要50行（非常平稳，但是精度不及其它，比fedbuff好），t4是33行和105行时间系数不要50行（后期乏力）
import copy
import logging
import math
import torch
import torch.nn.functional as F

from .fedbuff import FedBuffClient

from .base import BaseServer

class FedGCRServer(BaseServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)
        self.buffer_size = params.buffer_size
        # self.h_t = {}
        self.h_t = {key: torch.zeros_like(value, device=self.device).float() for key, value in model.state_dict().items()}
        self.gamma = self.params.gamma

    def second_correct(self, delta, client_id, model_version, key):
        # 如果是bn层，直接返回
        if model_version == 0 or 'bn' in key or 'downsample' in key:
            return delta
        
        g_delta = self.h_t[key]

        d_vec = delta.view(-1)
        g_vec = g_delta.view(-1)

        beta = torch.dot(d_vec, g_vec)
        alpha = torch.dot(g_vec, g_vec)
        scale = beta / alpha
        select = 0
        if scale > 1:
            select = 1
            res_vec = d_vec - g_vec
        else:
            select = 2
            res_vec = d_vec - scale * g_vec

        
        # Second
        # if scale > 1:
        #     select = 1
        #     res_vec = d_vec
        # else:
        #     if scale > 0:
        #         select = 2
        #         res_vec = d_vec + g_vec
        #     else:
        #         select = 3
        #         res_vec = d_vec - scale * g_vec + g_vec

        # Third
        # if scale > 1:
        #     select = 1
        #     res_vec = d_vec - g_vec
        # elif scale < 0 and alpha > 1e-5:
        #         select = 2
        #         res_vec = d_vec - scale * g_vec
        # else:
        #     select = 3
        #     res_vec = d_vec + g_vec

        logging.info(f"{key}: selected={select} alpha={alpha.item():.3e}, beta={beta.item():.3e}, scale={scale.item():.3f}")
        return res_vec.view_as(delta)

    def recv_msg(self, msg):
        delta, sample, client_id, model_version = msg
        for k in delta:
            # 纠正delta
            tmp = delta[k].float().clone()
            delta[k] = self.second_correct(delta[k].float(), client_id, model_version, k)
            self.h_t[k] += tmp / self.params.num_clients
        
        self.buffer.append((delta, sample, client_id, model_version))
        
        logging.info(f"[Client {client_id}] Update uploaded ({len(self.buffer)}/{self.buffer_size}) at {self.env.now:.2f}s")
        self.recorder.record_buffer_update(
            self.env.now,
            [x[2] for x in self.buffer]
        )
        self.total_staleness += self.model_version - model_version
        self.client_update_count += 1

        # 检查是否达到缓冲区大小
        if not self.aggregation_trigger.triggered and len(self.buffer) >= self.buffer_size:
            self.aggregation_trigger.succeed()

    def aggregate(self):
        logging.info(f"Staleness: {[self.model_version - model_version for _, _, _, model_version in self.buffer]}")
        print(f"Staleness: {[self.model_version - model_version for _, _, _, model_version in self.buffer]}")     
        avg_delta = {}
        for key, value in self.global_model.state_dict().items():
            layer_sum = torch.stack(
                [delta[key] for delta, _, _, model_version in self.buffer]
            ).sum(0)
            avg_delta[key] = layer_sum / len(self.buffer)

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

class FedGCRClient(FedBuffClient):
    def __init__(self, client_id, base_model, data_loaders, recorder, params, speed_factor):
        super().__init__(client_id, base_model, data_loaders, recorder, params, speed_factor)
        self.old_delta = None
        self.last_global_params = None
        self.local_round = 1
        # self.local_train = self.gcr_local_train_minibatch_old

    def send_to_server(self, server):
        delta = {}
        ddelta = {}
        for k in self.initial_state:
            delta[k] = self.initial_state[k] - self.model.state_dict()[k]
            ddelta[k] = delta[k] - self.old_delta[k] \
                if self.old_delta is not None else delta[k].detach()

        self.old_delta = delta

        # 上传前注册
        self.registration(server)
        # 上传参数
        with server.res.request() as req:
            yield req
            server.recv_msg((ddelta, len(self.train_data_loader.dataset), self.client_id, self.model_version))


    def gcr_local_train_minibatch_old(self):
        total_global_delta = None
        if self.last_global_params is not None:
            # 直接计算total_global_delta，不需要中间字典
            total_global_delta = torch.cat([
                (self.last_global_params[key] - w).flatten()
                for key, w in self.model.named_parameters()
            ])

        self.last_global_params = copy.deepcopy(self.model.state_dict())
        self.model.train()

        for _ in range (self.params.local_rounds):
            for images, labels in self.train_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                proximal_term = torch.tensor(0.0, device=self.device)
                loss = F.cross_entropy(outputs, labels)
                if total_global_delta is not None:
                    total_local_delta = torch.cat([
                        (self.last_global_params[key] - w).flatten()
                        for key, w in self.model.named_parameters()
                    ])
                    proximal_term = torch.sum((total_local_delta - total_global_delta)**2)
                    if not torch.isnan(proximal_term):
                        loss += (self.params.mu / 2) * proximal_term                    

                loss.backward()
                self.optimizer.step()


    def gcr_local_train_minibatch(self):
        total_global_delta = None
        if self.last_global_params is not None:
            # 直接计算total_global_delta，不需要中间字典
            total_global_delta = torch.cat([
                (self.last_global_params[key] - w).flatten()
                for key, w in self.model.named_parameters()
            ])

        self.last_global_params = copy.deepcopy(self.model.state_dict())
        self.model.train()

        for _ in range (self.params.local_rounds):
            for images, labels in self.train_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward(retain_graph=True)
                if total_global_delta is not None:
                    # 获取梯度向量 g
                    local_grad = torch.cat([
                        param.grad.detach().flatten()
                        for param in self.model.parameters() if param.grad is not None
                    ])

                    delta_alignment = torch.sum((local_grad*self.params.learning_rate - total_global_delta)**2)
                    self.optimizer.zero_grad() 
                    loss += delta_alignment * self.params.mu / 2
                    loss.backward()

                self.optimizer.step()

        self.local_round += 1

Client = FedGCRClient
Server = FedGCRServer