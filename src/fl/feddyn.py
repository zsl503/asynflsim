# t2是要33行时间系数不要50行（），t3是不要33行时间系数要50行（非常平稳，但是精度不及其它，比fedbuff好），t4是33行和105行时间系数不要50行（后期乏力）
import copy
import logging
import math
import torch
import torch.nn.functional as F

from .fedavg import FedAvgServer, FedAvgClient

class FedDynServer(FedAvgServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)

    def aggregate(self):
        total_samples = sum(samples for _, samples, _, _ in self.buffer)
        avg_delta = {}
        for key, value in self.global_model.state_dict().items():
            layer_sum = torch.stack(
                [delta[key] * samples for delta, samples, _, model_version in self.buffer]
            ).sum(0)
            avg_delta[key] = layer_sum / total_samples

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
        return (self.model_version, self.global_model.state_dict())

class FedDynClient(FedAvgClient):
    def __init__(self, client_id, base_model, data_loaders, recorder, params, speed_factor):
        super().__init__(client_id, base_model, data_loaders, recorder, params, speed_factor)
        self.old_delta = None
        self.last_global_params = None
        self.local_train = self.gcr_local_train_minibatch
        self.prev_grads = None
        for param in self.model.parameters():
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = torch.zeros_like(param.view(-1))
            else:
                self.prev_grads = torch.cat((self.prev_grads, torch.zeros_like(param.view(-1))), dim=0)

    def gcr_local_train_minibatch(self):
        self._init_optimizer()
        self.model.train()
        global_state_dict = copy.deepcopy(self.model.state_dict())
        for _ in range (self.params.local_rounds):
            for images, labels in self.train_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)

                # lin_penalty = 0.0
                # curr_params = None
                # for name, param in self.model.named_parameters():
                #     if not isinstance(curr_params, torch.Tensor):
                #         curr_params = param.view(-1)
                #     else:
                #         curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

                # lin_penalty = torch.sum(curr_params * self.prev_grads)
                # loss -= lin_penalty

            # 添加 FedProx proximal term
                prox_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_state_dict:
                        prox_term += ((param - global_state_dict[name].to(param.device)) ** 2).sum()
                loss += (self.params.mu / 2) * prox_term
                # if prox_term > 0:
                #     print(f"[FedProx] Proximal Term: {prox_term.item():.4f}")
                loss.backward()
                # Update the previous gradients
                self.prev_grads = None
                for param in self.model.parameters():
                    if not isinstance(self.prev_grads, torch.Tensor):
                        self.prev_grads = param.grad.view(-1).clone()
                    else:
                        self.prev_grads = torch.cat((self.prev_grads, param.grad.view(-1).clone()), dim=0)

                self.optimizer.step()

Client = FedDynClient
Server = FedDynServer
