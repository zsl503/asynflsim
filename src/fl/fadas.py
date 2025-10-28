import copy
import logging
import math
import random
import time

from .fedbuff import FedBuffServer, FedBuffClient
import torch
def layer_can_select(key):
    if 'bn' in key or 'downsample' in key or 'bias' in key:
        return False
    return True

class FADASServer(FedBuffServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)
        self.buffer_size = params.buffer_size
        self.m = {key: torch.zeros_like(value, device=self.device).float() for key, value in model.state_dict().items()}
        self.v = {key: torch.zeros_like(value, device=self.device).float() for key, value in model.state_dict().items()}
        self.v_hat = {key: torch.zeros_like(value, device=self.device).float() for key, value in model.state_dict().items()}
        self.beta1 = params.beta1
        self.beta2 = params.beta2

    def aggregate(self):
        iteration = self.model_version
        avg_delta = {}

        # 输出 staleness 信息
        logging.info(f"Staleness: {[self.model_version - model_version for _, _, _, model_version in self.buffer]}")
        print(f"Staleness: {[self.model_version - model_version for _, _, _, model_version in self.buffer]}")

        # 加权平均 delta
        for key in self.global_model.state_dict().keys():
            layer_sum = torch.stack([delta[key].float() for delta, _, _, _ in self.buffer]).sum(0)
            avg_delta[key] = layer_sum / len(self.buffer)
            # avg_delta[key] = torch.clamp(avg_delta[key], min=-10, max=10)


        # lr_decay = 1.0 / math.sqrt(iteration)
        # # 计算偏差校正系数
        bias_correction1 = max(1 - self.beta1 ** iteration, 1e-1)
        # bias_correction2 = max(1 - self.beta2 ** iteration, 1e-1)
        
        current_weights = self.global_model.state_dict()
        new_weights = {}

        # # 更新动量项
        for key in avg_delta:
            if not layer_can_select(key):
                new_weights[key] = current_weights[key] - self.params.server_lr * avg_delta[key]
                continue
            grad = avg_delta[key]
            grad_sq = grad * grad

            # 一阶动量更新
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad

            # 二阶动量更新（复数安全）
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad_sq

            # # AMSGrad：v_hat 取最大
            self.v_hat[key] = self.v[key]
            self.v_hat[key] = torch.maximum(self.v_hat[key], self.v[key])

            # else:
            # denom = 1
            # if self.model_version < 20:
            #     p = 1e-1
            # else:
            #     p = 1e-8
            denom = self.v_hat[key].sqrt().add(1e-8)
            # corrected_m = self.m[key] / 0.1
            step_size = self.params.server_lr
            # step_size = self.params.server_lr * lr_decay / bias_correction1
            new_weights[key] = current_weights[key] - step_size * self.m[key] / denom
            # new_weights[key] = current_weights[key] - step_size * self.m[key]
            # new_weights[key] = current_weights[key] - self.params.server_lr * self.m[key]

        # 加载新模型权重
        self.global_model.load_state_dict(new_weights)

        self.buffer.clear()
        self.aggregation_count += 1
        self.model_version += 1

        self.check_and_validate()
        self.recorder.record_aggregation(self.env.now, self.model_version)
        self.recorder.aggregation_times.append(self.env.now)
        self.check_stop_condition()

Client = FedBuffClient
Server = FADASServer