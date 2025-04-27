import copy
import logging
import math
import torch

from .fedbuff import FedBuffServer, FedBuffClient

class FedACServer(FedBuffServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)
        self.history_buffer_size = self.params.history_buffer_size
        self.model_buffer = [self.global_model.state_dict()]
        
    def aggregate(self):
        def staleness(delta, model_version, key):
            # 计算每个客户端的过时程度
            delta_version = self.model_version - model_version
            if delta[key].dtype == torch.long or delta_version == 0:
                return 1
            # logging.info(f"self.model_buffer[-1-delta_version][key]: {self.model_buffer[-1-delta_version][key]}")
            # logging.info(f"self.model_buffer[-1][key]: {self.model_buffer[-1][key]}")
            # 判断self.model_buffer[-1-delta_version][key]是否等于self.model_buffer[-1][key]
            # if torch.equal(self.model_buffer[-1-delta_version][key], self.model_buffer[-1][key]):
            #     logging.info(f"{key}:self.model_buffer[-1-delta_version][key] == self.model_buffer[-1][key]")
            g_delta = (self.model_buffer[-1-delta_version][key] - self.model_buffer[-1][key]).view(-1)
            l_delta = delta[key].view(-1)
            # g_normal = torch.norm(g_delta, p=2)
            # l_normal = torch.norm(l_delta, p=2)
            # logging.info(f"delta_version: {delta_version} / {len(self.model_buffer)}, g_normal: {g_normal}, l_normal: {l_normal}")
            # 返回余弦相似度
            return torch.cosine_similarity(g_delta, l_delta, dim=0).item() if g_delta.norm() != 0 and l_delta.norm() != 0 else 1
        
        avg_delta = {}
        for key in self.global_model.state_dict().keys():
            staleness_list = [staleness(delta, model_version, key) for delta, _, _, model_version in self.buffer]
            sum_staleness = sum(staleness_list)
            logging.info(f"Staleness for {key}: {staleness_list}")
            avg_delta[key] = torch.stack(
                [delta[key].float() * staleness_list[idnex] / sum_staleness for idnex, (delta, _, _, model_version) in enumerate(self.buffer)]
            ).sum(0)

        # 应用全局更新（含服务器学习率）
        current_weights = self.global_model.state_dict()
        new_weights = {
            k: current_weights[k] - self.params.server_lr * avg_delta[k]
            for k in current_weights
        }
        self.global_model.load_state_dict(new_weights)

        self.model_buffer.append(copy.deepcopy(self.global_model.state_dict()))
        if len(self.model_buffer) > self.history_buffer_size:
            self.model_buffer.pop(0)
            
        self.buffer.clear()
        self.aggregation_count += 1
        self.model_version += 1
        
        self.check_and_validate()
        self.recorder.record_aggregation(self.env.now, self.model_version)
        self.recorder.aggregation_times.append(self.env.now)
        self.check_stop_condition()

Client = FedBuffClient
Server = FedACServer  
