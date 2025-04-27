import copy
import logging
import math
import random
import torch
import torch.nn.functional as F

from .fedbuff import FedBuffClient, FedBuffServer

class FedProxClient(FedBuffClient):
    def __init__(self, client_id, base_model, data_loaders, recorder, params, speed_factor):
        super().__init__(client_id, base_model, data_loaders, recorder, params, speed_factor)

    def local_train(self):
        # Scaffold本地训练
        self.model.train()
        global_weights = copy.deepcopy(self.model.state_dict())
        total_global_weights = torch.cat([global_weights[key].float().flatten() for key in global_weights.keys() if 'bn' not in key])
        for _ in range(self.params.local_rounds):
            for images, labels in self.train_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                # 计算proximal term
                tmp = torch.cat([self.model.state_dict()[key].float().flatten()
                                 for key in self.model.state_dict().keys() if 'bn' not in key])
                proximal_term = torch.sum((tmp - total_global_weights)**2)  / (2 * self.params.mu)
                loss += proximal_term

                loss.backward()
                self.optimizer.step()

Client = FedProxClient
Server = FedBuffServer
