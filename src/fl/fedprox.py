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
        self.model.train()
        total_global_weights = torch.cat([
            param.data.float().flatten()
            for name, param in self.model.named_parameters()
        ])
        for _ in range(self.params.local_rounds):
            for images, labels in self.train_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                # 计算proximal term
                tmp = torch.cat([
                    param.detach().float().flatten()
                    for name, param in self.model.named_parameters()
                ])
                proximal_term = self.params.mu * torch.sum((tmp - total_global_weights)**2)  / 2
                loss += proximal_term

                loss.backward()
                self.optimizer.step()


Client = FedProxClient
Server = FedBuffServer
