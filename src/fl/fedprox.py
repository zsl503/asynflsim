import copy
import torch
import torch.nn.functional as F

from .fedbuff import FedBuffClient, FedBuffServer

class FedProxClient(FedBuffClient):
    def __init__(self, client_id, base_model, data_loaders, recorder, params, speed_factor):
        super().__init__(client_id, base_model, data_loaders, recorder, params, speed_factor)
        self.local_train = self.local_train_prox

    def local_train_prox(self):
        self.model.train()
        global_model = copy.deepcopy(self.model)
        for _ in range(self.params.local_rounds):
            for images, labels in self.train_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                # 计算proximal term
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss += (self.params.mu / 2) * proximal_term

                loss.backward()
                self.optimizer.step()
Client = FedProxClient
Server = FedBuffServer
