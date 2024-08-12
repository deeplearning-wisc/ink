from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class InkPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(InkPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.num_classes = self.config.dataset.num_classes
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        net.eval()
        features, labels = [], []
        id_loader = id_loader_dict['train']
        id_loader = torch.utils.data.DataLoader(
            id_loader.dataset, batch_size=id_loader.batch_size, shuffle=True, num_workers=id_loader.num_workers
        )
        count = 0
        with torch.no_grad():
            for batch in tqdm(id_loader, desc='Setup: ', position=0, leave=True):
                data = batch['data'].cuda()
                features.append(net.intermediate_forward(data))
                labels.append(batch['label'].cuda())
                count += 1
                if count == 100:
                    break
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        self.prototypes = torch.zeros(self.num_classes, features.shape[1]).cuda()
        for i in range(self.num_classes):
            self.prototypes[i] = torch.mean(features[labels == i], dim=0)
            self.prototypes[i] = self.prototypes[i] / torch.norm(self.prototypes[i], p=2)
        print(self.prototypes)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        features = net.intermediate_forward(data)
        logits = torch.matmul(features, torch.transpose(self.prototypes, 0, 1)) / self.temperature
        energy = self.temperature * torch.logsumexp(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        return pred, energy

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]

    def get_hyperparam(self):
        return self.temperature