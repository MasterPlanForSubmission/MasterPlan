# -*- encoding: utf-8 -*-
'''
@File      : GRU.py
@Describe  : GRU for generating state embedding
@Time      : 2024/01/05 15:48:26
@Author    : xinqichen
'''

import numpy as np
import torch
import torch.nn as nn
from algorithm.algo_config import GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, EMBEDDING_DIM


class GRUEmbedding(nn.Module):
    def __init__(self, N, K, num_channels, embedding_dim=EMBEDDING_DIM):
        super(GRUEmbedding, self).__init__()
        # N-way (N classes, N = 1 for non-classification tasks), K-shot (K samples used to generate per embedding)
        self.N = N
        self.K = K
        self.embedding_dim = embedding_dim
        self.bidirectional = True
        self.directions = 2 if self.bidirectional else 1
        self.net = nn.GRU(num_channels, GRU_HIDDEN_SIZE, num_layers=GRU_NUM_LAYERS, batch_first=True, bidirectional=self.bidirectional)
        self.embedding_layer = nn.Linear(GRU_HIDDEN_SIZE * GRU_NUM_LAYERS * self.directions, embedding_dim)
        self.relu = nn.ReLU()

    def init_hidden(self, num_sequences):
        return torch.zeros((self.directions * GRU_NUM_LAYERS, num_sequences, GRU_HIDDEN_SIZE))

