import torch
import torch.nn as nn
import torch.nn.functional as F


class FCUnit(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(FCUnit, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.batchnorm = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = F.elu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)

        return x


class Network(nn.Module):
    def __init__(self, params):

        super(Network, self).__init__()

        embedding_sizes = params['embedding_sizes']
        n_continous = len(params['continuous'])
        n_categorical = len(params['categorical'])
        size_fc = params['model']['size_fc']
        size_final = params['model']['size_final']
        dropout_emb = params['model']['dropout_emb']
        dropout_fc = params['model']['dropout_fc']

        # prepend combined cont/cat input vector size
        # append final linear layer size
        layer_sizes = [n_continous + n_categorical] + size_fc + [size_final]
        # get pairs of input/output size
        fc_dims = [(x, y) for x, y in zip(layer_sizes, layer_sizes[1:])]

        # ----- Categorical Inputs -----
        # create categorical embeddings
        self.embeddings = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in embedding_sizes]
        )
        # create dropout for embeddings
        self.dropout_emb = nn.Dropout(dropout_emb)

        # ----- Continuous Inputs -----
        # batchnorm for continuous
        self.bn_continuous = nn.BatchNorm1d(n_continous)

        # ----- Fully Connected (FC) -----
        self.fc_layers = nn.ModuleList(
            [FCUnit(in_size, out_size, dropout_fc) for in_size, out_size in fc_dims]
        )

        self.output_linear = nn.Linear(size_final, 1)

    def forward(self, cont_data, cat_data):
        x_cont = self.bn_continuous(cont_data)

        x_cat = [embedding(cat_data[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = self.dropout_emb(x_cat)

        x = torch.cat([x_cont, x_cat], 1)
        x = self.fc_layers(x)
        x = self.output_linear(x)
        x = F.sigmoid(x)

        return x

