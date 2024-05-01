import torch
from torch import nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, skip=False):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            torch.nn.init.normal_(self.bias, mean=0.0, std=0.1)
        else:
            self.register_parameter('bias', None)

        if skip:
            self.skip = nn.Parameter(torch.FloatTensor(out_ft))
            torch.nn.init.normal_(self.skip, mean=0.0, std=0.1)
        else:
            self.register_parameter('skip', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False, skip=False):
        seq_fts = self.fc(seq)

        if skip:
            skip_out = seq_fts * self.skip

        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)

        if skip:
            out += skip_out

        if self.bias is not None:
            out += self.bias

        return out