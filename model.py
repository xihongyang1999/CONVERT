from layers import *


class Encoder_Net(nn.Module):
    def __init__(self, dims, cluster_num):
        super(Encoder_Net, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.low = nn.Linear(dims[1], cluster_num)

    def forward(self, x):
        out1 = self.layers1(x)
        out1 = F.normalize(out1, dim=1, p=2)
        logits = self.low(out1)
        logits = F.normalize(logits, dim=1, p=2)
        return out1, logits



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, cluster_num):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.low = nn.Linear(out, cluster_num)
        # self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, 0.1, training = self.training)
        x = self.gc2(x, adj)
        logits = self.low(x)
        logits = F.normalize(logits, dim=1, p=2)
        return x, logits

    def get_emb(self, x, adj):
        return F.relu(self.gc1(x, adj)).detach()



#reversible network
class reversible_model(nn.Module):
    def __init__(self, dims):
        super(reversible_model, self).__init__()

        self.down1 = nn.Linear(dims[0], dims[0]//2)
        self.down2 = nn.Linear(dims[0]//2, dims[0])

        self.up1 = nn.Linear(dims[0], dims[0] * 2)
        self.up2 = nn.Linear(dims[0] * 2, dims[0])

    def forward(self, x, flag):

        if flag:
            down_feature = self.down2(self.down1(x))
            down_feature = F.normalize(down_feature, dim=1, p=2)
            return down_feature


        else:
            up_feature = self.up2(self.up1(x))
            up_feature = F.normalize(up_feature, dim=1, p=2)
            return up_feature


def loss_cal(x, x_aug):
    T = 1.0
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss