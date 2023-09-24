import argparse
import time
from utils import *
from tqdm import tqdm
from torch import optim
from model import *
from layers import *
from sklearn.decomposition import PCA


# parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=5, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=128, help='hidden_num')
parser.add_argument('--dims', type=int, default=800, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.5, help='Loss balance parameter')
parser.add_argument('--beta', type=float, default=0.5, help='Loss balance parameter')
parser.add_argument('--threshold', type=float, default=0.95, help='the threshold')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='number of clusters.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# parameter settings
if args.dataset == 'cora':
    args.gnnlayers = 5
    args.dims = 800
    args.lr = 1e-5

elif args.dataset == 'amap':
    args.gnnlayers = 5
    args.dims = 500
    args.lr = 1e-3

elif args.dataset == 'citeseer':
    args.gnnlayers = 7
    args.dims = 1500
    args.lr = 1e-3

elif args.dataset == 'bat':
    args.gnnlayers = 7
    args.dims = 50
    args.lr = 1e-3

elif args.dataset == 'eat':
    args.gnnlayers = 10
    args.dims = 100
    args.lr = 1e-7

elif args.dataset == 'uat':
    args.gnnlayers = 4
    args.dims = 100
    args.lr = 1e-3



acc_list = []
nmi_list = []
ari_list = []
f1_list = []

for seed in range(10):

    setup_seed(seed)

    adj, features, true_labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    pca = PCA(n_components=args.dims)
    features = pca.fit_transform(features)
    features = torch.FloatTensor(features)

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_tensor = torch.tensor(adj.todense(), dtype=torch.float32)

    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    sm_fea_s = torch.FloatTensor(sm_fea_s)


    # reversible network
    reversible_net = reversible_model([features.shape[1]])
    reversible_net = reversible_net.cuda()
    optimizer_reversible_net = optim.SGD(reversible_net.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0)
    reversible_net.train()

    # encoder network
    model = Encoder_Net([features.shape[1]] + [args.dims], args.cluster_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()


    # init
    best_acc, best_nmi, best_ari, best_f1, predict_labels, centers, dis = clustering(sm_fea_s, true_labels, args.cluster_num)

    # GPU
    if args.cuda:
        model.cuda()
        sm_fea_s = sm_fea_s.cuda()
        reversible_net.cuda()

    print('Start Training...')

    best_acc = 0

    start = time.time()
    for epoch in tqdm(range(args.epochs)):

        optimizer_reversible_net.zero_grad()
        optimizer.zero_grad()

        # obtain augmented feature
        aug_feature = reversible_net(sm_fea_s, True)

        # obtain embedding
        z1, logits_z1 = model(sm_fea_s)
        z2, logits_z2 = model(aug_feature)

        # reversible embedding
        z11 = reversible_net(z1, True)
        z22 = reversible_net(z2, False)

        # contrastive
        loss_1 = loss_cal(z1, z22)
        loss_2 = loss_cal(z2, z11)
        contra_loss = loss_1 + loss_2

        # sim loss
        cross_sim_ori_pro = z1 * z11
        cross_sim_re_ori = z22 * z1
        loss_semantic = F.mse_loss(cross_sim_ori_pro, cross_sim_re_ori)


        if epoch > 200:
            # label matching
            # z1 and z2 semantic labels
            pseudo_z1 = torch.softmax(logits_z1, dim=-1)
            pseudo_z2 = torch.softmax(logits_z2, dim=-1)


            z_cluster = (z1 + z2) / 2
            _, _, _, _, predict_labels,centers, dis = clustering(z_cluster, true_labels, args.cluster_num)
            high_confidence = torch.min(dis, dim=1).values.cpu()
            threshold = torch.sort(high_confidence).values[int(len(high_confidence) * args.threshold)]
            high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
            h_i = high_confidence_idx.numpy()
            y_sam = torch.tensor(predict_labels, device=args.device)[high_confidence_idx]



            loss_match = (F.cross_entropy(pseudo_z1[h_i], y_sam)).mean() + (F.cross_entropy(pseudo_z2[h_i], y_sam)).mean()

            # total loss
            total_loss = contra_loss + args.alpha * loss_semantic + args.beta * loss_match

        else:

            total_loss = contra_loss + args.alpha * loss_semantic

        total_loss.backward()
        optimizer_reversible_net.step()
        optimizer.step()


        # test stage
        if epoch % 5 == 0:
            model.eval()
            z1, p = model(sm_fea_s)
            z2, p = model(aug_feature)
            hidden_emb = (z1 + z2) / 2
            acc, nmi, ari, f1, predict_labels,centers, dis = clustering(hidden_emb, true_labels, args.cluster_num)
            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
                best_embed = hidden_emb
    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

    tqdm.write("Optimization Finished!")
    tqdm.write('best_acc: {}, best_nmi: {}, best_ari: {}, best_f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)
print(acc_list.mean(), "±", acc_list.std())
print(nmi_list.mean(), "±", nmi_list.std())
print(ari_list.mean(), "±", ari_list.std())
print(f1_list.mean(), "±", f1_list.std())