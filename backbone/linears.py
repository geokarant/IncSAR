import math
import torch
from torch import nn
from torch.nn import functional as F


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}
    
class custom_cnn(nn.Module):
    def __init__(self, in_features= 3, out_features = 10):
        super(custom_cnn, self).__init__()
        self.in_features = in_features
        self.out_features= out_features
        self.conv16 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7)
        self.conv32 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.4)
        self.flatten=  nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv16(x))
        x = self.pool(x)
        x = F.relu(self.conv32(x))
        x = self.pool(x)
        x = F.relu(self.conv64(x))
        x = self.pool(x)
        x = F.relu(self.conv128(x))
        x = self.dropout(x)
        x= self.flatten(x)
        return x


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)

class attention_layer(nn.Module):
    def __init__(self,
        emb_dim=672,
        tf_layers=4, tf_head=8, tf_dim=336,
        activation="gelu", dropout=0.1,
        pre_norm=True):
        super(attention_layer, self).__init__()

        self.emb_dim = emb_dim

        self.cls_token = nn.Parameter(torch.randn(self.emb_dim))
        self.cls_token.requires_grad = True

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=tf_head,
                dim_feedforward=tf_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=pre_norm,
            ),
            num_layers=tf_layers,
        )

    def forward(self, x_vit, x_cnn):
        x = torch.cat([x_vit, x_cnn], dim=1)
        b_size = x.shape[0]
        x = x.reshape(b_size, -1, self.emb_dim)
        cls_token = self.cls_token.expand(b_size, 1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.transformer(x)

        x = x[:,0,:]
        return x

class attention_layer_mac(nn.Module):
    def __init__(self,
        emb_dim=672,
        tf_layers=4, tf_head=8, tf_dim=336,
        activation="gelu", dropout=0.1,
        pre_norm=True):
        super(attention_layer_mac, self).__init__()

        self.emb_dim = emb_dim

        self.cls_token = nn.Parameter(torch.randn(self.emb_dim))
        self.cls_token.requires_grad = True

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=tf_head,
                dim_feedforward=tf_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=pre_norm,
            ),
            num_layers=tf_layers,
        )

    def forward(self, x):
        #x = torch.cat([x_vit, x_cnn], dim=1)
        b_size = x.shape[0]
        x = x.reshape(b_size, -1, self.emb_dim)
        cls_token = self.cls_token.expand(b_size, 1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.transformer(x)

        x = x[:,0,:]
        return x
class BEAR_WWTY(nn.Module):
    # L = W W^T Y modeling
    def __init__(self, inp, k=1):
        """
        Just contain one linear layer.

        The weight of this layer is the W in the paper.
        """
        super().__init__()
        self.ln = nn.Linear(inp, k, bias=False)

    def forward(self, x):
        L = self.ln(x) @ self.ln.weight
        return L

    def clamper(self):
        for p in self.parameters():
            p.data.clamp_(0.0)

class BEAR_ABY(nn.Module):
    # L = A B Y
    def __init__(self, inp, k=1):
        """
        Use two weight matrix A, B.

        This is relaxed version of BEAR_WWTY,
        which imposes the transpose relationship.

        The weight of this layer is the W in the paper.
        """
        super().__init__()
        self.ln1 = nn.Linear(inp, k, bias=False)
        self.ln2 = nn.Linear(k, inp, bias=False)

    def forward(self, x):
        L = self.ln2(self.ln1(x))
        return L

    def clamper(self):
        for p in self.parameters():
            p.data.clamp_(0.0)

