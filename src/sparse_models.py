import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb

from base_models import NeuralNetwork, ParallelNetworks


def get_activation(activation="relu"):
    if activation == "relu":
        return F.relu
    elif activation == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    else:
        raise NotImplementedError


class SparseDecoder(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_point=41,
                 activation="relu", normalize_attn=True, mlp=True, layernorm=True):
        super(SparseDecoder, self).__init__()
        self.name = f"SparseDecoder_embd={n_embd}_layer={n_layer}_head={n_head}"

        # configs
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.activation = get_activation(activation)
        self.normalize_attn = normalize_attn
        self.layernorm = layernorm
        self.mlp = mlp
        self.n_point = n_point
        # layers
        self._read_in = nn.Linear(n_dims, n_embd)
        
        self._queries = nn.ModuleList()
        self._keys = nn.ModuleList()
        self._values = nn.ModuleList()
        self._mlps = nn.ModuleList()
        self._lns_1 = nn.ModuleList()
        self._lns_2 = nn.ModuleList()

        for i in range(n_layer):
            if i != n_layer-1:
                self._queries.append(nn.Linear(n_dims-1,self.n_head*(n_dims-1), bias=False))
                self._keys.append(nn.Linear(n_dims-1, self.n_head*(n_dims-1), bias=False))
                self._values.append(nn.Linear(n_dims-1, self.n_head*(n_dims-1), bias=False))
                self._lns_1.append(nn.LayerNorm([self.n_embd]))
                self._mlps.append(
                    nn.Sequential(
                        nn.Linear(n_embd, 4*n_embd),
                        nn.ReLU(),
                        nn.Linear(4*n_embd, n_embd),
                    )
                )
                self._lns_2.append(nn.LayerNorm([self.n_embd]))
            else:
                self._queries.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
                self._keys.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
                self._values.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
                self._lns_1.append(nn.LayerNorm([self.n_embd]))
                self._mlps.append(
                    nn.Sequential(
                        nn.Linear(n_embd, 4*n_embd),
                        nn.ReLU(),
                        nn.Linear(4*n_embd, n_embd),
                    )
                )
                self._lns_2.append(nn.LayerNorm([self.n_embd]))

        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        xs_e = torch.cat((xs_b, torch.zeros(bsize,points,1).cuda()), dim=2)
        ys_b_wide = torch.cat(
            (
                xs_b,
                ys_b.view(bsize, points, 1),
            ),
            axis=2,
        )
        zs = torch.stack((xs_e, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim+1)
        return zs

    def forward(self, xs, ys, head_mask, inds=None, return_hidden_states=False):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        bs, n_points, _ = zs.shape
        hidden_states = []
        
        H = self._read_in(zs)
        hidden_states.append(H)
        z1 = torch.zeros(bs, self.n_head, n_points, self.n_dims).cuda()
        z1.requires_grad = False
        z2 = torch.zeros(bs, self.n_head, n_points, 1).cuda()
        z2.requires_grad = False
        for idx, (q, k, v, ln1, mlp, ln2) in enumerate(zip(
            self._queries, self._keys, self._values,
            self._lns_1, self._mlps, self._lns_2,
        )):
            if idx != self.n_layer-1:
                query = q(H[:,:,self.n_dims:(2*self.n_dims-1)])
                key = H[:,:,0:self.n_dims-1]
                key = key.repeat(1,1,self.n_head)
                value = v(H[:,:,0:self.n_dims-1])

                query = query.view(bs, n_points, self.n_head, self.n_dims-1).permute(0, 2, 1, 3) 
                key = key.view(bs, n_points, self.n_head, self.n_dims-1).permute(0, 2, 1, 3) 
                value = value.view(bs, n_points, self.n_head, self.n_dims-1).permute(0, 2, 1, 3) 

                query = torch.cat((z1, query, z2), dim = 3)* head_mask
                key = torch.cat((z1, key, z2), dim = 3)* head_mask
                value = torch.cat((z1, value, z2), dim = 3)* head_mask
            else:
                query = q(H)
                key = k(H)
                value = v(H)

                query = query.view(bs, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) * head_mask
                key = key.view(bs, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) * head_mask
                value = value.view(bs, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) * head_mask

            attn_weights =self.activation(torch.einsum('abid,abjd->abij', query, key))


            maskNorm = torch.zeros(n_points, n_points)
            maskNorm.requires_grad = False
            for i in range(1, n_points+1):
                maskNorm[i-1, :i] = 1/i
            maskNorm  = maskNorm.unsqueeze(0).repeat(self.n_head, 1, 1)

            attn_weights = attn_weights * maskNorm.cuda()
            attn_weights = torch.einsum('abij,abjd->abid', attn_weights, value)

            attn_weights = torch.sum(attn_weights, dim=1)


            H = H + attn_weights
            if self.layernorm:
                H = ln1(H)

            if self.mlp:
                H = H + mlp(H)
                if self.layernorm:
                    H = ln2(H)

            hidden_states.append(H)

        prediction = self._read_out(H)
        # if return_hidden_states:
        #     return prediction[:, (n_points-ys.shape[1]):n_points, 0], hidden_states
        # return prediction[:, (n_points-ys.shape[1]):n_points, 0]
        if return_hidden_states:
            return prediction[:,::2, 0], hidden_states
        return prediction[:, ::2, 0]
    

class DiagDecoder(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_point=41,
                 activation="relu", normalize_attn=True, mlp=True, layernorm=True):
        super(DiagDecoder, self).__init__()
        self.name = f"DiagDecoder_embd={n_embd}_layer={n_layer}_head={n_head}"

        # configs
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.activation = get_activation(activation)
        self.normalize_attn = normalize_attn
        self.layernorm = layernorm
        self.mlp = mlp
        self.n_point = n_point
        # layers
        self._read_in = nn.Linear(n_dims, n_embd)
        
        self._queries = nn.ModuleList()
        self._keys = nn.ModuleList()
        self._values = nn.ModuleList()
        self._mlps = nn.ModuleList()
        self._lns_1 = nn.ModuleList()
        self._lns_2 = nn.ModuleList()
        for i in range(n_layer):
            self._queries.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
            self._keys.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
            self._values.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd]))
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(n_embd,4*n_embd),
                    nn.ReLU(),
                    nn.Linear(4*n_embd, n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))

        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        xs_e = torch.cat((xs_b, torch.zeros(bsize,points,1).cuda()), dim=2)
        ys_b_wide = torch.cat(
            (
                xs_b,
                ys_b.view(bsize, points, 1),
            ),
            axis=2,
        )
        zs = torch.stack((xs_e, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim+1)
        return zs

    def forward(self, xs, ys, head_mask, inds=None, return_hidden_states=False):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        n_batch, n_points, _ = zs.shape
        hidden_states = []
        
        H = self._read_in(zs)
        hidden_states.append(H)
        for (q, k, v, ln1, mlp, ln2) in zip(
            self._queries, self._keys, self._values,
            self._lns_1, self._mlps, self._lns_2,
        ):
            query = q(H)
            key = k(H)
            value = v(H)

            query = query.view(n_batch, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) * head_mask
            key = key.view(n_batch, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) * head_mask
            value = value.view(n_batch, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) * head_mask

            # attn_weights = self.activation(torch.einsum('bid,bjd->bij', query, key))
            attn_weights =self.activation(torch.einsum('abid,abjd->abij', query, key))


            maskNorm = torch.zeros(n_points, n_points)
            maskNorm.requires_grad = False
            for i in range(1, n_points+1):
                maskNorm[i-1, :i] = 1/i
            maskNorm  = maskNorm.unsqueeze(0).repeat(self.n_head, 1, 1)

            attn_weights = attn_weights * maskNorm.cuda()
            attn_weights = torch.einsum('abij,abjd->abid', attn_weights, value)

            attn_weights = torch.sum(attn_weights, dim=1)


            H = H + attn_weights
            if self.layernorm:
                H = ln1(H)

            if self.mlp:
                H = H + mlp(H)
                if self.layernorm:
                    H = ln2(H)

            hidden_states.append(H)

        prediction = self._read_out(H)
        # if return_hidden_states:
        #     return prediction[:, (n_points-ys.shape[1]):n_points, 0], hidden_states
        # return prediction[:, (n_points-ys.shape[1]):n_points, 0]
        if return_hidden_states:
            return prediction[:,::2, 0], hidden_states
        return prediction[:, ::2, 0]
