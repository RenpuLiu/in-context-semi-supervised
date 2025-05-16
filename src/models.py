import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
import math


from sparse_models import SparseDecoder
from base_models import NeuralNetwork, ParallelNetworks


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "SoftmaxEncoder":
        # backward compatible
        if 'encoder_activation' not in conf.keys():
            conf.encoder_activation = "softmax"
        if 'normalize_attn' not in conf.keys():
            conf.normalize_attn = True

        model = SoftmaxEncoder(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            n_point=conf.n_point,
            activation=conf.encoder_activation,
            normalize_attn=conf.normalize_attn,
        )
    elif conf.family == "LassoEncoder":
        # backward compatible
        if 'encoder_activation' not in conf.keys():
            conf.encoder_activation = "relu"
        if 'normalize_attn' not in conf.keys():
            conf.normalize_attn = True

        model = LassoEncoder(
            n_dims=conf.n_dims + 1,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            activation=conf.encoder_activation,
            normalize_attn=conf.normalize_attn,
        )
    elif conf.family == "ReluDecoder":
        # backward compatible
        if 'encoder_activation' not in conf.keys():
            conf.encoder_activation = "relu"
        if 'normalize_attn' not in conf.keys():
            conf.normalize_attn = True

        model = ReluDecoder(
            n_dims=conf.n_dims+1,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            n_point=conf.n_point,
            #n_point=conf.training.curriculum.points.end,
            activation=conf.encoder_activation,
            normalize_attn=conf.normalize_attn,
        )
    elif conf.family == "SparseDecoder":
        if 'encoder_activation' not in conf.keys():
            conf.encoder_activation = "relu"
        if 'normalize_attn' not in conf.keys():
            conf.normalize_attn = True

        model = SparseDecoder(
            n_dims=conf.n_dims+1,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            n_point=conf.n_point,
            #n_point=conf.training.curriculum.points.end,
            activation=conf.encoder_activation,
            normalize_attn=conf.normalize_attn,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs


def get_activation(activation="relu"):
    if activation == "relu":
        return F.relu
    elif activation == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    else:
        raise NotImplementedError


class SubBlockLinear(nn.Module):
    def __init__(self, in_features, out_features, sub_in_features, sub_out_features, bias=False):
        super(SubBlockLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sub_in_features = sub_in_features
        self.sub_out_features = sub_out_features

        # Define the trainable sub-block weight
        self.sub_weight = nn.Parameter(torch.Tensor(sub_out_features, sub_in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the sub-block weight
        nn.init.kaiming_uniform_(self.sub_weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # Compute the part of the input that corresponds to the sub-block
        input_sub = input[..., :self.sub_in_features]
        # Compute the output for the sub-block
        output_sub = F.linear(input_sub, self.sub_weight)
        # Initialize the full output tensor with zeros
        output = torch.zeros(*input.shape[:-1], self.out_features, device=input.device, dtype=input.dtype)
        # Place the sub-block output into the upper-left corner of the output tensor
        output[..., :self.sub_out_features] = output_sub
        # Add bias if applicable
        if self.bias is not None:
            output += self.bias
        return output
    
# Old version of SoftmaxEncoder

# class SoftmaxEncoder(nn.Module):
#     def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_point=11,
#                  activation="softmax", normalize_attn=True, mlp=True, layernorm=True):
#         super(SoftmaxEncoder, self).__init__()
#         self.name = f"SoftmaxEncoder_embd={n_embd}_layer={n_layer}_head={n_head}"

#         # configs
#         self.n_positions = n_positions
#         self.n_dims = n_dims
#         self.n_embd = n_embd
#         self.n_head = n_head
#         self.n_layer = n_layer
#         self.activation = get_activation(activation)
#         self.normalize_attn = normalize_attn
#         self.layernorm = layernorm
#         self.mlp = mlp
#         self.n_point = n_point
#         # layers
#         self._read_in = nn.Linear(n_embd, n_embd)
        
#         self._queries = nn.ModuleList()
#         self._keys = nn.ModuleList()
#         self._values = nn.ModuleList()
#         self._mlps = nn.ModuleList()
#         self._lns_1 = nn.ModuleList()
#         self._lns_2 = nn.ModuleList()
#         for _ in range(n_layer):
#             self._queries.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
#             self._keys.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))

#             self._values.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
#             self._lns_1.append(nn.LayerNorm([self.n_embd]))
#             self._mlps.append(
#                 nn.Sequential(
#                     nn.Linear(n_embd, n_embd),
#                     nn.ReLU(),
#                     nn.Linear(n_embd, n_embd),
#                 )
#             )
#             self._lns_2.append(nn.LayerNorm([self.n_embd]))
#         self._read_out = nn.Linear(n_embd, n_embd)

#     @staticmethod
#     def _combine(xs_b, ys_b):
#         """
#         Directly stack the x's and y's into the same location
#         resulting sequence would be Bx(N+1)x(d+1), where (N+1)-th token is test
#         """
#         d = xs_b.size(-1)
#         # half_n = xs_b.size(1)//3
#         half_n = 5
#         zs = torch.cat((ys_b, xs_b), dim=2)
#         zs[:, half_n:, d:].zero_()
#         if xs_b.shape[1] < ys_b.shape[1]:
#             raise ValueError("Number of prompts in testing larger the training.")
#         return zs

#     def forward(self, xs, ys, head_mask, inds=None, return_hidden_states=False):
#         if inds is None:
#             inds = torch.arange(ys.shape[1])
#         else:
#             inds = torch.tensor(inds)
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")
#         zs = self._combine(xs, ys)
#         n_batch, n_points, _ = zs.shape
#         hidden_states = []
        
#         H = self._read_in(zs)
#         hidden_states.append(H)
#         for (q, k, v, ln1, mlp, ln2) in zip(
#             self._queries, self._keys, self._values,
#             self._lns_1, self._mlps, self._lns_2,
#         ):
#             query = q(H)
#             key = k(H)
#             value = v(H)
            
#             query = query.view(n_batch, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) 
#             key = key.view(n_batch, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) 
#             value = value.view(n_batch, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) 
#             attn_weights =self.activation(torch.einsum('abid,abjd->abij', query, key))



#             attn_weights = torch.einsum('abij,abjd->abid', attn_weights, value)

#             attn_weights = torch.sum(attn_weights, dim=1)

#             # if self.normalize_attn:
#             #     attn_weights = attn_weights/n_points
#             H = H + attn_weights

#             if self.layernorm:
#                 H = ln1(H)

#             if self.mlp:
#                 H = H + mlp(H)
#                 # if self.layernorm:
#                 #     H = ln2(H)

#             hidden_states.append(H)
#         prediction = self._read_out(H)
#         if return_hidden_states:
#             return prediction[:, :, self.n_dims:], hidden_states
#         return prediction[:, :, self.n_dims:]
#############################################################################

class SoftmaxEncoder(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_point=11, n_cot=5,
                 activation="softmax", normalize_attn=True, mlp=True, layernorm=True, return_cot=True):
        super(SoftmaxEncoder, self).__init__()
        self.name = f"SoftmaxEncoder_embd={n_embd}_layer={n_layer}_head={n_head}"

        # configs
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.activation = get_activation(activation)
        self.normalize_attn = normalize_attn
        self.return_cot = return_cot
        self.layernorm = layernorm
        self.mlp = mlp
        self.n_point = n_point
        self.n_cot = n_cot
        # layers
        self._read_in = nn.Linear(n_embd, n_embd) # Modifty the input dimension to 2*n_dims
        
        self._queries = nn.ModuleList()
        self._keys = nn.ModuleList()
        self._values = nn.ModuleList()
        self._mlps = nn.ModuleList()
        self._lns_1 = nn.ModuleList()
        self._lns_2 = nn.ModuleList()
        for _ in range(n_layer):
            self._queries.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
            self._keys.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))

            self._values.append(nn.Linear(n_embd, self.n_head*n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd]))
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(n_embd, n_embd),
                    nn.ReLU(),
                    nn.Linear(n_embd, n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))
        self._read_out = nn.Linear(n_embd, n_embd)


    def _combine(self, xs_b, ys_b):
        """
        Stack label vectors xs_b and data vectors ys_b along feature dim,
        zero‑out the unlabeled part, then append padding tokens.

        Resulting shape:  B × (N + self.n_dims) × (2d + d)  (see below)
        """
        device, dtype = xs_b.device, xs_b.dtype
        d = xs_b.size(-1)          # feature dim == #classes
        N = xs_b.size(1)
        B = xs_b.size(0)
        half_n  = 5                      # #labelled examples

        # 1. concatenate [y | x] -> (B,N,2d)
        zs = torch.cat((ys_b, xs_b), dim=2)

        # 2. zero unlabeled rows (positions half_n … N‑1)
        zs[:, half_n:, d:].zero_()       # keep only first d dims of xs

        # 3. append an all‑zero copy of xs_b for the test token
        zs = torch.cat((zs, xs_b.new_zeros(xs_b.size())), dim=2)   # (B,N,4d)
        zs = torch.cat((zs, xs_b.new_zeros(xs_b.size())), dim=2)

        
        # Here ################################################################
        zeros_pad = zs.new_zeros(zs.size(0), self.n_dims, zs.size(2))
        zs_appended = torch.cat([zs, zeros_pad], dim=1)

        eye = torch.eye(d, device=zs_appended.device)          
        zs_appended[:, N : N + d, 2*d : 3*d] = eye.unsqueeze(0).expand(B, -1, -1)
        # Here ################################################################

        if xs_b.shape[1] < ys_b.shape[1]:
            raise ValueError("Number of prompts in testing larger than training.")

        return zs_appended
        

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

        cot_list = []

        for r_step in range(self.n_cot):
            H_input = H.clone()
            # print("###############", H.shape)
            
            for (q, k, v, ln1, mlp, ln2) in zip(
                self._queries, self._keys, self._values,
                self._lns_1, self._mlps, self._lns_2,
            ):
                query = q(H)
                key = k(H)
                value = v(H)
                
                # Here ################################################################
                query = query.view(n_batch, n_points + (r_step)*self.n_dims, self.n_head, self.n_embd).permute(0, 2, 1, 3) 
                key = key.view(n_batch, n_points + (r_step)*self.n_dims, self.n_head, self.n_embd).permute(0, 2, 1, 3) 
                value = value.view(n_batch, n_points + (r_step)*self.n_dims, self.n_head, self.n_embd).permute(0, 2, 1, 3) 
                # Here ################################################################
                # query = query.view(n_batch, n_points , self.n_head, self.n_embd).permute(0, 2, 1, 3) 
                # key = key.view(n_batch, n_points , self.n_head, self.n_embd).permute(0, 2, 1, 3) 
                # value = value.view(n_batch, n_points , self.n_head, self.n_embd).permute(0, 2, 1, 3) 

                
                attn_weights =self.activation(torch.einsum('abid,abjd->abij', query, key))
                
                attn_weights = torch.einsum('abij,abjd->abid', attn_weights, value)

                attn_weights = torch.sum(attn_weights, dim=1)

                H = H + attn_weights

                if self.layernorm:
                    H = ln1(H)

                if self.mlp:
                    H = H + mlp(H)
                    
            # Here ################################################################
            H_cot = H[:, -self.n_dims:, :].clone()

            cot_list.append(H_cot)

            H = torch.cat([H_input, H_cot], dim=1)
            
            eye = torch.eye(self.n_dims, device=H.device)          # (d, d) on the right device
            H[:, -self.n_dims : , 2*self.n_dims : 3*self.n_dims] = eye.unsqueeze(0).expand(n_batch, -1, -1)
            # Here ################################################################

            


            hidden_states.append(H)
        prediction = self._read_out(H[:,:self.n_point, :])
        if self.return_cot:
            # Here ################################################################
            return prediction[:, :, self.n_dims:self.n_dims*2], [cot_list[i][:, :, -self.n_dims:] for i in range(self.n_cot)]
            # Here ################################################################
            # return prediction[:, :, self.n_dims:self.n_dims*2], []
        return prediction[:, :, self.n_dims:]
        

class CustomLinear(nn.Module):
    def __init__(self, d, n_head):
        super(CustomLinear, self).__init__()
        d = d-1
        self.d = d
        self.n_head = n_head
        self.n_pairs = self.n_head // 2  # Fixed integer division
        self.M_q = nn.Parameter(torch.randn(self.n_pairs, d, d))
        self.B_q = nn.Parameter(torch.randn(n_head, 1, 1))
        self.M_k = nn.Parameter(torch.randn(self.n_pairs, d, d))
        self.B_k = nn.Parameter(torch.randn(n_head, 1, 1))
        self.M_v = nn.Parameter(torch.randn(n_head, d, d))

    def forward_q(self, zs):  # Consistent variable name
        n_batch, n_points, n_embd = zs.size()
        d = (n_embd - 2) // 2  # Corrected division to integer
        Q = torch.zeros(self.n_head * n_embd, n_embd).to(zs.device)  # Fixed variable name
        for i in range(self.n_head):
            if i < self.n_pairs:
                Q[d + 1 + i * n_embd: 2*d + 1 + i * n_embd, d + 1: 2*d + 1] = self.M_q[i]  # Corrected multiplication
                
                Q[2*d + 1 + i * n_embd, 2*d + 1] = self.B_q[i]  # Corrected multiplication
            else:
                Q[d + 1 + i * n_embd, 2*d + 1] = self.B_q[i]  # Corrected multiplication

        q = torch.nn.functional.linear(zs, Q).to(zs.device)
        return q

    def forward_k(self, zs):  # Consistent variable name
        n_batch, n_points, n_embd = zs.size()
        d = (n_embd - 2) // 2  # Corrected division to integer
        K = torch.zeros(self.n_head * n_embd, n_embd).to(zs.device)  # Fixed variable name
        I = torch.eye(d).to(zs.device)  # Ensure tensor is on the same device
        for i in range(self.n_head):
            if i < self.n_pairs:
                K[d + 1 + i * n_embd: 2*d + 1 + i * n_embd, 0: d] = self.M_k[i]  # Corrected multiplication
                K[2*d + 1 + i * n_embd, 2*d + 1] = self.B_k[i]  # Corrected multiplication
            else:
                K[d + 1 + i * n_embd, d] = self.B_k[i]  # Corrected multiplication
        k = torch.nn.functional.linear(zs, K).to(zs.device)
        return k

    def forward_v(self, zs):  # Consistent variable name
        n_batch, n_points, n_embd = zs.size()
        d = (n_embd - 2) // 2  # Corrected division to integer
        V = torch.zeros(self.n_head * n_embd, n_embd).to(zs.device)  # Fixed variable name
        for i in range(self.n_head):
            V[d + 1 + i * n_embd: 2*d + 1 + i * n_embd, 0: d] = self.M_v[i]  # Corrected multiplication
        v = torch.nn.functional.linear(zs, V).to(zs.device)
        return v

    def forward(self, x):
        q = self.forward_q(x)
        k = self.forward_k(x)
        v = self.forward_v(x)
        return q, k, v
    

# class ReluDecoder(nn.Module):
#     def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_point=41,
#                  activation="relu", normalize_attn=True, mlp=True, layernorm=True):
#         super(ReluDecoder, self).__init__()
#         self.name = f"ReluDecoder_embd={n_embd}_layer={n_layer}_head={n_head}"
#         self.n_positions = n_positions
#         self.n_dims = n_dims
#         self.n_embd = n_embd
#         self.n_head = n_head
#         self.n_layer = n_layer
#         self.activation = get_activation(activation)
#         self.normalize_attn = normalize_attn
#         self.layernorm = layernorm
#         self.mlp = mlp
#         self.n_point = n_point

#         self._read_in = nn.Linear(n_dims, n_embd)
#         self._qkv_list = nn.ModuleList()
#         self._mlps = nn.ModuleList()
#         self._lns_1 = nn.ModuleList()
#         self._lns_2 = nn.ModuleList()

#         for i in range(n_layer):
#             self._qkv_list.append(CustomLinear(self.n_dims, self.n_head))
#             self._lns_1.append(nn.LayerNorm([self.n_embd]))
#             self._mlps.append(
#                 nn.Sequential(
#                     nn.Linear(n_embd, 4 * n_embd),
#                     nn.ReLU(),
#                     nn.Linear(4 * n_embd, n_embd),
#                 )
#             )
#             self._lns_2.append(nn.LayerNorm([self.n_embd]))

#         self._read_out = nn.Linear(n_embd, 1)
#         self._read_out_matrix = nn.Linear(n_embd, n_embd)
#         self._read_out_ham = ElementWiseProduct(n_embd)

#     @staticmethod
#     def _combine(xs_b, ys_b):
#         bsize, points, dim = xs_b.shape
#         xs_e = torch.cat((xs_b, torch.zeros(bsize, points, 1).to(xs_b.device)), dim=2)  # Ensure tensor is on the same device
#         ys_b_wide = torch.cat(
#             (
#                 xs_b,
#                 ys_b.view(bsize, points, 1),
#             ),
#             axis=2,
#         )
#         zs = torch.stack((xs_e, ys_b_wide), dim=2)
#         zs = zs.view(bsize, 2 * points, dim + 1)
#         return zs

#     def forward(self, xs, ys, head_mask, inds=None, return_hidden_states=False):
#         if inds is None:
#             inds = torch.arange(ys.shape[1])
#         else:
#             inds = torch.tensor(inds).to(xs.device)  # Ensure tensor is on the same device
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")
#         zs = self._combine(xs, ys)
#         n_batch, n_points, d_zs = zs.shape
#         hidden_states = []

#         H = self._read_in(zs)
#         hidden_states.append(H)
#         for (qkv, ln1, mlp, ln2) in zip(
#             self._qkv_list,
#             self._lns_1, self._mlps, self._lns_2,
#         ):
#             query, key, value = qkv(H)

#             query = query.view(n_batch, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) * head_mask
#             key = key.view(n_batch, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) * head_mask
#             value = value.view(n_batch, n_points, self.n_head, self.n_embd).permute(0, 2, 1, 3) * head_mask

#             attn_weights = self.activation(torch.einsum('abid,abjd->abij', query, key))

#             maskNorm = torch.zeros(n_points, n_points).to(xs.device)  # Ensure tensor is on the same device
#             maskNorm.requires_grad = False
#             for i in range(1, n_points + 1):
#                 maskNorm[i - 1, :i] = 1 / i
#             maskNorm = maskNorm.unsqueeze(0).repeat(self.n_head, 1, 1)

#             attn_weights = attn_weights * maskNorm
#             attn_weights = torch.einsum('abij,abjd->abid', attn_weights, value)

#             attn_weights = torch.sum(attn_weights, dim=1)

#             H = H + attn_weights
#             if (self.layernorm):
#                 H = ln1(H)

#             if (self.mlp):
#                 H = H + mlp(H)
#                 if (self.layernorm):
#                     H = ln2(H)

#             hidden_states.append(H)

#         prediction = self._read_out(H)
#         # prediction = (self._read_out_matrix(H) * H).sum(dim=2, keepdim=True)

#         # fixed quadratic read-out function
#         fix_readout = False

#         # H = self._read_out_matrix(H)
#         # H_reshaped = H.view(n_batch, self.n_point, 2, 2*d_zs)
#         # d = d_zs-1
#         # H_vectors = H_reshaped[:, :, 0, d+1:2*d+1]
#         # prediction = (H_vectors * xs).sum(dim=2, keepdim=True)

#         if fix_readout is not True:
#             if return_hidden_states:
#                 return prediction[:,::2, 0], hidden_states
#             return prediction[:, ::2, 0]
#         elif fix_readout is True:
#             return prediction[:, :, 0]



class ReluDecoder(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_point=41,
                 activation="relu", normalize_attn=True, mlp=True, layernorm=True):
        super(ReluDecoder, self).__init__()
        self.name = f"ReluDecoder_embd={n_embd}_layer={n_layer}_head={n_head}"

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
                    nn.Linear(n_embd, 4*n_embd),
                    nn.ReLU(),
                    nn.Linear(4*n_embd, n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))

        self._read_out = nn.Linear(n_embd, 1)
        self._read_out_matrix = nn.Linear(n_embd, n_embd)
        self._read_out_ham = ElementWiseProduct(n_embd)

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
        n_batch, n_points, d_zs = zs.shape
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
        # prediction = (self._read_out_matrix(H) * H).sum(dim=2, keepdim=True)

        # fixed quadratic read-out function
        
        fix_readout = False

        # H = self._read_out_matrix(H)
        # H_reshaped = H.view(n_batch, self.n_point, 2, 2*d_zs)
        # d = d_zs-1
        # H_vectors = H_reshaped[:, :, 0, d+1:2*d+1]
        # prediction = (H_vectors * xs).sum(dim=2, keepdim=True)

        if fix_readout is not True:
            if return_hidden_states:
                return prediction[:,::2, 0], hidden_states
            return prediction[:, ::2, 0]
        elif fix_readout is True:
            return prediction[:, :, 0]
    

class LassoEncoder(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_points = 41,
                 activation="relu", normalize_attn=True, mlp=True, layernorm=True):
        super(LassoEncoder, self).__init__()
        self.name = f"EncoderTF_embd={n_embd}_layer={n_layer}_head={n_head}"

        # configs
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_points_train = n_points
        self.activation = get_activation(activation)
        self.normalize_attn = normalize_attn
        self.layernorm = layernorm
        self.mlp = mlp

        # layers
        self._read_in = nn.Linear(n_dims, n_embd, bias=False)
        with torch.no_grad():
            self._read_in.weight.copy_(torch.cat((torch.eye(n_dims), torch.zeros(n_embd-n_dims, n_dims)), dim=0))
        self._queries = nn.ModuleList()
        self._keys = nn.ModuleList()
        self._values = nn.ModuleList()
        self._mlps = nn.ModuleList()
        self._lns_1 = nn.ModuleList()
        self._lns_2 = nn.ModuleList()
        for i in range(n_layer):
            if i != n_layer-1:
                self._queries.append(nn.Linear(n_dims-1, n_dims-1, bias=False))
                self._keys.append(nn.Linear(n_dims-1, n_dims-1, bias=False))
                self._values.append(nn.Linear(n_dims-1, n_dims-1, bias=False))
                self._lns_1.append(nn.LayerNorm([self.n_embd]))
                self._mlps.append(
                    nn.Sequential(
                        nn.Linear(n_embd, n_embd),
                        nn.ReLU(),
                        nn.Linear(n_embd, n_embd),
                    )
                )
                self._lns_2.append(nn.LayerNorm([self.n_embd]))
            else:
                self._queries.append(nn.Linear(n_embd, n_embd, bias=False))
                self._keys.append(nn.Linear(n_embd, n_embd, bias=False))
                self._values.append(nn.Linear(n_embd, n_embd, bias=False))
                self._lns_1.append(nn.LayerNorm([self.n_embd]))
                self._mlps.append(
                    nn.Sequential(
                        nn.Linear(n_embd, n_embd),
                        nn.ReLU(),
                        nn.Linear(n_embd, n_embd),
                    )
                )
                self._lns_2.append(nn.LayerNorm([self.n_embd]))
        self._read_out = nn.Linear(n_embd, 1, bias = False)
        with torch.no_grad():
            vec = torch.zeros(1, n_embd, requires_grad = False)
            vec[:, n_dims-1] = 1
            self._read_out.weight.copy_(vec)

    @staticmethod
    def _combine(n_points_train, xs_b, ys_b):
        """
        Directly stack the x's and y's into the same location
        resulting sequence would be Bx(N+1)x(d+1), where (N+1)-th token is test
        """
        zs = torch.cat((xs_b, ys_b.unsqueeze(2)), dim=2)
        zs[:, -1, -1].zero_()
        if n_points_train < ys_b.shape[1]:
            raise ValueError("Number of prompts in testing larger the training.")
        elif n_points_train > ys_b.shape[1]:
            bs,_, embd = zs.shape
            zeros = torch.zeros(bs,n_points_train,embd)
            zeros[:,n_points_train-ys_b.shape[1]:n_points_train,:] = zs
            zs = zeros.cuda()
        else:
            zs = zs
        return zs

    def forward(self, xs, ys, inds=None, return_hidden_states=False):

        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(self.n_points_train, xs, ys)

        bs,n_points,_ = zs.shape
        hidden_states = []
        for param in self._read_in.parameters():
            param.requires_grad = False
        H = self._read_in(zs)
        H[:,:,-1] = torch.ones(bs, n_points)
        H[:,-1,-1].zero_()
        hidden_states.append(H)
        
        for idx, (q, k, v, ln1, mlp, ln2) in enumerate(zip(
            self._queries, self._keys, self._values,
            self._lns_1, self._mlps, self._lns_2,
        )):
            if idx != self.n_layer-1:
                query = q(H[:,:,self.n_dims:(2*self.n_dims-1)])
                key = H[:,:,0:self.n_dims-1]
                value = v(H[:,:,0:self.n_dims-1])

                query = torch.cat((torch.zeros(bs, n_points, self.n_dims).cuda(), query, torch.zeros(bs, n_points, self.n_embd-2* self.n_dims+1).cuda()), dim = 2)
                key = torch.cat((torch.zeros(bs, n_points, self.n_dims).cuda(), key, torch.zeros(bs, n_points, self.n_embd-2* self.n_dims+1).cuda()), dim = 2)
                value = torch.cat((torch.zeros(bs, n_points, self.n_dims).cuda(), value, torch.zeros(bs, n_points, self.n_embd-2* self.n_dims+1).cuda()), dim = 2)
            else:
                query = q(H)
                key = k(H)
                value = v(H)
            attn_weights = self.activation(torch.einsum('bid,bjd->bij', query, key))
            if self.normalize_attn:
                attn_weights = attn_weights / ys.shape[1]
            H = H + torch.einsum('bij,bjd->bid', attn_weights, value)
            if self.layernorm:
                H = ln1(H)

            if self.mlp:
                H = H + mlp(H)
                if self.layernorm:
                    H = ln2(H)

            hidden_states.append(H)
        for param in self._read_out.parameters():
            param.requires_grad = False
        prediction = self._read_out(H)
        if return_hidden_states:
            return prediction[:, (n_points-ys.shape[1]):n_points, 0], hidden_states
        return prediction[:, (n_points-ys.shape[1]):n_points, 0]
        # if return_hidden_states:
        #     return prediction[:, ::2, 0], hidden_states
        # return prediction[:, ::2, 0]



class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
class ElementWiseProduct(nn.Module):
    def __init__(self, num_features):
        super(ElementWiseProduct, self).__init__()
        # Initialize the learnable parameters vector with num_features elements
        self.weights = nn.Parameter(torch.randn(num_features))

    def forward(self, x):
        # Element-wise multiplication of input x and the weights
        # Assuming x is of shape (batch_size, num_features)
        return x * self.weights
