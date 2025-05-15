import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import torch.nn.functional as F
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True



# def oracle_em_loss(xs, ys, cot, eps=1e-8, *, random_init=True):
#     """
#     Batched Oracle‑EM loss against a Chain‑of‑Thought (CoT) trajectory.

#     Parameters
#     ----------
#     xs : (B, N, K) tensor
#         One‑hot class labels for the first L positions in each sequence.
#         All‑zero rows correspond to unlabelled samples.
#     ys : (B, N, D) tensor
#         Data sampled from a K‑component Gaussian mixture (identity covariance).
#         In the toy setting we take D == K.
#     cot : list[T] of (B, N, D) tensors
#         Model‑predicted per‑sample means after each of the T CoT steps.
#     eps : float, optional (default 1e‑8)
#         Numerical floor to avoid division by zero when normalising.
#     random_init : bool, optional (default True)
#         • True  – initialise EM by picking K random samples per batch.  
#         • False – moment initialisation from whatever labels exist in `xs`
#                   (may be ill‑conditioned if some classes have no labels).

#     Returns
#     -------
#     total_loss : scalar tensor
#         Σ_{t=0}^{T‑1}  MSE( EM means after step t, cot[t] )
#         Ready for back‑prop if you optimise the CoT model.
#     mus_trace  : list[T] of (B, K, D) tensors
#         Component‑mean trajectory after every EM M‑step.
#     """
#     B, N, D = ys.shape
#     K       = xs.size(-1)
#     T       = len(cot)

#     if random_init:
#         # choose K distinct indices per batch for a k‑means‑ish start
#         idx = torch.multinomial(torch.ones(B, N, device=ys.device), K, False)  # (B,K)
#         mus = ys[torch.arange(B).unsqueeze(1), idx]                            # (B,K,D)
#     else:
#         counts = xs.sum(dim=1).clamp_min(eps)           # (B,K)
#         mus    = (xs.transpose(1, 2) @ ys) / counts.unsqueeze(-1)  # (B,K,D)

#     total_loss, mus_trace = 0.0, []

#     for t in range(T):
#         # ============ E‑step ==================================================
#         #   r_{b,n,k} ∝ exp(‑½‖y_{b,n} − μ_{b,k}‖²)
#         dist = ((ys.unsqueeze(2) - mus.unsqueeze(1))**2).sum(-1)   # (B,N,K)
#         r    = torch.softmax(-0.5 * dist, dim=-1)                  # (B,N,K)

#         # ============ M‑step ==================================================
#         r_sum = r.sum(dim=1).clamp_min(eps)                        # (B,K)
#         mus   = (r.transpose(1, 2) @ ys) / r_sum.unsqueeze(-1)     # (B,K,D)
#         mus_trace.append(mus.detach())

#         # ============ step loss ==============================================
#         mu_expanded = torch.einsum('bnk,bkd->bnd', r, mus)         # (B,N,D)
#         print("###############", mu_expanded.shape, cot[t].shape)
#         total_loss  = total_loss + F.mse_loss(mu_expanded, cot[t])
#     total_loss = total_loss/T

#     return total_loss, mus_trace


import torch, numpy as np, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment   # SciPy is on Colab


def _moment_init(xs, ys, eps=1e-8):
    """Label‑moment start; random fallback for unlabeled classes."""
    B, N, K = xs.shape
    D       = ys.size(-1)
    counts  = xs.sum(dim=1)                         # (B,K)
    mus     = torch.zeros(B, K, D, device=ys.device)

    mask = counts > 0
    if mask.any():
        mus[mask] = (xs.transpose(1, 2) @ ys)[mask] / counts.masked_fill(~mask, 1.).unsqueeze(-1)[mask]

    if (~mask).any():                              # random fallback
        rand_idx = torch.randint(N, (B, K), device=ys.device)
        mus[~mask] = ys[torch.arange(B).unsqueeze(1), rand_idx][~mask]
    return mus


def _permute_to_match(mus_b, target_b):
    """
    Align mus_b (K×D) to target_b (K×D) by minimum‑MSE permutation.
    Returns the permuted mus_b and the permutation indices.
    """
    # cost[i,j] = ‖mus_b[i] − target_b[j]‖²
    cost = (
        (mus_b.unsqueeze(1) - target_b.unsqueeze(0)).pow(2).sum(-1)
        .detach()               # <-- break autograd graph
        .cpu()
        .numpy()
    )
    row, col = linear_sum_assignment(cost)          # Hungarian
    perm = torch.as_tensor(row[np.argsort(col)], device=mus_b.device)
    return mus_b[perm], perm


def oracle_em_loss(xs, ys, cot, eps: float = 1e-8):
    """
    xs  : (B,N,K) one‑hot (zeros = unlabeled)
    ys  : (B,N,D)
    cot : list[T] of (B,K,D)  (ground‑truth component means)
    """
    B, N, D = ys.shape
    K, T    = xs.size(-1), len(cot)

    mus = _moment_init(xs, ys, eps)                 # (B,K,D)
    lbl_mask = xs.sum(-1, keepdim=True) > 0         # (B,N,1)

    total_loss, mus_trace = 0.0, []

    for t in range(T):
        # ---------- align & commit permutation ------------------------------
        for b in range(B):
            mus_b_aligned, perm = _permute_to_match(mus[b], cot[t][b])
            mus[b] = mus_b_aligned                       # commit order
        total_loss += F.mse_loss(mus, cot[t])

        # ---------- E‑step ---------------------------------------------------
        dist = ((ys.unsqueeze(2) - mus.unsqueeze(1))**2).sum(-1)  # (B,N,K)
        r    = torch.softmax(-0.5 * dist, dim=-1)

        if xs.size(1) != ys.size(1):
            print(">> MISMATCH: xs has N =", xs.size(1), "but ys/r have N =", ys.size(1))
            raise ValueError("xs and ys sequence lengths differ!")
        r    = torch.where(lbl_mask, xs, r)                       # hard labels

        # ---------- M‑step ---------------------------------------------------
        r_sum = r.sum(dim=1).clamp_min(eps)                       # (B,K)
        mus   = (r.transpose(1, 2) @ ys) / r_sum.unsqueeze(-1)    # (B,K,D)
        mus_trace.append(mus.detach())

    return total_loss / T, mus_trace


import torch

def cot_mean_accuracy(xs, ys, cot, *, step: int = -1):

    C = xs.size(-1)

    mus = cot[step][:, -C:, :]        # (B, C, D)

    dist2 = ((ys.unsqueeze(2) - mus.unsqueeze(1))**2).sum(-1)

    preds = dist2.argmin(dim=-1)      # (B, N) predicted class index
    true  = xs.argmax(dim=-1)         # (B, N) ground-truth index

    acc = (preds == true).float().mean().item()
    return acc




def train_step(model, xs, ys, head_mask, optimizer, loss_func):
    
    
    optimizer.zero_grad()
    output, cot = model(xs, ys, head_mask)
    
    if 'SoftmaxEncoder' in model.name:
        loss_1 = loss_func(output[:,5:,:], xs[:,5:,:])

        xs_proc = xs.clone()
        xs_proc[:, 5:, :].zero_()

        loss_2, _ = oracle_em_loss(xs_proc, ys, cot)
        
        # loss = loss_1+loss_2
        loss = loss_2
        # loss = loss_func(output, xs)
        acc = cot_mean_accuracy(xs,ys,cot)
    else:
        loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach(), loss_2, loss_2, acc


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)

    decay_rate = 0.999989
    last_epoch: int  = 600000
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
    
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()
    n_dims = args.model.n_dims
    n_head = args.model.n_head
    n_point = args.training.curriculum.points.end
    n_embd = args.model.n_embd
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    if 'Decoder' in model.name:
        coe = 2
    else:
        coe = 1
    unmask_every_iter: int = 10000

    print('Model name is: ', model.name)
    
    head_mask_all = torch.zeros(n_head, n_head, coe*n_point, n_embd)
    for h in range(n_head):
        for i in range(h+1):
            head_mask_all[h][i] = torch.ones(coe*n_point, n_embd)

    
    
    for i in pbar:

        
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)
        if i == 1:
            print(xs[0],ys[0])

        loss_func = task.get_training_metric()

        head_mask = head_mask_all[min(i//unmask_every_iter, n_head-1)]

        loss, output, loss_1, loss_2, acc = train_step(model, xs.cuda(), ys.cuda(), head_mask.cuda(), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        if "semi" in args.training.task:
            point_wise_loss = point_wise_loss_func(output, xs.cuda()).mean(dim=0)
        else:
            point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss_1,
                    "CoT_loss": loss_2,
                    "excess_loss": loss / baseline_loss,
                    "Prediction accuracy: " acc,
                    # "pointwise/loss": dict(
                    #     zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    # ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        #curriculum.update_rand()
        curriculum.update()
        #if i <= last_epoch and i > 300000:
        #    lr_scheduler.step()
        pbar.set_description(f"Prediction loss： {loss_1}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    # if not args.test_run:
    #     _ = get_run_metrics(args.out_dir, skip_baselines=True)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "ReluEncoder", "SoftmaxEncoder", "LassoEncoder", "ReluDecoder", "SparseDecoder"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)
    torch.cuda.set_device('cuda:1')
    main(args)
