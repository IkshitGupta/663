import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=5000, help="training iterations per run")
parser.add_argument("--eval_every", type=int, default=25)
parser.add_argument("--n_train", type=int, default=2000)
parser.add_argument("--n_val", type=int, default=2000)
parser.add_argument("--classes", type=int, nargs=2, default=[0, 1],
                    help="the two MNIST digits to use (paper does not say which)")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--data_dir", type=str, default="./data")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

ds = datasets.MNIST(args.data_dir, train=True, download=True)
X_all = ds.data.float().div(255.0).reshape(-1, 784)
y_all = ds.targets
c0, c1 = args.classes
mask = (y_all == c0) | (y_all == c1)
X_all, y_all = X_all[mask], y_all[mask]
tau0_all = (y_all == c1).float()

perm = torch.randperm(len(X_all))
tr_idx = perm[: args.n_train]
va_idx = perm[args.n_train: args.n_train + args.n_val]
X_tr, tau0_tr = X_all[tr_idx].to(device), tau0_all[tr_idx].to(device)
X_va, tau0_va = X_all[va_idx].to(device), tau0_all[va_idx].to(device)
norm_tr = X_tr.norm(dim=1)
print(f"train {len(X_tr)}, val {len(X_va)}, mean ||x|| = {norm_tr.mean().item():.2f}")

def make_net(width=256, depth=6, d_in=784):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.ReLU()]
        d_in = width
    layers.append(nn.Linear(d_in, 1))
    return nn.Sequential(*layers)

def run(k, beta):
    target = tau0_tr + beta * torch.sin(k * norm_tr)
    torch.manual_seed(args.seed)
    net = make_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    steps, tr_l, va_l = [], [], []
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            with torch.no_grad():
                va = loss_fn(net(X_va).squeeze(1), tau0_va).item()
                tr = loss_fn(net(X_tr).squeeze(1), target).item()
            steps.append(step); tr_l.append(tr); va_l.append(va)
        opt.zero_grad()
        loss = loss_fn(net(X_tr).squeeze(1), target)
        loss.backward()
        opt.step()
    return steps, tr_l, va_l

betas = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ks = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
panels = [
    ("(a) k = 0.1",  [(0.1, b) for b in betas], lambda k, b: f"beta={b}"),
    ("(b) k = 1",    [(1.0, b) for b in betas], lambda k, b: f"beta={b}"),
    ("(c) beta = 0.5", [(k, 0.5) for k in ks],  lambda k, b: f"k={k}"),
    ("(d) beta = 1",   [(k, 1.0) for k in ks],  lambda k, b: f"k={k}"),
]

fig_va, axes_va = plt.subplots(1, 4, figsize=(18, 3.6))
fig_tr, axes_tr = plt.subplots(1, 4, figsize=(18, 3.6))
for (title, configs, lab), ax_v, ax_t in zip(panels, axes_va, axes_tr):
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(configs)))
    for color, (k, b) in zip(cmap, configs):
        steps, tr_l, va_l = run(k, b)
        print(f"{title}  k={k} beta={b}  best val {min(va_l):.4f} at step {steps[int(np.argmin(va_l))]}"
              f"  final val {va_l[-1]:.4f}")
        ax_v.plot(steps, va_l, color=color, label=lab(k, b))
        ax_t.plot(steps, tr_l, color=color, label=lab(k, b))
    for ax, ylab in ((ax_v, "Validation MSE (vs clean labels)"), (ax_t, "Training MSE (vs noisy target)")):
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7)

fig_va.tight_layout()
fig_va.savefig("exp3_validation.png", dpi=150)
fig_tr.tight_layout()
fig_tr.savefig("exp3_training.png", dpi=150)
print("saved exp3_validation.png and exp3_training.png")