import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20000, help="training iterations per run")
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--runs", type=int, default=2, help="random-phase runs to average")
parser.add_argument("--Ls", type=int, nargs="+", default=[0, 4, 10, 16], help="petal counts")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
np.random.seed(args.seed)

N = 1000
freqs = np.arange(20, 220, 20)
z = np.linspace(0, 1, N, endpoint=False)

def flower(L):
    R = 1 + 0.5 * np.sin(2 * np.pi * L * z)
    return np.stack([R * np.cos(2 * np.pi * z), R * np.sin(2 * np.pi * z)], axis=1)

def make_target(phases):
    return sum(np.sin(2 * np.pi * k * z + p) for k, p in zip(freqs, phases))

def make_net(width=256, depth=6, d_in=2):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.ReLU()]
        d_in = width
    layers.append(nn.Linear(d_in, 1))
    return nn.Sequential(*layers)

def spectrum_at_freqs(y):
    return np.abs(np.fft.rfft(y) * 2.0 / N)[freqs]

n_logs = args.steps // args.log_every + 1
spec = {L: np.zeros((args.runs, n_logs, len(freqs))) for L in args.Ls}
loss = {L: np.zeros((args.runs, n_logs)) for L in args.Ls}

for r in range(args.runs):
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    y = make_target(phases)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

    for L in args.Ls:
        X_t = torch.tensor(flower(L), dtype=torch.float32, device=device)
        torch.manual_seed(args.seed + r)
        net = make_net().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        for step in range(args.steps + 1):
            if step % args.log_every == 0:
                with torch.no_grad():
                    out = net(X_t)
                    i = step // args.log_every
                    spec[L][r, i] = spectrum_at_freqs(out.squeeze(1).cpu().numpy())
                    loss[L][r, i] = loss_fn(out, y_t).item()
            opt.zero_grad()
            l = loss_fn(net(X_t), y_t)
            l.backward()
            opt.step()
        print(f"run {r+1}/{args.runs}  L={L:2d}  final loss {loss[L][r, -1]:.4f}")

fig, axes = plt.subplots(1, len(args.Ls) + 1, figsize=(4 * (len(args.Ls) + 1), 4))
for ax, L in zip(axes, args.Ls):
    m = spec[L].mean(axis=0)
    im = ax.imshow(np.clip(m, 0, 1), origin="lower", aspect="auto", cmap="viridis",
                   extent=[freqs[0] - 10, freqs[-1] + 10, 0, args.steps], vmin=0, vmax=1)
    ax.set_title(f"L = {L}")
    ax.set_xlabel("Frequency k")
    ax.set_xticks(freqs[::2])
axes[0].set_ylabel("Training iteration")
fig.colorbar(im, ax=axes[:len(args.Ls)], label="|f~(k)| / A (clipped to [0,1])", fraction=0.02)

x_it = np.arange(n_logs) * args.log_every
for L in args.Ls:
    axes[-1].plot(x_it, loss[L].mean(axis=0), label=f"L={L}")
axes[-1].set_xlabel("Training iteration")
axes[-1].set_ylabel("MSE")
axes[-1].set_title("Loss curves")
axes[-1].legend()
fig.savefig("exp5_manifold.png", dpi=150, bbox_inches="tight")

fig2, ax2 = plt.subplots(figsize=(6, 4))
for L in args.Ls:
    ax2.plot(x_it, spec[L].mean(axis=0)[:, -1], label=f"L={L}")
ax2.set_xlabel("Training iteration")
ax2.set_ylabel(f"fraction of k={freqs[-1]} learned")
ax2.set_title("Highest frequency: faster with more petals")
ax2.legend()
fig2.tight_layout()
fig2.savefig("exp5_highest_freq.png", dpi=150)
print("saved exp5_manifold.png and exp5_highest_freq.png")