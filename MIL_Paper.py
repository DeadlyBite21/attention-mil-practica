"""
Reproducción del experimento MNIST-bags de Ilse, Tomczak y Welling (ICML 2018),
"Attention-based Deep Multiple Instance Learning", en PyTorch.

Fiel al paper (sección 4.2 y apéndice 6.4):
  - Bolsa: K imágenes MNIST 28x28, K = round(Normal(media, var)) con (10,2), (50,10) o (100,20).
  - Y = 1 si la bolsa contiene al menos un '9'.
  - Bolsas de train solo con imágenes del train de MNIST; 1000 bolsas de test con imágenes del test.
  - f = LeNet5 (tabla 8): conv(5)-20, maxpool, conv(5)-50, maxpool, fc-500.
  - MIL pooling: max / mean / attention-128 / gated-attention-128 (L = 128).
  - g = fc-1 + sigmoide.  Enfoque instance-level (tabla 9) también disponible.
  - Adam(lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-4), 200 épocas, batch = 1 bolsa (tabla 10).
  - Inicialización de Glorot y sesgos en cero.
  - Selección del modelo por menor error + pérdida en validación.

Funciona en CPU, CUDA (Colab) o MPS (Mac con Apple Silicon).

Ejemplos:
  pip install torch torchvision scikit-learn matplotlib
  python attention_mil_mnist_bags.py --pooling attention --train-bags 100
  python attention_mil_mnist_bags.py --pooling gated --train-bags 50 --mean-bag 10 --var-bag 2
  python attention_mil_mnist_bags.py --compare --train-bags 100 --epochs 50   # todos los poolings
  python attention_mil_mnist_bags.py --device cpu ...                          # forzar CPU
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# --------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------

class MnistBags(Dataset):
    """Bolsas de MNIST como en el código oficial (AMLab-Amsterdam/AttentionDeepMIL)."""

    def __init__(self, root, train, num_bags, target=9, mean_bag=10, var_bag=2, seed=1):
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ds = datasets.MNIST(root, train=train, download=True, transform=tf)
        # cargamos todo a memoria una vez (60k x 28 x 28 cabe sin problema)
        self.images = torch.stack([ds[i][0] for i in range(len(ds))])
        self.labels = ds.targets.clone()
        rng = np.random.RandomState(seed)
        self.bags = []
        for _ in range(num_bags):
            K = max(1, int(round(rng.normal(mean_bag, var_bag))))
            idx = torch.from_numpy(rng.randint(0, len(self.labels), K))
            self.bags.append(idx)
        self.target = target

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        idx = self.bags[i]
        inst_labels = (self.labels[idx] == self.target).long()
        return self.images[idx], inst_labels.max().float(), inst_labels, self.labels[idx]


# --------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Ec. (8): a_k = softmax_k( w^T tanh(V h_k) ).
    Ec. (9) (gated=True): a_k = softmax_k( w^T [tanh(V h_k) ⊙ sigm(U h_k)] )."""

    def __init__(self, M=500, L=128, gated=False):
        super().__init__()
        self.V = nn.Linear(M, L)
        self.U = nn.Linear(M, L) if gated else None
        self.w = nn.Linear(L, 1)

    def forward(self, H):                      # H: (K, M)
        A = torch.tanh(self.V(H))              # (K, L)
        if self.U is not None:
            A = A * torch.sigmoid(self.U(H))   # compuerta
        A = self.w(A).squeeze(-1)              # (K,)
        a = torch.softmax(A, dim=0)            # pesos que suman 1
        z = a @ H                              # ec. (7): z = Σ a_k h_k   -> (M,)
        return z, a


class DeepMIL(nn.Module):
    """pooling in {'attention','gated','mean','max'} -> embedding-level (tabla 8)
       pooling in {'inst_max','inst_mean'}         -> instance-level (tabla 9)"""

    def __init__(self, pooling="attention", M=500, L=128):
        super().__init__()
        self.pooling = pooling
        # f: LeNet5
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(), nn.Linear(50 * 4 * 4, M), nn.ReLU(),
        )
        if pooling in ("attention", "gated"):
            self.attention = AttentionPooling(M, L, gated=(pooling == "gated"))
        self.classifier = nn.Linear(M, 1)      # g (embedding) o clasificador de instancia
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)  # Glorot & Bengio (2010)
            nn.init.zeros_(m.bias)

    def forward(self, X):                      # X: (K, 1, 28, 28) = una bolsa
        H = self.features(X)                   # (K, M)
        a = None
        if self.pooling == "attention" or self.pooling == "gated":
            z, a = self.attention(H)
            theta = torch.sigmoid(self.classifier(z)).squeeze()
        elif self.pooling == "mean":           # ec. (6)
            theta = torch.sigmoid(self.classifier(H.mean(0))).squeeze()
        elif self.pooling == "max":            # ec. (5), max por dimensión
            theta = torch.sigmoid(self.classifier(H.max(0).values)).squeeze()
        elif self.pooling == "inst_max":       # puntaje por instancia, luego max
            s = torch.sigmoid(self.classifier(H)).squeeze(-1)
            theta, a = s.max(), s              # a = puntajes de instancia (para comparar)
        elif self.pooling == "inst_mean":
            s = torch.sigmoid(self.classifier(H)).squeeze(-1)
            theta, a = s.mean(), s
        else:
            raise ValueError(self.pooling)
        return theta.clamp(1e-5, 1 - 1e-5), a


# --------------------------------------------------------------------------------------
# Entrenamiento / evaluación
# --------------------------------------------------------------------------------------

def run_epoch(model, ds, indices, device, opt=None):
    train = opt is not None
    model.train(train)
    probs, Ys, losses = [], [], []
    for i in indices:
        X, Y, _, _ = ds[i]
        X, Y = X.to(device), Y.to(device)
        with torch.set_grad_enabled(train):
            theta, _ = model(X)
            loss = F.binary_cross_entropy(theta, Y)   # -log-verosimilitud Bernoulli
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        probs.append(theta.item())
        Ys.append(Y.item())
        losses.append(loss.item())
    probs, Ys = np.array(probs), np.array(Ys)
    return float(np.mean(losses)), float(np.mean((probs >= 0.5) != Ys)), probs, Ys


def metrics(probs, Ys):
    pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(Ys, pred),
        "precision": precision_score(Ys, pred, zero_division=0),
        "recall": recall_score(Ys, pred, zero_division=0),
        "f1": f1_score(Ys, pred, zero_division=0),
        "auc": roc_auc_score(Ys, probs),
    }


@torch.no_grad()
def instance_auc(model, ds, device):
    """AUC de los pesos de atención para encontrar los '9' dentro de bolsas positivas.
    El paper lo muestra cualitativamente (fig. 4, 7-9); aquí se cuantifica."""
    model.eval()
    scores, labels = [], []
    for i in range(len(ds)):
        X, Y, y_inst, _ = ds[i]
        if Y.item() == 1 and 0 < y_inst.sum() < len(y_inst):
            _, a = model(X.to(device))
            scores += a.cpu().tolist()
            labels += y_inst.tolist()
    return roc_auc_score(labels, scores) if labels else float("nan")


def train_one(pooling, train_ds, test_ds, args, device):
    torch.manual_seed(args.seed)
    model = DeepMIL(pooling).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)
    n = len(train_ds)
    perm = np.random.RandomState(args.seed).permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    best, best_state = np.inf, None
    for ep in range(1, args.epochs + 1):
        np.random.shuffle(tr_idx)
        tr_loss, tr_err, _, _ = run_epoch(model, train_ds, tr_idx, device, opt)
        va_loss, va_err, _, _ = run_epoch(model, train_ds, val_idx, device)
        if va_err + va_loss < best:            # "lowest validation error + loss"
            best = va_err + va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"[{pooling}] época {ep:3d}  train loss {tr_loss:.3f} err {tr_err:.3f}  "
                  f"val loss {va_loss:.3f} err {va_err:.3f}", flush=True)
    model.load_state_dict(best_state)

    _, _, probs, Ys = run_epoch(model, test_ds, range(len(test_ds)), device)
    m = metrics(probs, Ys)
    if pooling in ("attention", "gated", "inst_max", "inst_mean"):
        m["instance_auc"] = instance_auc(model, test_ds, device)
    return model, m


# --------------------------------------------------------------------------------------
# Visualización (como la fig. 4 / 7-9 del paper)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def plot_bags(model, ds, device, path, n_each=2, max_inst=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    chosen, n_pos, n_neg = [], 0, 0
    for i in range(len(ds)):
        _, Y, _, _ = ds[i]
        if Y.item() == 1 and n_pos < n_each:
            chosen.append(i); n_pos += 1
        elif Y.item() == 0 and n_neg < 1:
            chosen.append(i); n_neg += 1
        if n_pos == n_each and n_neg == 1:
            break

    ncol = min(max_inst, max(len(ds.bags[i]) for i in chosen))
    fig, axes = plt.subplots(len(chosen), ncol, figsize=(1.1 * ncol, 1.8 * len(chosen)), dpi=130)
    axes = np.atleast_2d(axes)
    for r, i in enumerate(chosen):
        X, Y, y_inst, _ = ds[i]
        theta, a = model(X.to(device))
        a = a.cpu().numpy()
        for k in range(ncol):
            ax = axes[r, k]
            ax.axis("off")
            if k >= len(a):
                continue
            ax.imshow(X[k, 0].numpy(), cmap="gray")
            if y_inst[k]:
                ax.add_patch(plt.Rectangle((0, 0), 27, 27, fill=False, ec="#e5484d", lw=2.5))
            ax.set_title(f"a={a[k]:.4f}", fontsize=7)
        axes[r, 0].set_title(f"Y={int(Y)}  θ={theta.item():.3f}\na={a[0]:.4f}", fontsize=7, loc="left")
    fig.suptitle("Pesos por instancia (recuadro rojo = '9')", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------

def get_device(choice="auto"):
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="attention",
                    choices=["attention", "gated", "mean", "max", "inst_max", "inst_mean"])
    ap.add_argument("--compare", action="store_true", help="entrena los 6 poolings y compara")
    ap.add_argument("--train-bags", type=int, default=100)   # paper: 50..500
    ap.add_argument("--test-bags", type=int, default=1000)
    ap.add_argument("--mean-bag", type=float, default=10)    # paper: 10, 50, 100
    ap.add_argument("--var-bag", type=float, default=2)      # paper: 2, 10, 20
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="resultados_torch")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="auto elige cuda > mps > cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device(args.device)
    print("Dispositivo:", device)
    os.makedirs(args.out, exist_ok=True)

    common = dict(target=6, mean_bag=args.mean_bag, var_bag=args.var_bag)
    train_ds = MnistBags(args.data, True, args.train_bags, seed=args.seed, **common)
    test_ds = MnistBags(args.data, False, args.test_bags, seed=args.seed + 1000, **common)
    pos = sum(int(train_ds[i][1]) for i in range(len(train_ds)))
    print(f"Bolsas train: {len(train_ds)} ({pos} positivas) | test: {len(test_ds)}")

    poolings = ["inst_max", "inst_mean", "max", "mean", "attention", "gated"] if args.compare \
        else [args.pooling]
    results = {}
    for p in poolings:
        model, m = train_one(p, train_ds, test_ds, args, device)
        results[p] = m
        print(f"==> {p}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
        if p in ("attention", "gated"):
            plot_bags(model, test_ds, device, os.path.join(args.out, f"bolsas_{p}.png"))
            torch.save(model.state_dict(), os.path.join(args.out, f"modelo_{p}.pt"))

    tag = f"bags{args.train_bags}_mean{int(args.mean_bag)}_seed{args.seed}"
    with open(os.path.join(args.out, f"resultados_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    print("\nResumen (test):")
    keys = ["accuracy", "precision", "recall", "f1", "auc", "instance_auc"]
    print(f"{'pooling':10s}" + "".join(f"{k:>14s}" for k in keys))
    for p, m in results.items():
        print(f"{p:10s}" + "".join(f"{m.get(k, float('nan')):>14.3f}" for k in keys))


if __name__ == "__main__":
    main()