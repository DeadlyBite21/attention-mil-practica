"""
Attention-based Deep MIL (Ilse, Tomczak y Welling, ICML 2018) implementado desde cero en NumPy.

Objetivo: ver cada ecuación del paper en código, sin PyTorch, y reproducir a escala pequeña
el experimento MNIST-bags (sección 4.2) usando los dígitos 8x8 que vienen con scikit-learn
(funciona sin internet).

  - Bolsa X = {x_1..x_K}: K imágenes de dígitos; K ~ round(Normal(10, 2)).
  - Etiqueta de bolsa: Y = 1 si contiene al menos un dígito objetivo (por defecto '9', como el paper;
    se cambia con --target) (ec. 1-2: Y = max_k y_k).
  - Las etiquetas por instancia (y_k) NO se usan para entrenar, solo para evaluar
    si la atención encontró las instancias clave.

Modelo (enfoque "embedding-level", fig. 6c del paper):
  f  : h_k = ReLU(W2 · ReLU(W1 x_k + b1) + b2)            (instancia -> embedding, M dims)
  σ  : z  = Σ_k a_k h_k                                    (MIL pooling, ec. 7)
       a_k = softmax_k( w^T tanh(V h_k) )                   (atención, ec. 8)
       a_k = softmax_k( w^T [tanh(V h_k) ⊙ sigm(U h_k)] )   (atención con compuerta, ec. 9)
       (también 'mean' (ec. 6) y 'max' (ec. 5) como líneas base)
  g  : θ(X) = sigm(c^T z + d)                               (probabilidad de la bolsa)
  Pérdida: -[Y log θ + (1-Y) log(1-θ)]   (log-verosimilitud Bernoulli, sección 2.1)

Uso:
  python attention_mil_numpy_demo.py --gradcheck        # verifica el backprop manual
  python attention_mil_numpy_demo.py                    # experimento completo + figuras
"""

import argparse
import json
import os

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------------------
# Datos: bolsas de dígitos (análogo a MNIST-bags)
# --------------------------------------------------------------------------------------

TARGET = 9  # dígito objetivo; se sobrescribe con --target


def load_split(seed=0, test_frac=0.4):
    """Separa IMÁGENES en train/test antes de armar bolsas (como el paper con MNIST)."""
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0  # 64 píxeles en [0, 1]
    y = d.target
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    return (X[idx[n_test:]], y[idx[n_test:]]), (X[idx[:n_test]], y[idx[:n_test]])


def make_bags(X, y, n_bags, mean_len=10, std_len=2, seed=0):
    """Igual que el código oficial: largo ~ Normal(mean, std), instancias al azar."""
    rng = np.random.RandomState(seed)
    bags = []
    for _ in range(n_bags):
        K = max(2, int(round(rng.normal(mean_len, std_len))))
        idx = rng.randint(0, len(X), K)
        inst = (y[idx] == TARGET).astype(int)
        bags.append({"X": X[idx], "y_inst": inst, "Y": int(inst.max()), "digits": y[idx]})
    return bags


# --------------------------------------------------------------------------------------
# Modelo con forward y backward escritos a mano
# --------------------------------------------------------------------------------------

def glorot(rng, fan_out, fan_in):
    lim = np.sqrt(6.0 / (fan_in + fan_out))  # Glorot & Bengio (2010), como en el paper
    return rng.uniform(-lim, lim, size=(fan_out, fan_in))


def sigm(x):
    return 1.0 / (1.0 + np.exp(-x))


class MILNet:
    def __init__(self, pooling="attention", D_in=64, H=128, M=64, L=32, seed=0):
        assert pooling in ("mean", "max", "attention", "gated")
        self.pooling = pooling
        r = np.random.RandomState(seed)
        self.p = {
            "W1": glorot(r, H, D_in), "b1": np.zeros(H),
            "W2": glorot(r, M, H), "b2": np.zeros(M),
            "V": glorot(r, L, M),                    # V ∈ R^{L×M}
            "U": glorot(r, L, M),                    # U ∈ R^{L×M} (solo gated)
            "w": glorot(r, 1, L).ravel(),            # w ∈ R^{L}
            "c": glorot(r, 1, M).ravel(), "d": np.zeros(1),
        }

    # ---- forward -----------------------------------------------------------------
    def forward(self, X):
        p, c = self.p, {}
        c["X"] = X
        c["pre1"] = X @ p["W1"].T + p["b1"]
        c["h1"] = np.maximum(c["pre1"], 0)
        c["pre2"] = c["h1"] @ p["W2"].T + p["b2"]
        H = np.maximum(c["pre2"], 0)                  # h_k (K×M)
        c["H"] = H
        K = len(H)

        if self.pooling == "mean":                    # ec. (6)
            a = np.full(K, 1.0 / K)
        elif self.pooling == "max":                   # ec. (5): max por dimensión
            c["argmax"] = H.argmax(axis=0)
            a = None
        else:
            c["T"] = np.tanh(H @ p["V"].T)            # tanh(V h_k^T)   (K×L)
            if self.pooling == "gated":
                c["S"] = sigm(H @ p["U"].T)           # sigm(U h_k^T)   (K×L)
                c["G"] = c["T"] * c["S"]              # ⊙
            else:
                c["G"] = c["T"]
            s = c["G"] @ p["w"]                       # w^T (...)       (K)
            s = s - s.max()                           # estabilidad numérica del softmax
            a = np.exp(s) / np.exp(s).sum()           # ec. (8)/(9): suman 1

        z = H.max(axis=0) if a is None else a @ H     # ec. (7): z = Σ a_k h_k
        c["a"], c["z"] = a, z
        theta = sigm(z @ p["c"] + p["d"][0])          # g: probabilidad de la bolsa
        c["theta"] = theta
        self.cache = c
        return theta, a

    @staticmethod
    def loss(theta, Y, eps=1e-7):
        theta = np.clip(theta, eps, 1 - eps)
        return -(Y * np.log(theta) + (1 - Y) * np.log(1 - theta))

    # ---- backward ----------------------------------------------------------------
    def backward(self, Y):
        p, c = self.p, self.cache
        g = {k: np.zeros_like(v) for k, v in p.items()}
        H, a, z = c["H"], c["a"], c["z"]

        dlogit = c["theta"] - Y                       # d(BCE)/d(logit)
        g["c"] = dlogit * z
        g["d"] = np.array([dlogit])
        dz = dlogit * p["c"]                          # (M)

        if self.pooling == "max":
            dH = np.zeros_like(H)
            dH[c["argmax"], np.arange(H.shape[1])] = dz   # solo la instancia ganadora
        else:
            # z = Σ a_k h_k  ->  el gradiente a h_k va escalado por a_k
            # ("la atención filtra el gradiente", sección 4.2)
            dH = np.outer(a, dz)
            if self.pooling in ("attention", "gated"):
                da = H @ dz                                       # (K)
                ds = a * (da - (a * da).sum())                    # derivada del softmax
                g["w"] = c["G"].T @ ds
                dG = np.outer(ds, p["w"])                         # (K×L)
                if self.pooling == "gated":
                    dT, dS = dG * c["S"], dG * c["T"]
                    dpreU = dS * c["S"] * (1 - c["S"])
                    g["U"] = dpreU.T @ H
                    dH += dpreU @ p["U"]
                else:
                    dT = dG
                dpreV = dT * (1 - c["T"] ** 2)
                g["V"] = dpreV.T @ H
                dH += dpreV @ p["V"]

        dpre2 = dH * (c["pre2"] > 0)
        g["W2"] = dpre2.T @ c["h1"]
        g["b2"] = dpre2.sum(0)
        dh1 = dpre2 @ p["W2"]
        dpre1 = dh1 * (c["pre1"] > 0)
        g["W1"] = dpre1.T @ c["X"]
        g["b1"] = dpre1.sum(0)
        return g


class Adam:
    """Adam con weight decay L2 (como torch.optim.Adam(weight_decay=...))."""

    def __init__(self, params, lr=5e-4, b1=0.9, b2=0.999, wd=1e-4, eps=1e-8):
        self.lr, self.b1, self.b2, self.wd, self.eps = lr, b1, b2, wd, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            gk = grads[k] + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# --------------------------------------------------------------------------------------
# Entrenamiento y evaluación
# --------------------------------------------------------------------------------------

def evaluate(model, bags):
    probs, losses = [], []
    for b in bags:
        th, _ = model.forward(b["X"])
        probs.append(th)
        losses.append(MILNet.loss(th, b["Y"]))
    Y = np.array([b["Y"] for b in bags])
    probs = np.array(probs)
    auc = roc_auc_score(Y, probs) if len(set(Y)) > 1 else float("nan")
    err = np.mean((probs >= 0.5) != Y)
    return auc, err, float(np.mean(losses)), probs


def attention_instance_auc(model, bags):
    """¿La atención apunta al dígito objetivo? AUC de a_k vs y_k dentro de bolsas positivas.
    (El paper lo muestra cualitativamente en la fig. 4; aquí lo medimos.)"""
    scores, labels = [], []
    for b in bags:
        if b["Y"] == 1 and 0 < b["y_inst"].sum() < len(b["y_inst"]):
            _, a = model.forward(b["X"])
            scores += list(a)
            labels += list(b["y_inst"])
    return roc_auc_score(labels, scores)


def train(pooling, train_bags, val_bags, epochs=60, seed=0, lr=5e-4, wd=1e-4):
    model = MILNet(pooling=pooling, seed=seed)
    opt = Adam(model.p, lr=lr, wd=wd)
    rng = np.random.RandomState(seed)
    best, best_crit = None, np.inf
    for _ in range(epochs):
        for i in rng.permutation(len(train_bags)):           # batch = 1 bolsa (como el paper)
            b = train_bags[i]
            model.forward(b["X"])
            opt.step(model.p, model.backward(b["Y"]))
        _, err, loss, _ = evaluate(model, val_bags)
        crit = err + loss                                     # "lowest validation error+loss"
        if crit < best_crit:
            best_crit = crit
            best = {k: v.copy() for k, v in model.p.items()}
    model.p = best
    return model


def gradcheck():
    """Compara el gradiente analítico con diferencias finitas, para cada pooling."""
    (Xtr, ytr), _ = load_split()
    bag = make_bags(Xtr, ytr, 1, seed=3)[0]
    for pooling in ("mean", "max", "attention", "gated"):
        m = MILNet(pooling=pooling, seed=1)
        # pequeño sesgo positivo para que las ReLU estén activas
        m.p["b1"] += 0.05
        m.p["b2"] += 0.05
        m.forward(bag["X"])
        g = m.backward(bag["Y"])
        worst = 0.0
        for k in m.p:
            flat = m.p[k].reshape(-1)
            for j in np.random.RandomState(0).choice(flat.size, min(15, flat.size), replace=False):
                old = flat[j]
                flat[j] = old + 1e-5
                lp = MILNet.loss(m.forward(bag["X"])[0], bag["Y"])
                flat[j] = old - 1e-5
                lm = MILNet.loss(m.forward(bag["X"])[0], bag["Y"])
                flat[j] = old
                num = (lp - lm) / 2e-5
                ana = g[k].reshape(-1)[j]
                worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        print(f"  {pooling:9s} error relativo máximo: {worst:.2e}  {'OK' if worst < 1e-4 else 'REVISAR'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sizes", default="25,50,100,200")
    ap.add_argument("--target", type=int, default=9, choices=range(10),
                    help="dígito que vuelve positiva la bolsa (paper: 9)")
    ap.add_argument("--out", default=None, help="por defecto resultados_numpy_target<dígito>")
    ap.add_argument("--replot", action="store_true", help="solo rehace la figura desde resumen.json")
    args = ap.parse_args()
    global TARGET
    TARGET = args.target
    if args.out is None:
        args.out = f"resultados_numpy_target{TARGET}"

    if args.replot:
        with open(os.path.join(args.out, "resumen.json")) as f:
            bag = json.load(f)["bag_auc"]
        bag = {p: {int(n): v for n, v in d.items()} for p, d in bag.items()}
        plot_results(bag, sorted(next(iter(bag.values()))), args.out)
        return

    if args.gradcheck:
        print("Verificación de gradientes:")
        gradcheck()
        return

    os.makedirs(args.out, exist_ok=True)
    (Xtr, ytr), (Xte, yte) = load_split()
    test_bags = make_bags(Xte, yte, 1000, seed=999)          # 1000 bolsas de test fijas
    sizes = [int(s) for s in args.sizes.split(",")]
    poolings = ["mean", "max", "attention", "gated"]
    res = {p: {n: [] for n in sizes} for p in poolings}
    inst = {p: {n: [] for n in sizes} for p in ("attention", "gated")}

    for n in sizes:
        for s in range(args.seeds):
            bags = make_bags(Xtr, ytr, n, seed=100 * s + n)
            n_val = max(5, n // 5)
            tr, va = bags[n_val:], bags[:n_val]
            for p in poolings:
                m = train(p, tr, va, epochs=args.epochs, seed=s)
                auc, *_ = evaluate(m, test_bags)
                res[p][n].append(auc)
                if p in inst:
                    inst[p][n].append(attention_instance_auc(m, test_bags))
            print(f"n={n:4d} seed={s}  " + "  ".join(f"{p}={res[p][n][-1]:.3f}" for p in poolings),
                  flush=True)

    summary = {
        "bag_auc": {p: {n: [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))]
                        for n, v in d.items()} for p, d in res.items()},
        "attention_instance_auc": {p: {n: [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))]
                                       for n, v in d.items()} for p, d in inst.items()},
    }
    with open(os.path.join(args.out, "resumen.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAUC de bolsa en test (media ± error estándar):")
    print("pooling    " + "".join(f"{n:>16d}" for n in sizes))
    for p in poolings:
        print(f"{p:10s} " + "".join(f"{np.mean(res[p][n]):>9.3f} ± {np.std(res[p][n]) / np.sqrt(args.seeds):.3f}"
                                    for n in sizes))
    print(f"\nAUC de los pesos a_k para encontrar los '{TARGET}' (instancias) en bolsas positivas:")
    for p in inst:
        print(f"{p:10s} " + "".join(f"{np.mean(inst[p][n]):>9.3f} ± {np.std(inst[p][n]) / np.sqrt(args.seeds):.3f}"
                                    for n in sizes))

    # modelo final con más bolsas para la figura de atención
    bags = make_bags(Xtr, ytr, 200, seed=7)
    model = train("attention", bags[40:], bags[:40], epochs=args.epochs, seed=0)
    np.savez(os.path.join(args.out, "modelo_attention.npz"), **model.p)
    plot_results(summary["bag_auc"], sizes, args.out)
    plot_attention_examples(model, test_bags, args.out)


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------

COLORS = {"mean": "#257de0", "max": "#f47230", "attention": "#00b894", "gated": "#f2a600"}
LABELS = {"mean": "Embedding + mean", "max": "Embedding + max",
          "attention": "Attention", "gated": "Gated attention"}
MARKERS = {"mean": "o", "max": "s", "attention": "D", "gated": "^"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors="#666666")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_results(summary, sizes, out):
    """summary = resumen.json['bag_auc'] -> {pooling: {n: [media, error_estandar]}}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    _style(ax)
    for p in summary:
        m = np.array([summary[p][n][0] for n in sizes])
        se = np.array([summary[p][n][1] for n in sizes])
        ax.plot(sizes, m, color=COLORS[p], lw=2, marker=MARKERS[p], ms=7,
                markeredgecolor="white", markeredgewidth=1.5, label=LABELS[p], zorder=3)
        ax.fill_between(sizes, m - se, m + se, color=COLORS[p], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Número de bolsas de entrenamiento", color="#191919")
    ax.set_ylabel("AUC de bolsa (test, 1000 bolsas)", color="#191919")
    ax.set_title(f"Bolsas de dígitos 8×8 (≈10 instancias por bolsa): positivo si hay un '{TARGET}'",
                 loc="left", color="#191919", fontsize=11)
    ax.legend(frameon=False, loc="lower right", labelcolor="#191919")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "auc_vs_bolsas.png"))
    plt.close(fig)


def plot_attention_examples(model, test_bags, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_multi = next(b for b in test_bags if b["y_inst"].sum() >= 2 and len(b["y_inst"]) >= 10)
    pos_one = next(b for b in test_bags if b["y_inst"].sum() == 1 and len(b["y_inst"]) >= 10)
    neg = next(b for b in test_bags if b["Y"] == 0 and len(b["y_inst"]) >= 10)
    rows = [(f"Bolsa positiva (varios '{TARGET}')", pos_multi),
            (f"Bolsa positiva (un '{TARGET}')", pos_one),
            (f"Bolsa negativa (sin '{TARGET}')", neg)]
    ncol = max(len(b["y_inst"]) for _, b in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.05 * ncol, 1.75 * len(rows)), dpi=150)
    for r, (title, b) in enumerate(rows):
        theta, a = model.forward(b["X"])
        for k in range(ncol):
            ax = axes[r, k]
            ax.axis("off")
            if k >= len(a):
                continue
            ax.imshow(b["X"][k].reshape(8, 8), cmap="gray_r")
            if b["y_inst"][k]:
                ax.add_patch(plt.Rectangle((-0.5, -0.5), 8, 8, fill=False, ec="#e5484d", lw=2.5))
            ax.set_title(f"a={a[k]:.3f}", fontsize=7, color="#191919",
                         fontweight="bold" if a[k] == a.max() else "normal")
        axes[r, 0].text(-0.5, -4.2, f"{title}  ·  θ(X) = {theta:.3f}", fontsize=9,
                        color="#191919", transform=axes[r, 0].transData, ha="left")
    fig.suptitle(f"Pesos de atención a_k por instancia (recuadro rojo = '{TARGET}' real, nunca visto en entrenamiento)",
                 fontsize=9.5, color="#191919", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pesos_atencion.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()