import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threshold import fit_subthreshold

PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

def _seen(y, fails):
    """Hide points with zero failures: log(0) cannot be drawn and 0 is not a measurement."""
    y = np.array(y, float)
    y[np.asarray(fails) == 0] = np.nan
    return y

def plot_threshold(res, th, th_err, path, title, pairs=None):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ps = np.array(res["ps"])
    for c, d in zip(PAL, res["distances"]):
        ax.errorbar(ps, _seen(res["p_L"][d], res["fails"][d]), yerr=res["err"][d],
                    marker="o", ms=4, capsize=2, color=c, label=f"d = {d}", lw=1.4)
    if np.isfinite(th):
        if np.isfinite(th_err):
            ax.axvspan(th - th_err, th + th_err, color="k", alpha=0.08, lw=0)
        ax.axvline(th, color="k", ls="--", lw=1)
        lab = f"$p_{{th}}\\approx{th:.4f}$" + (f" $\\pm$ {th_err:.4f}" if np.isfinite(th_err) else "")
        ax.text(th, 0.04, "  " + lab, transform=ax.get_xaxis_transform(), fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$")
    rounds = "1 round" if res["level"] == "code_capacity" else "d rounds"
    ax.set_ylabel(f"logical failure probability ({rounds})")
    sub = "" if not pairs else "  crossings: " + ", ".join(f"{a}/{b}: {v:.4f}" for (a, b), v in pairs.items())
    ax.set_title(f"Surface-code threshold: {title}\n{sub}", fontsize=9)
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def plot_logical_vs_physical(res, th, path):
    """Left: per-round p_L vs p with break-even line. Right: fitted exponent vs (d+1)/2."""
    ps = np.array(res["ps"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    a.plot(ps, ps, "k:", lw=1, label="no encoding ($p_L=p$)")
    for c, d in zip(PAL, res["distances"]):
        a.plot(ps, _seen(res["p_round"][d], res["fails"][d]), "o-", ms=4, color=c, label=f"d = {d}")
    a.axvline(th, color="k", ls="--", lw=1)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("physical error rate $p$"); a.set_ylabel("logical error per round $p_L$")
    a.set_title("Logical vs physical error"); a.grid(True, which="both", alpha=0.25); a.legend()
    ds = res["distances"]
    ks = [fit_subthreshold(ps, res["p_round"][d], th, res["fails"][d])[1] for d in ds]
    b.plot(ds, [(d + 1) / 2 for d in ds], "k--", label="theory $(d+1)/2$ (as $p\\to 0$)")
    b.plot(ds, ks, "o-", color=PAL[0], label="fitted exponent ($p < p_{th}/2$)")
    b.set_xlabel("code distance $d$"); b.set_ylabel("exponent $k$ in $p_L\\propto p^{k}$")
    b.set_xticks(ds)
    b.set_title("Sub-threshold error suppression"); b.grid(alpha=0.25); b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def plot_decoder_comparison(res, path, xlabel):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.array([p if np.isscalar(p) else p[1] for p in res["params"]], float)
    ax.errorbar(x, res["mismatched"], res["mismatched_sd"], fmt="s-", color=PAL[3],
                capsize=3, label="decoder assumes uniform noise")
    ax.errorbar(x, res["informed"], res["informed_sd"], fmt="o-", color=PAL[0],
                capsize=3, label="decoder knows per-qubit rates")
    ax.set_xlabel(xlabel); ax.set_ylabel(f"logical error (d={res['d']}, {res['d']} rounds)")
    ax.set_title(f"Non-uniform noise, mean $p$ fixed at {res['p']}")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def agg(det):
    agg = {}
    for (x, y, _t), v in det.items():
        agg.setdefault((x, y), []).append(v)
    return {k: float(np.mean(v)) for k, v in agg.items()}

def plot_detection_map(dm, path):
    prof = dm["profile"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    co, qs = prof.coords, sorted(prof.coords)
    sc = axs[0].scatter([co[q][0] for q in qs], [co[q][1] for q in qs],
                        c=[prof.m(q) for q in qs], cmap="viridis", s=140, edgecolor="k")
    plt.colorbar(sc, ax=axs[0], label="error-rate multiplier")
    axs[0].set_title(f"Profile: {prof.name}, d = {dm['d']}")
    maps = {k: agg(dm[k]) for k in ("det_uniform", "det_true")}
    vmax = max(max(m.values()) for m in maps.values())
    for ax, key, ttl in ((axs[1], "det_uniform", "Syndrome firing rate: uniform chip"),
                         (axs[2], "det_true", "Syndrome firing rate: this chip")):
        xs, ys = zip(*maps[key].keys())
        s = ax.scatter(xs, ys, c=list(maps[key].values()), cmap="magma", s=260,
                       marker="s", edgecolor="k", vmin=0, vmax=vmax)
        plt.colorbar(s, ax=ax, label="P(detection event)")
        ax.set_title(ttl)
    for ax in axs:
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)