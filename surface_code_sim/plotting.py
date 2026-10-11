"""plotting.py -- matplotlib figures (colour-blind-safe Okabe-Ito palette)."""
from __future__ import annotations

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import fit_subthreshold

PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def plot_threshold(res: Dict, th: float, th_sd: float, path: str, key="p_round"):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ps = np.array(res["ps"])
    for c, d in zip(PAL, res["distances"]):
        y = np.array(res[key][d]); e = np.array(res["err"][d])
        ax.errorbar(ps, y, yerr=e if key == "p_L" else None, marker="o", ms=4,
                    color=c, label=f"d = {d}", lw=1.4)
    if np.isfinite(th):
        ax.axvline(th, color="k", ls="--", lw=1)
        ax.annotate(f"$p_{{th}}\\approx{th:.4f}$", (th, ax.get_ylim()[0] * 1.5),
                    xytext=(5, 6), textcoords="offset points", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error per round" if key == "p_round" else "logical failure probability over $d$ rounds")
    ax.set_title(f"Surface-code threshold ({res['level'].replace('_', ' ')} noise, MWPM)")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_logical_vs_physical(res: Dict, th: float, path: str):
    """Left: p_L vs p with y=x line (break-even).  Right: sub-threshold scaling
    exponent from fits vs the theoretical (d+1)/2."""
    ps = np.array(res["ps"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    a.plot(ps, ps, "k:", lw=1, label="no encoding ($p_L=p$)")
    for c, d in zip(PAL, res["distances"]):
        a.plot(ps, res["p_round"][d], "o-", ms=4, color=c, label=f"d = {d}")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("physical error rate $p$"); a.set_ylabel("logical error per round $p_L$")
    a.set_title("Logical vs physical error"); a.grid(True, which="both", alpha=0.25); a.legend()
    ds, ks = [], []
    for d in res["distances"]:
        _, k = fit_subthreshold(ps, res["p_round"][d], d, th)
        ds.append(d); ks.append(k)
    b.plot(ds, [(d + 1) / 2 for d in ds], "k--", label="theory $(d+1)/2$")
    b.plot(ds, ks, "o-", color=PAL[0], label="fitted exponent")
    b.set_xlabel("code distance $d$"); b.set_ylabel("exponent of $p$ in $p_L\\propto p^{k}$")
    b.set_title("Sub-threshold error suppression"); b.grid(alpha=0.25); b.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_heterogeneity(res: Dict, path: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.array([p if np.isscalar(p) else p[1] for p in res["params"]], float)
    ax.errorbar(x, res["mismatched"], res["mismatched_sd"], fmt="s-", color=PAL[3],
                capsize=3, label="decoder assumes uniform noise")
    ax.errorbar(x, res["informed"], res["informed_sd"], fmt="o-", color=PAL[0],
                capsize=3, label="decoder knows per-qubit rates")
    ax.set_xlabel(xlabel); ax.set_ylabel(f"logical error (d={res['d']}, {res['d']} rounds)")
    ax.set_title(f"Non-uniform noise, mean $p$ fixed at {res['p']}")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_detection_map(dm: Dict, path: str):
    prof = dm["profile"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    co = prof.coords
    qs = sorted(co)
    sc = axs[0].scatter([co[q][0] for q in qs], [co[q][1] for q in qs],
                        c=[prof.m(q) for q in qs], cmap="viridis", s=140, edgecolor="k")
    plt.colorbar(sc, ax=axs[0], label="error-rate multiplier")
    axs[0].set_title("Hardware profile (hot data qubits)")
    for ax, key, ttl in ((axs[1], "det_uniform", "Syndrome firing rate: uniform device"),
                         (axs[2], "det_true", "Syndrome firing rate: hot-spot device")):
        agg = {}
        for (x, y, t), v in dm[key].items():
            agg.setdefault((x, y), []).append(v)
        xs, ys = zip(*agg.keys()); vs = [np.mean(v) for v in agg.values()]
        s = ax.scatter(xs, ys, c=vs, cmap="magma", s=260, marker="s", edgecolor="k",
                       vmin=0, vmax=max(max(vs), max(np.mean(v) for v in
                                        _agg(dm["det_true"]).values())))
        plt.colorbar(s, ax=ax, label="P(detection event)")
        ax.set_title(ttl)
    for ax in axs:
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _agg(det):
    agg = {}
    for (x, y, t), v in det.items():
        agg.setdefault((x, y), []).append(v)
    return agg
