"""
reporting.py
------------
Tabular and graphical reporting of mutual information results.

Generalises the notebook's fixed "camera / lidar / fused, InfoNCE vs MINE"
display into something that works for any set of representations and any set of
estimators. Results are passed in as a nested mapping

    results[estimator_name][representation_name] = mi_nats

so adding a third estimator or a fourth representation needs no code change.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _verdict(delta: float, tol: float = 0.01) -> str:
    if delta > tol:
        return f'fusion ADDS information (+{delta:.4f} nats)'
    if delta >= -tol:
        return f'fusion is REDUNDANT ({delta:+.4f} nats)'
    return f'fusion LOSES information ({delta:+.4f} nats)'


def format_table(results: Dict[str, Dict[str, float]],
                 representations: Optional[List[str]] = None) -> str:
    """
    Build a fixed-width comparison table as a string.

    Parameters
    ----------
    results         : estimator -> {representation -> mi_nats}.
    representations : row order; defaults to the keys of the first estimator.

    Returns
    -------
    str (multi-line). Columns are estimators; rows are representations.
    """
    estimators = list(results.keys())
    if representations is None:
        representations = list(next(iter(results.values())).keys())

    name_w = max([len('representation')] + [len(r) for r in representations])
    col_w = max(12, max(len(e) for e in estimators))

    header = f'{"representation":<{name_w}}  ' + '  '.join(f'{e:>{col_w}}' for e in estimators)
    sep = '-' * len(header)
    lines = [sep, header, sep]
    for rep in representations:
        cells = '  '.join(f'{results[e].get(rep, float("nan")):>{col_w}.4f}' for e in estimators)
        lines.append(f'{rep:<{name_w}}  {cells}')
    lines.append(sep)
    return '\n'.join(lines)


def fusion_summary(results: Dict[str, Dict[str, float]], fused_key: str,
                   unimodal_keys: List[str], tol: float = 0.01) -> str:
    """
    Per-estimator delta_I verdict plus an agreement check.

    Parameters
    ----------
    results       : estimator -> {representation -> mi_nats}.
    fused_key     : representation name of the fused feature.
    unimodal_keys : representation names of the single modalities.
    tol           : dead-zone half-width (nats) for the REDUNDANT verdict.

    Returns
    -------
    str (multi-line) summarising delta_I for each estimator and whether they
    agree on the sign.
    """
    lines: List[str] = []
    deltas = {}
    for est, mi in results.items():
        best_uni = max(mi[k] for k in unimodal_keys)
        delta = mi[fused_key] - best_uni
        deltas[est] = delta
        lines.append(f'  {est:<10} delta_I = {delta:+.4f}  ->  {_verdict(delta, tol)}')

    signs = {('+' if d > tol else '-' if d < -tol else '0') for d in deltas.values()}
    lines.append('')
    if len(signs) == 1:
        lines.append('All estimators AGREE on the direction of fusion gain '
                     '(conclusion is estimator-agnostic).')
    else:
        lines.append('Estimators DISAGREE on the direction of fusion gain '
                     '(treat with caution; consider more epochs / samples).')
    return '\n'.join(lines)


def plot_comparison(results: Dict[str, Dict[str, float]], out_path: str,
                    representations: Optional[List[str]] = None,
                    title: str = 'Mutual information by representation') -> str:
    """
    Grouped bar chart: one group per representation, one bar per estimator.

    Parameters
    ----------
    results         : estimator -> {representation -> mi_nats}.
    out_path        : where to write the PNG.
    representations : x-axis order; defaults to the first estimator's keys.
    title           : figure title.

    Returns
    -------
    str out_path.

    Raises
    ------
    ImportError if matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError('plot_comparison requires matplotlib.') from exc

    estimators = list(results.keys())
    if representations is None:
        representations = list(next(iter(results.values())).keys())

    x = np.arange(len(representations))
    n_est = len(estimators)
    width = 0.8 / max(n_est, 1)

    fig, ax = plt.subplots(figsize=(1.6 * len(representations) + 3, 5))
    for i, est in enumerate(estimators):
        vals = [results[est].get(rep, float('nan')) for rep in representations]
        offset = (i - (n_est - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=est, alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(representations)
    ax.set_ylabel('MI lower bound (nats)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path
