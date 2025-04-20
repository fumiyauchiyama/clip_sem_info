import os
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def dump_config(config, path):
    yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(config))
    with open(path, "w") as f:
        f.write(yaml_dump)


def calc_r2(x, y):
    x = np.array(x)
    y = np.array(y)
    lr = LinearRegression()
    n = len(x)
    X = x.reshape(n, 1)
    lr.fit(X, y)
    # print('slope=',lr.coef_,'intercept=',lr.intercept_)
    predict_x = lr.predict(X)
    # print('predict=',predict_x)
    resudial_y = y - predict_x
    # print('resudial=',resudial_y)
    r2 = lr.score(X, y)
    # print('r_square=',r2)
    return r2


def plot_grid(
    metrics_lists: Dict[str, List[Union[int, float]]],
):
    n_metrics = len(metrics_lists)

    fig, axes = plt.subplots(
        n_metrics, n_metrics, figsize=(3 * n_metrics, 3 * n_metrics)
    )

    # 全組み合わせで散布図をプロット
    for i, (name_x, x) in enumerate(metrics_lists.items()):
        for j, (name_y, y) in enumerate(metrics_lists.items()):
            ax = axes[i, j]
            ax.scatter(x, y, marker=".")
            r2 = calc_r2(x, y)
            ax.text(0.05, 0.95, f"$R^2={r2:.2f}$", transform=ax.transAxes, va="top")
            ax.set_xlabel(name_x)
            ax.set_ylabel(name_y)
            ax.set_title(f"{name_x} vs {name_y}")

    plt.tight_layout()
    plt.show()


def save_grid(
    metrics_lists: Dict[str, List[Union[int, float]]],
    save_name_prefix: str = "grid",
    save_dir: str = "./output",
    cfg: Any = None,
    save_each: bool = False,
):
    n_metrics = len(metrics_lists)

    fig, axes = plt.subplots(
        n_metrics, n_metrics, figsize=(3 * n_metrics, 3 * n_metrics)
    )

    # 全組み合わせで散布図をプロット
    for i, (name_x, x) in enumerate(metrics_lists.items()):
        for j, (name_y, y) in enumerate(metrics_lists.items()):
            ax = axes[i, j]
            ax.scatter(x, y, marker=".")
            r2 = calc_r2(x, y)
            ax.text(0.05, 0.95, f"$R^2={r2:.2f}$", transform=ax.transAxes, va="top")
            ax.set_xlabel(name_x)
            ax.set_ylabel(name_y)
            ax.set_title(f"{name_x} vs {name_y}")

    plt.tight_layout()

    date_str = time.strftime("%Y%m%d")
    save_dir = os.path.join(save_dir, date_str)
    os.makedirs(save_dir, exist_ok=True)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    save_name = f"{time_str}_{save_name_prefix}"
    save_name = save_name.replace(" ", "_").replace(".", "_")
    save_path = os.path.join(save_dir, f"{save_name}.png")

    plt.savefig(save_path)
    plt.close(fig)

    if save_each:
        for name_x, x in metrics_lists.items():
            for name_y, y in metrics_lists.items():
                fig_single, ax_single = plt.subplots(figsize=(4, 4))
                ax_single.scatter(x, y, marker=".")
                r2 = calc_r2(x, y)
                ax_single.text(0.05, 0.95, f"$R^2={r2:.2f}$", transform=ax_single.transAxes, va="top")
                ax_single.set_xlabel(name_x)
                ax_single.set_ylabel(name_y)
                # ax_single.set_title(f"{name_x} vs {name_y}")
                plt.tight_layout()

                single_name = (
                    f"{name_x}_vs_{name_y}"
                    .replace(" ", "_")
                    .replace(".", "_")
                )
                save_dir_each = os.path.join(save_dir, save_name)
                os.makedirs(save_dir_each, exist_ok=True)
                single_path = os.path.join(save_dir_each, f"{single_name}.png")
                plt.savefig(single_path)
                plt.close(fig_single)

    if cfg is not None:
        config_path = os.path.join(save_dir, f"{save_name}.yaml")
        dump_config(cfg, config_path)
