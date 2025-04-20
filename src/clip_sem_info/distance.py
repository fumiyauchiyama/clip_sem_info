from enum import Enum
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import ot
import torch
from geomloss import SamplesLoss
from torch.nn import functional as F
from torchtyping import TensorType as TT


# Distance for Embedding
class EmbeddingDistance(Enum):
    angular = "angular"
    euclidean = "euclidean"


# def angular_distance(x: TT["hidden_dim"], y: TT["hidden_dim"]) -> TT["num_sample"]: # noqa: F821
#     return torch.acos(torch.clamp(F.cosine_similarity(x, y, dim=-1), min=-1, max=1)) # type: ignore


# def euclidean_distance(x: TT["hidden_dim"], y: TT["hidden_dim"]): # noqa: F821
#     return torch.linalg.norm(x - y, dim=-1)


def angular_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    cos_sim = np.sum(x * y, axis=-1) / (norm_x * norm_y)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.arccos(cos_sim)


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - y, axis=-1)


# Distance for Probablistic Distribution
def kl_divergence(
    x_prob: TT["num_sample"],  # noqa: F821
    y_prob: TT["num_sample"],  # noqa: F821
    eps: float = 1e-12,
) -> TT:
    # ratio = (p + eps)/(q + eps)
    ratio = (x_prob + eps) / (y_prob + eps)

    # 要素ごとの KL 項: p * log(p/q)
    kl_elementwise = x_prob * ratio.log()

    # x_prob=0 の箇所を強制的に 0 に置き換え => 0log0=0 を実現
    kl_elementwise[x_prob == 0] = 0.0

    return torch.sum(kl_elementwise)  # type: ignore


class OTAlgorithm(Enum):
    emd = "emd"
    sinkhorn = "sinkhorn"
    geomloss = "geomloss"


Loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)


def ot_disrance(
    x: TT["num_sample", "hidden_dim"],  # noqa: F821
    y: TT["num_sample", "hidden_dim"],  # noqa: F821
    x_prob: TT["num_sample"],  # noqa: F821
    y_prob: TT["num_sample"],  # noqa: F821
    distance_type: EmbeddingDistance = EmbeddingDistance.angular,
    algorithm: OTAlgorithm = OTAlgorithm.sinkhorn,
    reg: float = 5.0,
) -> float:
    if distance_type == EmbeddingDistance.angular:
        cost = angular_distance
    elif distance_type == EmbeddingDistance.euclidean:
        cost = euclidean_distance
    else:
        raise NotImplementedError

    cost_list = []
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            cost_list.append(cost(x[i], y[j]))  # type: ignore
    # C = torch.stack(cost_list).reshape(x.shape[0], y.shape[0])
    C = np.array(cost_list).reshape(x.shape[0], y.shape[0])

    if algorithm == OTAlgorithm.emd:
        return ot.lp.emd2(x_prob, y_prob, C)  # type: ignore
    elif algorithm == OTAlgorithm.sinkhorn:
        return ot.bregman.sinkhorn2(x_prob, y_prob, C, reg)  # type: ignore
    elif algorithm == OTAlgorithm.geomloss:
        return Loss(x_prob, x, y_prob, y)
    else:
        raise NotImplementedError
