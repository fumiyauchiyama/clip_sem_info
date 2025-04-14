import argparse
from io import BytesIO
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import requests
import torch
from datasets import load_dataset
from PIL import Image
from src.clip_sem_info.distance import (
    EmbeddingDistance,
    OTAlgorithm,
    kl_divergence,
    ot_disrance,
)
from src.clip_sem_info.predictor import CLIPPredictor
from src.clip_sem_info.visualize import save_grid
from torchtyping import TensorType as TT
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, # type: ignore
    AutoTokenizer, # type: ignore
    CLIPModel, # type: ignore
    CLIPProcessor, # type: ignore
)


# for embedding norm helper
def compute_G(
    features: TT["num_sample", "hidden_dim"],  # noqa: F821
    center: TT["hidden_dim"],  # noqa: F821
    prior: TT["num_sample"],  # noqa: F821
) -> TT["hidden_dim", "hidden_dim"]:  # noqa: F821
    X: TT["num_sample", "hidden_dim"] = features - center  # noqa: F821 # type: ignore

    # G = ∑₍i₎ p[i] * (X[i] ⊗ X[i]) を計算
    # ここでは (X.T * p) @ X として実装
    G = (X.t() * prior) @ X
    return G  # type: ignore


def calc_metrics(
    u_hat: TT["hidden_dim"],  # noqa: F821
    eval_data: Union[List[str], List[Image.Image]],
    encode_fn: Union[
        Callable[[List[str]], TT["num_batch", "hidden_dim"]],  # noqa: F821
        Callable[[List[Image.Image]], TT["num_batch", "hidden_dim"]],  # noqa: F821
    ],  # noqa: F821
    prior: TT["num_sample"],  # noqa: F821
    sample_emb: TT["num_sample", "hidden_dim"],  # noqa: F821
    posterior_fn: Callable[[TT["hidden_dim"]], TT["num_sample"]],  # noqa: F821
    lm_tokenizer: AutoTokenizer,
    lm: AutoModelForCausalLM,
    lists: Optional[Dict[str, List[Union[int, float]]]] = None,
    ot_algorithm: OTAlgorithm = OTAlgorithm.sinkhorn,
    device: str = "cuda",
):
    if lists is None:
        lists = dict()
    lists['$OT_{angle}$'] = []
    lists['$OT_{euclid}$'] = []
    lists['$||u||^2$'] = []
    lists['$||u-u_0||^2$'] = []
    lists['$(u-u_0)^T G (u-u_0)$'] = []
    lists['$2KL$'] = []
    if isinstance(eval_data[0], str):
        lists['chars'] = []
        lists['words'] = []
        lists['$-log(p)$'] = []

    v_hat = torch.mean(sample_emb, dim=0).to(device)
    G = compute_G(sample_emb, v_hat, prior).to(device)  # type: ignore

    for i, condition in enumerate(eval_data):
        u = encode_fn([condition])[0]  # type: ignore
        posterior = posterior_fn(u)  # type: ignore
        ot_dist_a = ot_disrance(
            sample_emb,
            sample_emb,
            posterior,
            prior,
            algorithm=ot_algorithm,
            distance_type=EmbeddingDistance.angular,
        )
        ot_dist_e = ot_disrance(
            sample_emb,
            sample_emb,
            posterior,
            prior,
            algorithm=ot_algorithm,
            distance_type=EmbeddingDistance.euclidean,
        )
        kl_dist = kl_divergence(posterior, prior)
        raw_norm = torch.linalg.norm(u, dim=0)
        centerized_u = u - u_hat
        centerized_norm = torch.linalg.norm(centerized_u, dim=0)
        norm_G = (centerized_u[None, :] @ G @ centerized_u[:, None]).item()

        lists['$OT_{angle}$'].append(ot_dist_a.item())
        lists['$OT_{euclid}$'].append(ot_dist_e.item())
        lists['$2KL$'].append(2 * kl_dist.item())
        lists['$||u||^2$'].append(raw_norm.item() ** 2)
        lists['$||u-u_0||^2$'].append(centerized_norm.item() ** 2)
        lists['$(u-u_0)^T G (u-u_0)$'].append(norm_G)
        if isinstance(condition, str):
            text = condition
            len_words = len(text.split())
            lists['chars'].append(len(text))
            lists['words'].append(len_words)
            input_ids = lm_tokenizer.encode(text, return_tensors="pt").to(lm.device) # type: ignore
            with torch.no_grad():
                output = lm(input_ids, labels=input_ids, num_items_in_batch=1) # type: ignore
            loss = output.loss
            lists['$-log(p)$'].append(loss.item())
        else:
            text = ""
            len_words = 0
            loss = 0
        print(
            f"[{i}] ot: {ot_dist_a:.4f},{ot_dist_e:.4f}\tkl: {kl_dist:.4f}\tnorm: {raw_norm:.2f}({centerized_norm:.2f}, {norm_G:.2f})\tlen: {len(text)}({len_words})\ttext: ({loss:.4f}){text}"
        )
    return lists


def main(
    dataset_name: str = "laion/laion400m",
    n_samples: int = 300,
    lm_model_name: str = "meta-llama/Llama-3.2-1B",
):
    # テキストの確率推定
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    lm = AutoModelForCausalLM.from_pretrained(lm_model_name, device_map="auto")

    """## データセットの用意"""
    # stream datasets from hf, and take samples
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    sample_image = []
    sample_text = []
    count = 0
    for data in tqdm(dataset):
        if count >= n_samples:
            break
        url = data["url"]
        try:
            response = requests.get(url, stream=True, timeout=3.5)
            if response.status_code != 200:
                print(
                    f"URLからの取得に失敗しました (ステータスコード: {response.status_code}): {url}, 収集完了数: {count}"
                )
                continue
            image = Image.open(BytesIO(response.content))
            image.verify()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            sample_image.append(image)
            sample_text.append(data["caption"])
            count += 1
        except Exception as e:
            print(f"画像読み込みエラー ({url}): {e}, 収集完了数: {count}")
            continue

    """## 埋め込み・推定事前分布の計算"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    predictor = CLIPPredictor(processor, model, device)  # type: ignore

    predictor.encode_sample_image(sample_image)
    predictor.encode_sample_text(sample_text)
    predictor.calc_image_prior()
    predictor.calc_text_prior()

    print("Image Prior")
    print(predictor.image_prior)
    print("Text Prior")
    print(predictor.text_prior)

    """## Sinkhorn"""

    # 条件づける方
    assert predictor.sample_text_emb is not None
    u_hat = torch.mean(predictor.sample_text_emb, dim=0).to(device)
    eval_data = sample_text
    encode_fn = predictor.encode_text
    # 条件づけられる方
    assert predictor.image_prior is not None
    prior = predictor.image_prior
    assert predictor.sample_image_emb is not None
    sample_emb = predictor.sample_image_emb
    posterior_fn = predictor.calc_image_posterior

    metrics = calc_metrics(
        u_hat,  # type: ignore
        eval_data,
        encode_fn,
        prior,
        sample_emb,
        posterior_fn,
        device=device,
        lm_tokenizer=lm_tokenizer,
        lm=lm,
        ot_algorithm=OTAlgorithm.sinkhorn,
    )
    save_grid(metrics, save_name_prefix="sinkhorn_text")

    # 条件づける方
    assert predictor.sample_image_emb is not None
    u_hat = torch.mean(predictor.sample_image_emb, dim=0).to(device)
    eval_data = sample_image
    encode_fn = predictor.encode_image
    # 条件づけられる方
    assert predictor.text_prior is not None
    prior = predictor.text_prior
    assert predictor.sample_text_emb is not None
    sample_emb = predictor.sample_text_emb
    posterior_fn = predictor.calc_text_posterior

    metrics = calc_metrics(
        u_hat,  # type: ignore
        eval_data,
        encode_fn,
        prior,
        sample_emb,
        posterior_fn,
        device=device,
        lm_tokenizer=lm_tokenizer,
        lm=lm,
        ot_algorithm=OTAlgorithm.sinkhorn,
        lists=metrics,
    )
    save_grid(metrics, save_name_prefix="sinkhorn_image")

    """## EMD"""

    # 条件づける方
    assert predictor.sample_text_emb is not None
    u_hat = torch.mean(predictor.sample_text_emb, dim=0).to(device)
    eval_data = sample_text
    encode_fn = predictor.encode_text
    # 条件づけられる方
    assert predictor.image_prior is not None
    prior = predictor.image_prior
    assert predictor.sample_image_emb is not None
    sample_emb = predictor.sample_image_emb
    posterior_fn = predictor.calc_image_posterior

    metrics = calc_metrics(
        u_hat,  # type: ignore
        eval_data,
        encode_fn,
        prior,
        sample_emb,
        posterior_fn,
        device=device,
        lm_tokenizer=lm_tokenizer,
        lm=lm,
        ot_algorithm=OTAlgorithm.emd,
    )
    save_grid(metrics, save_name_prefix="emd_text")

    # 条件づける方
    assert predictor.sample_image_emb is not None
    u_hat = torch.mean(predictor.sample_image_emb, dim=0).to(device)
    eval_data = sample_image
    encode_fn = predictor.encode_image
    # 条件づけられる方
    assert predictor.text_prior is not None
    prior = predictor.text_prior
    assert predictor.sample_text_emb is not None
    sample_emb = predictor.sample_text_emb
    posterior_fn = predictor.calc_text_posterior

    metrics = calc_metrics(
        u_hat,  # type: ignore
        eval_data,
        encode_fn,
        prior,
        sample_emb,
        posterior_fn,
        device=device,
        lm_tokenizer=lm_tokenizer,
        lm=lm,
        ot_algorithm=OTAlgorithm.emd,
        lists=metrics,
    )
    save_grid(metrics, save_name_prefix="emd_image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="laion/laion400m",
        help="Dataset name to use.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of samples to use.",
    )
    parser.add_argument(
        "--lm_model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Language model name to use.",
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        n_samples=args.n_samples,
        lm_model_name=args.lm_model_name,
    )
