import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
import open_clip
import webdataset as wds
import io
import os
import csv

from typing import Literal, Tuple, List, Union, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression

from transformers import AutoTokenizer, AutoModelForCausalLM

def get_features(
    model: torch.nn.Module,
    preprocess: torch.nn.Module,
    image_input: list[Image],
    text_input: list[str],
    normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    model.eval()
    image = torch.stack([preprocess(i) for i in image_input]).to("cuda")
    text = tokenizer(text_input).to("cuda")

    with torch.no_grad(), torch.autocast("cuda"):
        image_features = model.encode_image(image, normalize=normalize)
        text_features = model.encode_text(text, normalize=normalize)

    return (image_features.to(torch.float64), text_features.to(torch.float64))


def get_probs(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    model: torch.nn.Module,
    model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP",
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    if model_type == "CLIP":
        text_probs = (model.logit_scale.exp() * image_features @ text_features.T).softmax(dim=-1)
        image_probs = (model.logit_scale.exp() * text_features @ image_features.T).softmax(dim=-1)
    elif model_type == "SIGLIP":
        logits = model.logit_scale.exp() * image_features @ text_features.T
        if model.logit_bias is not None:
            logits += model.logit_bias
        text_probs = F.sigmoid(logits)
        image_probs = F.sigmoid(logits.T)

    return (image_probs, text_probs)


def get_mean_features(
    features: torch.Tensor, # (num_samples, num_dims)
    ) -> torch.Tensor: # (num_dims,)
    return features.mean(dim=0)


def get_kl(
    prior: torch.Tensor,     # (num_text_samples,)
    posterior: torch.Tensor, # (num_image_samples, num_text_samples,)
    eps: float = 1e-12,
    ) -> torch.Tensor:           # (num_image_samples,)
    assert prior.dim() == 1
    assert posterior.dim() == 2

    # 事前を後段でブロードキャスト可能な形に
    prior = prior.unsqueeze(0)
    
    # ratio = (p + eps)/(q + eps)
    ratio = (posterior + eps) / (prior + eps)
    
    # 要素ごとの KL 項: p * log(p/q)
    kl_elementwise = posterior * ratio.log()
    
    # posterior=0 の箇所を強制的に 0 に置き換え => 0log0=0 を実現
    kl_elementwise[posterior == 0] = 0.0
    
    return kl_elementwise.sum(dim=-1)


def get_centerized_norm(
    features: torch.Tensor, # (num_samples, num_dims)
    center: torch.Tensor, # (num_dims,)
    ) -> torch.Tensor:  # (num_samples,)
    return (features - center).norm(dim=-1)


def compute_G(
    features: torch.Tensor, # (num_samples, num_dims)
    center: torch.Tensor, # (num_dims,)
    prior: torch.Tensor, # (num_samples,)
    ): # (num_dims, num_dims)

    X = features - center # (num_samples, num_dims)

    # G = ∑₍i₎ p[i] * (X[i] ⊗ X[i]) を計算
    # ここでは (X.T * p) @ X として実装
    G = (X.t() * prior) @ X
    return G


def plot_kl_norm(
    norm: torch.Tensor, # make sure it's squared
    kl: torch.Tensor,
    title: str,
    model_name: str,
    save_dir: str = ".",
    prob_estimated: torch.Tensor = None,
    ) -> None:

    save_dir = f"{save_dir}/plots"

    # make dir
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots()
    if prob_estimated is not None:
        sc = ax.scatter(
            norm, 2 * kl, 
            c=prob_estimated, 
            cmap='plasma',
            norm=mcolors.LogNorm(vmin=prob_estimated.min(), vmax=prob_estimated.max()),
            marker='.', s=5,
            )
        plt.colorbar(sc, label='LM prob')
    else:
        ax.scatter(norm, 2 * kl, marker='.', s=5,)

    # scikit-learn で回帰
    X = norm.reshape(-1, 1)  # sklearn 用に 2次元に
    model = LinearRegression()
    model.fit(X, 2 * kl)
    slope = model.coef_[0]
    intercept = model.intercept_

    # R^2 を計算
    r2 = model.score(X, 2 * kl)

    # 回帰直線を描画
    x_line = np.linspace(norm.min(), norm.max(), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='gray',
            label=f'y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r2:.3f}')

    # 凡例の表示
    plt.legend()

    ax.set_xlabel("$Norm^2$")
    ax.set_ylabel("2KL")
    ax.set_title(title)

    fig.tight_layout()
    plt.savefig(f"{save_dir}/{title.replace(' ', '_')}.png")

    print(f"Saved {title} plot to {save_dir}")


def save_top_and_bottom_value_samples(
    values: np.ndarray,
    value_name: str,
    sample_input: List[Any],
    model_name: str,
    top_k: int = 100,
    save_dir: str = ".",
    ) -> None:

    save_dir = f"{save_dir}/{value_name}"
    os.makedirs(save_dir, exist_ok=True)

    # トップとボトムのインデックスを取得
    top_k_indices = np.argsort(values)[-top_k:]
    bottom_k_indices = np.argsort(values)[:top_k]

    # テキストの場合はCSVで保存
    if isinstance(sample_input[0], str):
        # トップサンプルをCSV用にリスト作成
        top_samples = []
        for i, idx in enumerate(top_k_indices):
            rank = top_k - i  # 高い値がより上位とする
            top_samples.append([rank, values[idx], sample_input[idx]])
            print(f"Top {rank}: {sample_input[idx]}")
        with open(f"{save_dir}/top_samples.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", value_name, "Sample"])
            writer.writerows(top_samples)
        
        # ボトムサンプルをCSV用にリスト作成
        bottom_samples = []
        for i, idx in enumerate(bottom_k_indices):
            rank = i + 1  # 低い値がより下位とする
            bottom_samples.append([rank, values[idx], sample_input[idx]])
            print(f"Bottom {rank}: {sample_input[idx]}")
        with open(f"{save_dir}/bottom_samples.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", value_name, "Sample"])
            writer.writerows(bottom_samples)
    
    # 画像の場合はPNGで保存
    elif isinstance(sample_input[0], Image.Image):
        for i, idx in enumerate(top_k_indices):
            sample_input[idx].save(f"{save_dir}/top_{top_k-i}.png")
        for i, idx in enumerate(bottom_k_indices):
            sample_input[idx].save(f"{save_dir}/bottom_{i+1}.png")
    
    else:
        raise ValueError("sample_input should be either str or Image")

    print(f"Saved top and bottom {value_name} samples to {save_dir}")

    
if __name__ == "__main__":
    print("Start")

    ### SELECT PARAMS

    ############## EXP 11 ##############

    normalize = False
    model_arch = 'ViT-B-32'
    model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP"
    ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_22-14_17_14-model_ViT-B-32-lr_0.001-b_320-j_8-p_amp_bf16/checkpoints/epoch_64.pt"

    # normalize = True
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_22-14_20_09-model_ViT-B-32-lr_0.001-b_320-j_8-p_amp_bf16/checkpoints/epoch_64.pt"
    
    ####### EXP 11(w/ normalize) ########

    # normalize = True
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_22-14_17_14-model_ViT-B-32-lr_0.001-b_320-j_8-p_amp_bf16/checkpoints/epoch_64.pt"

    # normalize = False
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_22-14_20_09-model_ViT-B-32-lr_0.001-b_320-j_8-p_amp_bf16/checkpoints/epoch_64.pt"

    ############### EXP 9 ###############

    # normalize = False
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_17-10_06_09-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_32.pt"

    # normalize = True
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_17-10_06_26-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_32.pt"
    
    ######## EXP 9(w/ normalize) ########

    # normalize = True
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_17-10_06_09-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_32.pt"

    ############### EXP 8 ###############

    # normalize = False
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "CLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_16-14_18_45-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_37.pt"

    # normalize = True
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "CLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_16-14_18_56-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_37.pt"

    ######## EXP 8(w/ normalize) ########

    # normalize = True
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "CLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_16-14_18_45-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_37.pt"

    ############### EXP 2 ###############

    # normalize = False
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "CLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_15-17_55_07-model_ViT-B-32-lr_0.0005-b_320-j_8-p_amp/checkpoints/epoch_32.pt"

    # normalize = True
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "CLIP"
    # ckpt = "/groups/gag51404/fumiyau/repos/clip_sem_info/src/logs/2025_03_15-17_55_23-model_ViT-B-32-lr_0.0005-b_320-j_8-p_amp/checkpoints/epoch_32.pt"

    ############# PRETRAINED #############

    # normalize = True
    # model_arch = 'ViT-B-32'
    # model_type: Literal["CLIP", "SIGLIP"] = "CLIP"
    # ckpt = "pretrained:laion2b_s34b_b79k"

    # normalize = True
    # model_arch = 'ViT-B-16-SigLIP'
    # model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP"
    # ckpt = "pretrained:webli"

    #####################################

    data_dir = "/groups/gag51404/fumiyau/data/cc12m/cc12m/{00000..01242}.tar" 
    # data_dir = "/groups/gag51404/fumiyau/data/cc3m/cc3m_train/{00000..00331}.tar"
    num_samples = 4000

    lm_name: Optional[str] = None

    normalize_prob: bool = True # 正規化して経験分布にする
    prior_type: Literal["UNIFORM", "MODEL", "MEAN"] = "MEAN"

    # make save dir
    model_name_list = ckpt.split("/")
    if len(model_name_list) > 3:
        model_name = model_name_list[-3]
    else:
        model_name = ckpt

    if model_type == "SIGLIP":
        save_dir = f"output/vis/{data_dir.split('/')[-2]}_{str(num_samples)}/{model_type}/{model_name}/{prior_type}/logit_normalize_{str(normalize)}_prob_normalize_{str(normalize_prob)}"
    elif model_type == "CLIP":
        save_dir = f"output/vis/{data_dir.split('/')[-2]}_{str(num_samples)}/{model_type}/{model_name}/{prior_type}/normalize_{str(normalize)}"
    else:
        raise ValueError(f"model_type should be either CLIP or SIGLIP: {model_type}")
    
    os.makedirs(save_dir, exist_ok=True)

    # load data
    dataset = wds.WebDataset(data_dir).shuffle(10000)

    text_input = []
    image_input = []

    count = 0
    for sample in dataset:
        if isinstance(sample["txt"], bytes):
            t = sample["txt"].decode("utf-8")
            print(t)
        elif isinstance(sample["txt"], str):
            t = sample["txt"]
        else:
            raise ValueError(f"sample['txt'] should be either bytes or str: {type(sample['txt'])}")
        text_input.append(t)
        image_input.append(Image.open(io.BytesIO(sample["jpg"])))
        count += 1
        if count >= num_samples:
            break

    print("Data loaded")

    text_lm_probs = None
    if lm_name is not None:
        print("Start loading language model")
        # Estimate sentence prob via language model
        lm_tokenizer = AutoTokenizer.from_pretrained(lm_name)
        lm_model = AutoModelForCausalLM.from_pretrained(lm_name).to("cuda")
        lm_model.eval()

        text_lm_probs = torch.zeros(len(text_input))
        for idx, t in enumerate(text_input):
            input_ids = lm_tokenizer.encode(t, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = lm_model(input_ids, labels=input_ids)
            loss = outputs.loss
            text_lm_probs[idx] = torch.exp(-loss)

        print("Language model loaded.")
    
    print("Start loading CLIP model")

    # load model
    model_kwargs = {}
    if model_type == "SIGLIP":
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10

    pretrained: bool = "pretrained:" in ckpt
    if pretrained:
        ckpt = ckpt[len("pretrained:"):]
        model_kwargs['pretrained'] = ckpt

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_arch, 
        **model_kwargs
        )

    if not pretrained:
        checkpoint = torch.load(ckpt, weights_only=True)
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
    
    model.eval().to("cuda")
    tokenizer = open_clip.get_tokenizer(model_arch)

    print("Model loaded. Start getting features")

    # get features
    image_features, text_features = get_features(
        model, preprocess, 
        image_input, 
        text_input, 
        normalize=False
        )

    print("Features loaded. Start computing metrics")
    center_text = get_mean_features(text_features)
    center_image = get_mean_features(image_features)

    # concat center and features
    cat_text = torch.cat([center_text.unsqueeze(0).detach().clone(), text_features.detach().clone()], dim=0)
    cat_image = torch.cat([center_image.unsqueeze(0).detach().clone(), image_features.detach().clone()], dim=0)

    if normalize:
        cat_text = F.normalize(cat_text, dim=-1)
        cat_image = F.normalize(cat_image, dim=-1)

    # computer prob
    image_probs, text_probs = get_probs(cat_image, cat_text, model, model_type)

    # decompose prior and posterior
    prior_text, posterior_text = text_probs[0,1:], text_probs[1:,1:]
    prior_image, posterior_image = image_probs[0,1:], image_probs[1:,1:]

    # make sure summed prob is 1
    if model_type == "SIGLIP" and normalize_prob:
        prior_text = prior_text / (prior_text.sum(dim=-1, keepdim=True) + 1e-18)
        prior_image = prior_image / (prior_image.sum(dim=-1, keepdim=True) + 1e-18)
        posterior_text = posterior_text / (posterior_text.sum(dim=-1, keepdim=True) + 1e-18)
        posterior_image = posterior_image / (posterior_image.sum(dim=-1, keepdim=True) + 1e-18)

    if prior_type == "UNIFORM":
        prior_text = torch.ones_like(prior_text) / len(prior_text)
        prior_image = torch.ones_like(prior_image) / len(prior_image)
    elif prior_type == "MODEL":
        pass
    elif prior_type == "MEAN":
        prior_text = get_mean_features(posterior_text)
        prior_image = get_mean_features(posterior_image)
    else:
        raise ValueError(f"prior_type should be either UNIFORM or MODEL: {prior_type}")

    # get raw norm
    norm_text = text_features.norm(dim=-1).detach().cpu().numpy()
    norm_image = image_features.norm(dim=-1).detach().cpu().numpy()

    # get centerized norm
    norm_text_centerized = get_centerized_norm(text_features, center_text).detach().cpu().numpy()
    norm_image_centerized = get_centerized_norm(image_features, center_image).detach().cpu().numpy()

    # get kl
    kl_image = get_kl(prior_text, posterior_text).detach().cpu().numpy()
    kl_text = get_kl(prior_image, posterior_image).detach().cpu().numpy()

    # get G
    G_text = compute_G(text_features, center_text, prior_text)
    G_image = compute_G(image_features, center_image, prior_image)

    # get norm with G
    centerized_text_features = text_features - center_text
    centerized_image_features = image_features - center_image

    norm_text_G = torch.diag((centerized_text_features @ G_text @ centerized_text_features.T),0).detach().cpu().numpy()
    norm_image_G = torch.diag((centerized_image_features @ G_image @ centerized_image_features.T),0).detach().cpu().numpy()

    # plot
    plot_kl_norm(norm_text ** 2, kl_text, "Text", model_name, save_dir=save_dir, prob_estimated=text_lm_probs)
    plot_kl_norm(norm_image ** 2, kl_image, "Image", model_name, save_dir=save_dir)
    plot_kl_norm(norm_text_centerized ** 2, kl_text, "Text_centerized", model_name, save_dir=save_dir, prob_estimated=text_lm_probs)
    plot_kl_norm(norm_image_centerized ** 2, kl_image, "Image_centerized", model_name, save_dir=save_dir)
    plot_kl_norm(norm_text_G, kl_text, "Text with G", model_name, save_dir=save_dir, prob_estimated=text_lm_probs)
    plot_kl_norm(norm_image_G, kl_image, "Image with G", model_name, save_dir=save_dir)

    # save top and bottom samples
    save_top_and_bottom_value_samples(kl_text, "kl_text", text_input, model_name, save_dir=save_dir)
    save_top_and_bottom_value_samples(kl_image, "kl_image", image_input, model_name, save_dir=save_dir)
    save_top_and_bottom_value_samples(norm_text ** 2, "norm_text", text_input, model_name, save_dir=save_dir)
    save_top_and_bottom_value_samples(norm_image ** 2, "norm_image", image_input, model_name, save_dir=save_dir)
    save_top_and_bottom_value_samples(norm_text_centerized ** 2, "norm_text_centerized", text_input, model_name, save_dir=save_dir)
    save_top_and_bottom_value_samples(norm_image_centerized ** 2, "norm_image_centerized", image_input, model_name, save_dir=save_dir)
    save_top_and_bottom_value_samples(norm_text_G, "norm_text_G", text_input, model_name, save_dir=save_dir)
    save_top_and_bottom_value_samples(norm_image_G, "norm_image_G", image_input, model_name, save_dir=save_dir)

    print(prior_text.sum(dim=-1), prior_text.shape)
    print(prior_image.sum(dim=-1), prior_image.shape)
    print(posterior_text.sum(dim=-1), posterior_text.shape)
    print(posterior_image.sum(dim=-1), posterior_image.shape)