import os
import io
import webdataset as wds
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal
import argparse

import numpy as np
import torch
import open_clip
from PIL import Image
import random


# ==============================
# データを評価するための抽象クラス
# ==============================
class Evaluator(ABC):
    @abstractmethod
    def __init__(self, threshold: float = None, percentile: float = None):
        # threshold か percentile のどちらか一方だけ指定されている想定
        assert threshold is not None or percentile is not None
        if percentile is not None:
            assert 0 <= percentile <= 1
        self.threshold = threshold
        self.percentile = percentile

    @abstractmethod
    def filter_sample(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """バッチ単位でフィルタを行い、通過したサンプルのみ返す。"""
        pass

class RandomEvaluator(Evaluator):
    """単純にランダムでサンプルを残すだけの例."""
    def __init__(self, threshold: float = None, percentile: float = None):
        super().__init__(threshold, percentile)

    def filter_sample(self, samples: List[dict]) -> List[dict]:
        """ランダムサンプリングでサンプルをフィルタ."""
        if self.threshold is not None:
            # threshold を「残すサンプル数」として扱う例
            k = int(self.threshold)
        elif self.percentile is not None:
            k = int(len(samples) * self.percentile)
        # k が範囲外にならないようにチェック
        k = min(k, len(samples))
        indices = random.sample(range(len(samples)), k)
        return [samples[i] for i in indices]

class RawNormEvaluator(Evaluator):
    """画像とテキストの特徴量ノルムを用いてフィルタリングする例。"""

    def __init__(
        self,
        threshold: float = None,
        percentile: float = None,
        model_arch: str = "ViT-B-32",
        model_type: Literal["CLIP", "SIGLIP"] = "SIGLIP",
        ckpt: str = None,
        normalize: bool = False,
    ):
        super().__init__(threshold, percentile)
        self.normalize = normalize

        # モデルの読み込み
        model_kwargs = {}
        if model_type == "SIGLIP":
            model_kwargs["init_logit_scale"] = np.log(10)
            model_kwargs["init_logit_bias"] = -10

        # "pretrained:" プレフィックスがついていたら open_clip の自前プリトレ重みを使う
        pretrained: bool = (ckpt is not None) and ckpt.startswith("pretrained:")
        if pretrained:
            ckpt = ckpt[len("pretrained:"):]
            model_kwargs["pretrained"] = ckpt

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_arch,
            **model_kwargs
        )

        if ckpt is not None and not pretrained:
            checkpoint = torch.load(ckpt, map_location="cpu")
            sd = checkpoint["state_dict"]
            # 分散学習などで 'module.' がついている場合は除去
            if next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            self.model.load_state_dict(sd)

        self.model.eval().to("cuda")
        self.tokenizer = open_clip.get_tokenizer(model_arch)

    def get_features(self, images: List[Image.Image], texts: List[str]) -> (torch.Tensor, torch.Tensor):
        """画像とテキストを model に通し、特徴ベクトルを返す。"""
        # PIL 画像をテンソル化
        image_tensors = [self.preprocess(img) for img in images]
        image_tensors = torch.stack(image_tensors).to("cuda")

        # テキストをトークナイズ
        text_tokens = self.tokenizer(texts).to("cuda")

        with torch.no_grad(), torch.autocast("cuda"):
            image_features = self.model.encode_image(image_tensors, normalize=self.normalize)
            text_features = self.model.encode_text(text_tokens, normalize=self.normalize)

        return image_features.float(), text_features.float()

    def filter_sample(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """各サンプルの __decoded_image__, __decoded_text__ の特徴量ノルムを算出し、条件を満たしたら通過。"""
        filtered = []

        for sample in samples:
            # デコード済みデータ
            images: List[Image.Image] = sample.get("__decoded_image__", [])
            texts: List[str] = sample.get("__decoded_text__", [])

            if len(images) == 0 or len(images) != len(texts):
                # 画像とテキストの数が揃っていない場合はスキップ
                continue

            # 特徴量取得
            image_features, text_features = self.get_features(images, texts)

            # ノルムを計算して平均
            norm_image = image_features.norm(dim=-1).cpu().numpy()
            norm_text = text_features.norm(dim=-1).cpu().numpy()
            norm_avg = (norm_image + norm_text) / 2

            # threshold or percentile でフィルタ
            if self.threshold is not None:
                # 平均ノルムが threshold 以上のものが1つでもあれば通過させる例
                if (norm_avg >= self.threshold).any():
                    filtered.append(sample)
            else:
                # percentile の場合: norm_avg の上位X%に当たる閾値を計算し、それを超えるか
                # ここではバッチ内で計算するだけなので厳密には一度にまとめてやるか、など要検討
                p_value = np.percentile(norm_avg, (1 - self.percentile) * 100)
                if (norm_avg >= p_value).any():
                    filtered.append(sample)

        return filtered

# ==============================
# デコード用の関数例
# ==============================
def decode_for_evaluation(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    元の生バイト ("jpg" 等) はそのまま保持しつつ、
    評価用に PIL 画像やテキストを __decoded_* に格納する例。
    """
    # 画像ファイルをデコード
    if "jpg" in sample:  # or "png", "jpeg", etc.
        try:
            bio = io.BytesIO(sample["jpg"])  # 生バイトをメモリで読み込み
            img = Image.open(bio).convert("RGB")
            # 複数画像を想定しない場合はリストで包んでおく
            sample.setdefault("__decoded_image__", []).append(img)
        except Exception:
            pass

    # テキストも同様に
    # 例: "txt" キーにバイナリで入っている場合
    if "txt" in sample:
        try:
            decoded_txt = sample["txt"].decode("utf-8", errors="replace")
            # こちらも複数テキストを想定しないならリストで包む
            sample.setdefault("__decoded_text__", []).append(decoded_txt)
        except Exception:
            pass

    return sample

# ==============================
# メイン部分
# ==============================
def main(args):
    # シャード出力ファイル名テンプレート
    output_template = os.path.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # 元の形式を維持するため、decode("pil") はしない
    # 代わりに .map(decode_for_evaluation) で必要な分だけ手動デコードする
    dataset = wds.WebDataset(args.input_tar_path).map(decode_for_evaluation)

    # Evaluator の用意
    if args.evaluator == "RandomEvaluator":
        evaluator = RandomEvaluator(args.threshold, args.percentile)
    elif args.evaluator == "RawNormEvaluator":
        evaluator = RawNormEvaluator(
            threshold=args.threshold,
            percentile=args.percentile,
            model_arch=args.model_arch,
            model_type=args.model_type,
            ckpt=args.ckpt,
            normalize=args.normalize,
        )
    else:
        raise ValueError(f"Unsupported evaluator: {args.evaluator}")

    pass_count = 0
    buffer = []

    # TarWriter の shard_size でシャード分割
    with wds.ShardWriter(output_template, maxsize=args.shard_size) as writer:
        for sample in dataset:
            # サンプルごとにバッファに追加
            buffer.append(sample)
            # バッファが一定数たまったらフィルタリング
            if len(buffer) >= args.batch_size:
                filtered = evaluator.filter_sample(buffer)
                pass_count += len(filtered)
                for fs in filtered:
                    # fs は {"jpg": ..., "txt": ..., ...} + {"__decoded_*": ...} が混在
                    # ただし TarWriter に書くときは、「元バイナリ」も含まれる dict そのものを渡す
                    writer.write(fs)
                buffer = []

        # バッファ残り分を処理
        if buffer:
            filtered = evaluator.filter_sample(buffer)
            pass_count += len(filtered)
            for fs in filtered:
                writer.write(fs)

    print(f"フィルタリング完了: 出力先 => {args.output_dir}")
    print(f"通過サンプル数: {pass_count}")

# ==============================
# エントリポイント
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 入出力
    parser.add_argument("--input_tar_path", type=str, default="/groups/gag51404/fumiyau/data/cc12m/cc12m/{00000..01242}.tar")
    parser.add_argument("--output_dir", type=str, default="outdir")
    parser.add_argument("--output_name", type=str, default="filtered_cc12m-%06d.tar")
    parser.add_argument("--shard_size", type=int, default=1024*1024*1024)  # 1GB
    # バッチサイズ
    parser.add_argument("--batch_size", type=int, default=1000)
    # Evaluator
    parser.add_argument("--evaluator", type=str, default="RawNormEvaluator")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--percentile", type=float, default=None)
    # モデル設定
    parser.add_argument("--model_arch", type=str, default="ViT-B-32")
    parser.add_argument("--model_type", type=str, default="SIGLIP")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    main(args)
