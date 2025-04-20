from ast import arg
import os
import requests
from io import BytesIO
import argparse
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Dataset

def main(
        dataset_name,
        n_samples,
        save_dir: str = "output/datasets",
        split: str = "train",
        image_url_key: str = "url",
        text_key: str = "caption",
        ):
    # stream datasets from hf, and take samples
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    sample_image = []
    sample_text = []
    count = 0
    for data in tqdm(dataset):
        if count >= n_samples:
            break
        url = data[image_url_key]
        try:
            if "jpg" in data.keys():
                image = data["jpg"].convert("RGB")
            else:
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
            sample_text.append(data[text_key])
            count += 1
        except Exception as e:
            print(f"画像読み込みエラー ({url}): {e}, 収集完了数: {count}")
            continue

    # create new hf dataset
    dataset = Dataset.from_dict(
        {
            "url": sample_image,
            "caption": sample_text,
        }
    )
    save_path = os.path.join(save_dir, f"{dataset_name}_sampled_{n_samples}")
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_path)
    print(f"サンプルデータセットを保存しました: {save_path}")
    print(f"収集完了数: {count}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_name",
        type=str,
        default="laion/laion400m",
        help="Dataset name to sample from.",
    )
    args.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to collect.",
    )
    args.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to sample from.",
    )
    args.add_argument(
        "--image_url_key",
        type=str,
        default="url",
        help="Key for image URL in the dataset.",
    )
    args.add_argument(
        "--text_key",
        type=str,
        default="caption",
        help="Key for text in the dataset.",
    )
    args = args.parse_args()
    main(
        dataset_name=args.dataset_name,
        n_samples=args.n_samples,
        split=args.split,
        image_url_key=args.image_url_key,
        text_key=args.text_key,
    )