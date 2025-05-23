from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from PIL import Image
from torch.nn import Module
from torch.nn import functional as F
from torchtyping import TensorType as TT  # type: ignore
from transformers import CLIPModel, CLIPProcessor # type: ignore


class Predictor(metaclass=ABCMeta):
    def __init__(self, model: Module, device: str) -> None:
        self.model = model
        self.device = device
        self.sample_image_emb: Optional[TT["num_sample", "hidden_dim"]] = None # noqa: F821
        self.sample_text_emb: Optional[TT["num_sample", "hidden_dim"]] = None # noqa: F821
        self.image_prior: Optional[TT["num_sample"]] = None # noqa: F821
        self.text_prior: Optional[TT["num_sample"]] = None # noqa: F821

        self.model.to(device)
        self.model.eval()

    @abstractmethod
    def calc_image_posterior(self, text_emb: TT["hidden_dim"]) -> TT["num_sample"]: # noqa: F821
        pass

    @abstractmethod
    def calc_text_posterior(self, image_emb: TT["hidden_dim"]) -> TT["num_sample"]: # noqa: F821
        pass

    def calc_image_prior(self) -> None:
        assert self.sample_image_emb is not None
        assert self.sample_text_emb is not None
        # calc marginal image posterior as image prior
        # for each text embeddings in self.sample_text_emb, calc calc_image_posterior() and mean them
        self.image_prior = torch.mean(
            torch.stack(
                [
                    self.calc_image_posterior(text_emb)  # type: ignore
                    for text_emb in self.sample_text_emb
                ]
            ),
            dim=0,
        )

    def calc_text_prior(self) -> None:
        assert self.sample_image_emb is not None
        assert self.sample_text_emb is not None
        # calc marginal text posterior as text prior
        self.text_prior = torch.mean(
            torch.stack(
                [
                    self.calc_text_posterior(image_emb)  # type: ignore
                    for image_emb in self.sample_image_emb
                ]
            ),
            dim=0,
        )

    @abstractmethod
    def encode_image(
        self, image_batch: List[Image.Image]
    ) -> TT["num_batch", "hidden_dim"]: # noqa: F821
        pass

    @abstractmethod
    def encode_text(self, text_batch: List[str]) -> TT["num_batch", "hidden_dim"]: # noqa: F821
        pass

    def encode_sample_image(self, sample_image: List[Image.Image]) -> None:
        self.sample_image_emb = self.encode_image(sample_image)

    def encode_sample_text(self, sample_text: List[str]) -> None:
        self.sample_text_emb = self.encode_text(sample_text)


class CLIPPredictor(Predictor):
    def __init__(
        self, processor: CLIPProcessor, model: CLIPModel, device: str = "cuda"
    ) -> None:
        super().__init__(model, device)
        self.processor = processor

    def calc_image_posterior(self, text_emb: TT["hiden_dim"]) -> TT["num_sample"]: # noqa: F821
        sim = torch.stack(
            [
                F.cosine_similarity(text_emb, image_emb, dim=-1)
                for image_emb in self.sample_image_emb # type: ignore
            ]
        )
        return F.softmax(sim, dim=-1) # type: ignore

    def calc_text_posterior(self, image_emb: TT["hiden_dim"]) -> TT["num_sample"]: # noqa: F821
        sim = torch.stack(
            [
                F.cosine_similarity(image_emb, text_emb, dim=-1)
                for text_emb in self.sample_text_emb # type: ignore
            ]
        )
        return F.softmax(sim, dim=-1) # type: ignore

    def encode_image(
        self, image_batch: List[Image.Image]
    ) -> TT["num_batch", "hidden_dim"]: # noqa: F821
        inputs = self.processor(
            images=image_batch, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs) # type: ignore
        return image_emb

    def encode_text(self, text_batch: List[str]) -> TT["num_batch", "hidden_dim"]: # noqa: F821
        inputs = self.processor(
            text=text_batch, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs) # type: ignore
        return text_emb
