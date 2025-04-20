from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from math import log
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from xml.parsers.expat import model

import numpy as np
import torch
from open_clip import CLIP, CustomTextCLIP, create_model_from_pretrained, get_tokenizer
from open_clip.tokenizer import HFTokenizer, SigLipTokenizer, SimpleTokenizer
from PIL import Image
from sympy import Si
from torch.nn import Module
from torch.nn import functional as F
from torchtyping import TensorType as TT  # type: ignore
from torchvision.transforms import Compose


class Similarity(Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"


@dataclass
class SigLipConfig:
    normalize_sigmoid: bool = False
    logit_scale: bool = False
    logit_bias: bool = False


@dataclass
class ModelConfig:
    model_name: str = "ViT-B-16"
    pretrained: str = "openai"
    model_type: str = "CLIP"  # Literal["CLIP", "SigLip"]
    similarity: Similarity = Similarity.COSINE
    siglip: SigLipConfig = field(default_factory=SigLipConfig)
    device: str = "cuda"


class Predictor(metaclass=ABCMeta):
    def __init__(
        self,
        model: CLIP,
        image_processor: Compose,
        tokenizer: Union[HFTokenizer, SigLipTokenizer, SimpleTokenizer],
        similarity: Similarity = Similarity.COSINE,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.similarity = similarity
        self.device = device
        self.sample_image_emb: Optional[TT["num_sample", "hidden_dim"]] = None  # noqa: F821
        self.sample_text_emb: Optional[TT["num_sample", "hidden_dim"]] = None  # noqa: F821
        self.image_prior: Optional[TT["num_sample"]] = None  # noqa: F821
        self.text_prior: Optional[TT["num_sample"]] = None  # noqa: F821

        self.model.to(device)
        self.model.eval()

    @abstractmethod
    def calc_image_posterior(self, text_emb: TT["hidden_dim"]) -> TT["num_sample"]:  # noqa: F821
        pass

    @abstractmethod
    def calc_text_posterior(self, image_emb: TT["hidden_dim"]) -> TT["num_sample"]:  # noqa: F821
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

    def encode_image(
        self, image_batch: List[Image.Image]
    ) -> TT["num_batch", "hidden_dim"]:  # noqa: F821
        inputs = [self.image_processor(image).to(self.device) for image in image_batch]  # type: ignore
        inputs = torch.stack(inputs)
        with torch.no_grad():
            image_emb = self.model.encode_image(inputs, normalize=False)
        return image_emb  # type: ignore

    def encode_text(self, text_batch: List[str]) -> TT["num_batch", "hidden_dim"]:  # noqa: F821
        inputs = self.tokenizer(
            text_batch, context_length=self.model.context_length
        ).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(inputs, normalize=False)
        return text_emb  # type: ignore

    def encode_sample_image(self, sample_image: List[Image.Image]) -> None:
        self.sample_image_emb = self.encode_image(sample_image)

    def encode_sample_text(self, sample_text: List[str]) -> None:
        self.sample_text_emb = self.encode_text(sample_text)


class CLIPPredictor(Predictor):
    def calc_image_posterior(self, text_emb: TT["hiden_dim"]) -> TT["num_sample"]:  # noqa: F821
        if self.similarity == Similarity.COSINE:
            sim = torch.stack(
                [
                    F.cosine_similarity(text_emb, image_emb, dim=-1)
                    for image_emb in self.sample_image_emb  # type: ignore
                ]
            )
        elif self.similarity == Similarity.DOT_PRODUCT:
            sim = torch.stack(
                [
                    torch.dot(text_emb, image_emb)
                    for image_emb in self.sample_image_emb  # type: ignore
                ]
            )
        return F.softmax(sim, dim=-1)  # type: ignore

    def calc_text_posterior(self, image_emb: TT["hiden_dim"]) -> TT["num_sample"]:  # noqa: F821
        if self.similarity == Similarity.COSINE:
            sim = torch.stack(
                [
                    F.cosine_similarity(image_emb, text_emb, dim=-1)
                    for text_emb in self.sample_text_emb  # type: ignore
                ]
            )
        elif self.similarity == Similarity.DOT_PRODUCT:
            sim = torch.stack(
                [
                    torch.dot(image_emb, text_emb)
                    for text_emb in self.sample_text_emb  # type: ignore
                ]
            )
        return F.softmax(sim, dim=-1)  # type: ignore


class SigLIPPredictor(Predictor):
    def __init__(
        self,
        model: CLIP,
        image_processor: Compose,
        tokenizer: SigLipTokenizer,
        siglip_cfg: SigLipConfig,
        similarity: Similarity = Similarity.COSINE,
        device: str = "cuda",
    ) -> None:
        super().__init__(model, image_processor, tokenizer, similarity, device)
        self.siglip_cfg = siglip_cfg

    def calc_image_posterior(self, text_emb: TT["hiden_dim"]) -> TT["num_sample"]:  # noqa: F821
        if self.similarity == Similarity.COSINE:
            sim = torch.stack(
                [
                    F.cosine_similarity(text_emb, image_emb, dim=-1)
                    for image_emb in self.sample_image_emb  # type: ignore
                ]
            )
        elif self.similarity == Similarity.DOT_PRODUCT:
            sim = torch.stack(
                [
                    torch.dot(text_emb, image_emb)
                    for image_emb in self.sample_image_emb  # type: ignore
                ]
            )
        if self.siglip_cfg.logit_scale:
            assert self.model.logit_scale is not None
            sim = sim * self.model.logit_scale.exp()
        if self.siglip_cfg.logit_bias:
            assert self.model.logit_bias is not None
            sim = sim + self.model.logit_bias
        logit = torch.sigmoid(sim)  # type: ignore
        return (
            logit / torch.sum(logit, dim=-1)
            if self.siglip_cfg.normalize_sigmoid
            else logit
        )  # type: ignore

    def calc_text_posterior(self, image_emb: TT["hiden_dim"]) -> TT["num_sample"]:  # noqa: F821
        if self.similarity == Similarity.COSINE:
            sim = torch.stack(
                [
                    F.cosine_similarity(image_emb, text_emb, dim=-1)
                    for text_emb in self.sample_text_emb  # type: ignore
                ]
            )
        elif self.similarity == Similarity.DOT_PRODUCT:
            sim = torch.stack(
                [
                    torch.dot(image_emb, text_emb)
                    for text_emb in self.sample_text_emb  # type: ignore
                ]
            )
        if self.siglip_cfg.logit_scale:
            assert self.model.logit_scale is not None
            sim = sim * self.model.logit_scale.exp()
        if self.siglip_cfg.logit_bias:
            assert self.model.logit_bias is not None
            sim = sim + self.model.logit_bias
        logit = torch.sigmoid(sim)  # type: ignore
        return (
            logit / torch.sum(logit, dim=-1)
            if self.siglip_cfg.normalize_sigmoid
            else logit
        )  # type: ignore


def get_predictor(
    model_cfg: ModelConfig,
) -> Predictor:
    device = model_cfg.device
    model_kwargs: Dict[str, Union[float, str]] = {}
    if model_cfg.model_type == "SigLip":
        # load model
        model_kwargs["init_logit_scale"] = np.log(10)  # different from CLIP
        model_kwargs["init_logit_bias"] = -10
    vl_model, image_processor = create_model_from_pretrained(
        model_cfg.model_name,
        pretrained=model_cfg.pretrained,
        device=device,
        **model_kwargs,
    )  # type: ignore
    tokenizer = get_tokenizer(model_cfg.model_name)

    assert isinstance(vl_model, Union[CLIP, CustomTextCLIP]), (
        f"model should be CLIP, but got {type(vl_model)}"
    )
    assert isinstance(image_processor, Compose), (
        f"image_processor should be Compose, but got {type(image_processor)}"
    )
    assert isinstance(tokenizer, (HFTokenizer, SigLipTokenizer, SimpleTokenizer)), (
        f"tokenizer should be HFTokenizer or SigLipTokenizer or SimpleTokenizer, but got {type(tokenizer)}"
    )

    if model_cfg.model_type == "CLIP":
        predictor = CLIPPredictor(
            vl_model,
            image_processor,
            tokenizer,
            similarity=model_cfg.similarity,
            device=device,
        )
    elif model_cfg.model_type == "SigLip":
        predictor = SigLIPPredictor(
            vl_model,
            image_processor,
            tokenizer,
            model_cfg.siglip,
            similarity=model_cfg.similarity,
            device=device,
        )
    else:
        raise ValueError(
            f"model_type should be 'SigLip' or 'CLIP', but got {model_cfg.model_type}"
        )

    return predictor
