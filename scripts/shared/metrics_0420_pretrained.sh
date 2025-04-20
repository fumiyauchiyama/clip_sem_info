NUM_SAMPLES=1000

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/conceptual-captions-12m-webdataset_sampled_$NUM_SAMPLES" \
    save_each=true \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.siglip.normalize_sigmoid=true \
    model.siglip.logit_scale=false \
    model.siglip.logit_bias=false \
    model.model_name=ViT-B-16  \
    model.pretrained=openai \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/conceptual-captions-12m-webdataset_sampled_$NUM_SAMPLES" \
    save_each=true \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.siglip.normalize_sigmoid=true \
    model.siglip.logit_scale=false \
    model.siglip.logit_bias=false \
    model.model_name=ViT-B-32  \
    model.pretrained=openai \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/conceptual-captions-12m-webdataset_sampled_$NUM_SAMPLES" \
    save_each=true \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.siglip.normalize_sigmoid=true \
    model.siglip.logit_scale=false \
    model.siglip.logit_bias=false \
    model.model_name=ViT-L-14  \
    model.pretrained=openai \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/conceptual-captions-12m-webdataset_sampled_$NUM_SAMPLES" \
    save_each=true \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=SigLip \
    model.siglip.normalize_sigmoid=true \
    model.siglip.logit_scale=false \
    model.siglip.logit_bias=false \
    model.model_name=ViT-B-16-SigLIP-512  \
    model.pretrained=webli \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/conceptual-captions-12m-webdataset_sampled_$NUM_SAMPLES" \
    save_each=true \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=SigLip \
    model.siglip.normalize_sigmoid=true \
    model.siglip.logit_scale=false \
    model.siglip.logit_bias=false \
    model.model_name=ViT-L-16-SigLIP-384  \
    model.pretrained=webli \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/conceptual-captions-12m-webdataset_sampled_$NUM_SAMPLES" \
    save_each=true \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=SigLip \
    model.siglip.normalize_sigmoid=true \
    model.siglip.logit_scale=false \
    model.siglip.logit_bias=false \
    model.model_name=ViT-SO400M-14-SigLIP-384  \
    model.pretrained=webli \
    model.similarity=COSINE