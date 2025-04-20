NUM_SAMPLES=1000

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
    model.model_name=ViT-B-32  \
    model.pretrained=models/abci_siglip_1/2025_03_17-10_06_09-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_32.pt \
    model.similarity=DOT_PRODUCT

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
    model.model_name=ViT-B-32  \
    model.pretrained=models/abci_siglip_1/2025_03_17-10_06_09-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_32.pt \
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
    model.model_name=ViT-B-32  \
    model.pretrained=models/abci_siglip_1/2025_03_17-10_06_26-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_32.pt \
    model.similarity=DOT_PRODUCT

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
    model.model_name=ViT-B-32  \
    model.pretrained=models/abci_siglip_1/2025_03_17-10_06_26-model_ViT-B-32-lr_0.001-b_3000-j_8-p_amp_bf16/checkpoints/epoch_32.pt \
    model.similarity=COSINE
