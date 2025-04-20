NUM_SAMPLES=1000

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-32  \
    model.pretrained=laion2b_e16 \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-32  \
    model.pretrained=laion2b_s34b_b79k \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-L-14  \
    model.pretrained=laion2b_s32b_b82k \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-H-14  \
    model.pretrained=laion2b_s32b_b79k \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-g-14  \
    model.pretrained=laion2b_s12b_b42k \
    model.similarity=COSINE