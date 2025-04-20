NUM_SAMPLES=1000

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/laion400m_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-16  \
    model.pretrained=laion400m_e32 \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/laion400m_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-L-14  \
    model.pretrained=laion400m_e32 \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/laion400m_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-32-quickgelu  \
    model.pretrained=laion400m_e32 \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/laion400m_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-16-plus-240  \
    model.pretrained=laion400m_e32 \
    model.similarity=COSINE