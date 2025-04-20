NUM_SAMPLES=1000

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-16  \
    model.pretrained=laion400m_e32 \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-L-14  \
    model.pretrained=laion400m_e32 \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-32-quickgelu  \
    model.pretrained=laion400m_e32 \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-16-plus-240  \
    model.pretrained=laion400m_e32 \
    model.similarity=COSINE



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


python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=SigLip \
    model.siglip.normalize_sigmoid=true \
    model.model_name=ViT-SO400M-14-SigLIP-384  \
    model.pretrained=webli \
    model.similarity=COSINE


python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-32  \
    model.pretrained=openai \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-B-16  \
    model.pretrained=openai \
    model.similarity=COSINE

python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-L-14  \
    model.pretrained=openai \
    model.similarity=COSINE


python3 -m apps.main.metrics \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false \
    model.model_type=CLIP \
    model.model_name=ViT-L-14-336  \
    model.pretrained=openai \
    model.similarity=COSINE

