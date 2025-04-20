NUM_SAMPLES=1000

python3 -m apps.main.metrics \
    config=output/20250416/20250416-182218_emd_text.yaml \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/conceptual-captions-12m-webdataset_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false

python3 -m apps.main.metrics \
    config=output/20250416/20250416-182218_emd_text.yaml \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/laion400m_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false

python3 -m apps.main.metrics \
    config=output/20250416/20250416-182218_emd_text.yaml \
    dataset.n_samples=$NUM_SAMPLES \
    dataset.dataset_name="output/datasets/laion/relaion2B-en-research_sampled_$NUM_SAMPLES" \
    metrics.calc_sinkhorn=false \
    metrics.calc_emd=false