from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="internlm/WildClawBench",
    repo_type="dataset",
    local_dir="./workspace",
    allow_patterns="workspace/*",
    resume_download=True
)