export HTTPS_PROXY=http://172.17.0.1:7892
export HTTP_PROXY=http://172.17.0.1:7892
hf download internlm/WildClawBench "workspace/*" --repo-type dataset --local-dir .
