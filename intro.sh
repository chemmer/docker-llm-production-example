git clone https://github.com/chemmer/docker-llm-production-example.git
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/ubuntu/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd docker-llm-production-example/
poetry lock
poetry install 
poetry shell
python training.py