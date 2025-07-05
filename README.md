# Text Embedding Benchmark for Recommender Systems

Benchmark for evaluating text embeddings in recommender systems.

## Download the dataset

OpenAI API key is required to generate descriptions for the movies in the dataset.

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export OPENAI_API_BASE=https://api.openai.com/v1
python download.py ml-1m
```

## Run the benchmark

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export OPENAI_API_BASE=https://api.openai.com/v1
python benchmark.py ml-1m text-embedding-3-large --dimensions 512
```

## Results

|                        | Embedding size | ml-1m   |
|------------------------|----------------|---------|
| bge-m3                 | 1024           | 0.11492 |
| mxbai-embed-large      | 1024           | 0.16595 |
| nomic-embed-text       | 768            | 0.15161 |
| text-embedding-v4      | 2048           | 0.17262 |
| text-embedding-v4      | 1024           | 0.16946 |
| text-embedding-v4      | 768            | 0.16674 |
| text-embedding-v4      | 512            | 0.16278 |
| text-embedding-v4      | 256            | 0.15280 |
| text-embedding-v4      | 128            | 0.13873 |
| text-embedding-v4      | 64             | 0.11969 |
| text-embedding-3-large | 3072           | 0.17395 |
| text-embedding-3-large | 2048           | 0.17241 |
| text-embedding-3-large | 1024           | 0.17026 |
| text-embedding-3-large | 768            | 0.16610 |
| text-embedding-3-large | 512            | 0.16362 |
| text-embedding-3-large | 256            | 0.15086 |
| text-embedding-3-large | 128            | 0.13769 |
| text-embedding-3-large | 64             | 0.11272 |
