import os
import heapq
from typing import List, Set, Optional

import click
import tqdm
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
)


def load_ratings(path: str) -> List[Set[int]]:
    with open(path) as f:
        return [{int(v) for v in line.strip().split(',')} for line in f]


def load_descriptions(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f]


def find_neighbors_by_ratings(
        ratings: List[Set[int]], item_id: int, k: int = 100
) -> Set[int]:
    h = []
    for i, item_ratings in enumerate(ratings):
        if i == item_id:
            continue
        s = len(item_ratings & ratings[item_id])
        heapq.heappush(h, (s, i))
        if len(h) > k:
            heapq.heappop(h)
    return {i for _, i in h}


def find_neighbors_by_embeddings(
        embeddings: np.array, item_id: int, k: int = 100
) -> Set[int]:
    distances = np.linalg.norm(embeddings - embeddings[item_id], axis=1)
    nearest_indices = np.argsort(distances)[1:k + 1]
    return {i for i in nearest_indices}


def get_embeddings(descriptions: List[str], model: str, dimensions: Optional[int], cache_dir: str) -> np.array:
    if dimensions is None:
        cache_file = os.path.join(cache_dir, f"{model}.npy")
    else:
        cache_file = os.path.join(cache_dir, f"{model}-{dimensions}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file)
    embeddings = []
    for description in tqdm.tqdm(descriptions):
        response = client.embeddings.create(
            input=description,
            model=model,
            dimensions=dimensions,
        )
        embeddings.append(response.data[0].embedding)
    embeddings = np.array(embeddings)
    np.save(cache_file, embeddings)
    return embeddings


@click.command()
@click.argument("dataset", type=click.Choice(["ml-1m"]))
@click.option("--model", type=str, default="text-embedding-v4", help="OpenAI model to use for embeddings")
@click.option("--dimensions", type=int, default=None, help="Number of dimensions for embeddings")
def main(dataset: str, model: str, dimensions: Optional[int] = None):
    # Load dataset
    dataset_dir = os.path.join("data", dataset)
    ratings = load_ratings(os.path.join(dataset_dir, "ratings.txt"))
    descriptions = load_descriptions(os.path.join(dataset_dir, "descriptions.txt"))

    # Text embeddings
    embeddings = get_embeddings(descriptions, model, dimensions, dataset_dir)

    recall = 0
    for item_id in tqdm.tqdm(range(len(descriptions))):
        by_ratings = find_neighbors_by_ratings(ratings, item_id)
        by_embeddings = find_neighbors_by_embeddings(embeddings, item_id)
        recall += len(by_ratings & by_embeddings) / len(by_ratings)
    recall = recall / len(descriptions)
    print(f"model: {model}, dimensions: {dimensions}, recall: {recall}")


if __name__ == "__main__":
    main()
