import ast
import csv
import dbm
import json
import os
import shutil
import zipfile

import click
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
)

DATASET_URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}

KAGGLE_URLS = {"food.com": "shuyangli94/food-com-recipes-and-user-interactions"}


class Dict:

    def __init__(self):
        self.stoi = {}
        self.itos = []

    def __len__(self) -> int:
        assert len(self.stoi) == len(self.itos)
        return len(self.itos)

    def __contains__(self, key: str) -> bool:
        return key in self.stoi

    def id(self, key: str) -> int:
        if key not in self.stoi:
            self.stoi[key] = len(self.itos)
            self.itos.append(key)
        return self.stoi[key]

    def s(self, id: int) -> str:
        if id > len(self.itos):
            return None
        return self.itos[id]


@click.command()
@click.argument("dataset_name", type=click.Choice(["ml-1m"]))
def main(dataset_name: str):
    # Create the dataset directory if it doesn't exist
    dataset_dir = os.path.join("data", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download and extract the dataset
    if dataset_name in KAGGLE_URLS:
        if len(os.listdir(dataset_dir)) > 0:
            print(f"Dataset {dataset_name} already exists at {dataset_dir}.")
        else:
            print(f"Downloading dataset {dataset_name}...")
            path = kagglehub.dataset_download(KAGGLE_URLS[dataset_name])
            download_path = os.path.join(dataset_dir, KAGGLE_URLS[dataset_name])
            shutil.copytree(path, download_path)
            print(f"Dataset {dataset_name} downloaded to {dataset_dir}.")
    elif dataset_name in DATASET_URLS:
        if len(os.listdir(dataset_dir)) > 0:
            print(f"Dataset {dataset_name} already exists at {dataset_dir}.")
        else:
            print(f"Downloading dataset {dataset_name}...")
            dataset_url = DATASET_URLS[dataset_name]
            dataset_zip_path = os.path.join(dataset_dir, f"{dataset_name}.zip")
            # Download the dataset
            response = requests.get(dataset_url)
            if response.status_code == 200:
                with open(dataset_zip_path, "wb") as f:
                    f.write(response.content)
                # Extract the dataset
                with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
                    zip_ref.extractall(dataset_dir)
                # Remove the zip file
                os.remove(dataset_zip_path)
                print(
                    f"Dataset {dataset_name} downloaded and extracted to {dataset_dir}."
                )
            else:
                print(
                    f"Failed to download dataset {dataset_name}. Status code: {response.status_code}"
                )
    else:
        print(f"Dataset {dataset_name} not found.")

    # Parse the dataset
    if dataset_name == "ml-1m":
        parse_ml_1m()
    elif dataset_name == "food.com":
        parse_food_com()


def parse_ml_100k():
    dataset_dir = os.path.join("data", "ml-100k")
    ratings_file = os.path.join(dataset_dir, "ml-100k/u.data")
    items_file = os.path.join(dataset_dir, "ml-100k/u.item")
    users_file = os.path.join(dataset_dir, "ml-100k/u.user")
    manifest_file = os.path.join(dataset_dir, "manifest.json")

    user_dict = Dict()
    item_dict = Dict()
    manifest = {}

    with open(os.path.join(dataset_dir, "ratings.csv"), "w") as o:
        with open(ratings_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                user_id, item_id, rating, _ = line.strip().split("\t")
                user_index = user_dict.id(user_id)
                item_index = item_dict.id(item_id)
                o.write(f"{user_index},{item_index},{rating}\n")

    items = [[]] * len(item_dict)
    embeddings = [[]] * len(item_dict)
    cache = dbm.open(os.path.join(dataset_dir, "cache"), "c")
    with open(items_file, "r", encoding="unicode_escape") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().split("|")
            item_id = item_dict.id(fields[0])
            assert len(fields[5:]) == 19
            # Extract the labels from the fields
            labels = [i for i, v in enumerate(fields[5:]) if v == "1"]
            items[item_id] = labels
            # Get embedding for the description
            if fields[1] in cache:
                data = json.loads(cache[fields[1]])
                description = data["description"]
                embeddings[item_id] = data["embedding"]
            else:
                description = get_description(fields[1])
                embedding = get_embedding(description)
                embeddings[item_id] = embedding
                cache[fields[1]] = json.dumps(
                    {
                        "description": description,
                        "embedding": embedding,
                    }
                )
    with open(os.path.join(dataset_dir, "items.csv"), "w") as o:
        for item in items:
            o.write(",".join([str(i) for i in item]) + "\n")
    embeddings = np.array(embeddings)
    np.save(os.path.join(dataset_dir, "embeddings.npy"), embeddings)

    users = [[]] * len(user_dict)
    with open(users_file, "r") as f:
        for line in f.readlines():
            fields = line.strip().split("|")
            user_id = user_dict.id(fields[0])
            gender = 1 if fields[2] == "M" else 0
            users[user_id] = [gender]
    with open(os.path.join(dataset_dir, "users.csv"), "w") as o:
        for user in users:
            o.write(",".join([str(i) for i in user]) + "\n")

    manifest["num_users"] = len(user_dict)
    manifest["num_items"] = len(item_dict)
    manifest["num_item_features"] = 19
    manifest["num_user_features"] = 2
    json.dump(manifest, open(manifest_file, "w"), indent=4)


def parse_ml_1m():
    dataset_dir = os.path.join("data", "ml-1m")
    user_dict = Dict()
    item_dict = Dict()

    ratings = []
    with open(os.path.join(dataset_dir, "ml-1m/ratings.dat"), "r") as f:
        lines = f.readlines()
        for line in lines:
            user_id, item_id, rating, _ = line.strip().split("::")
            user_index = user_dict.id(user_id)
            item_index = item_dict.id(item_id)
            if item_index >= len(ratings):
                ratings.append([])
            ratings[item_index].append(user_index)
    with open(os.path.join(dataset_dir, "ratings.txt"), "w") as f:
        for users in ratings:
            f.write(",".join([str(u) for u in users]) + "\n")

    descriptions = [""] * len(ratings)
    cache = dbm.open(os.path.join(dataset_dir, "cache"), "c")
    with open(os.path.join(dataset_dir, "ml-1m/movies.dat"), "r", encoding="unicode_escape") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().split("::")
            if fields[0] not in item_dict:
                continue
            item_id = item_dict.id(fields[0])
            if fields[1] not in cache:
                cache[fields[1]] = get_description(fields[1])
            descriptions[item_id] = cache[fields[1]]
    with open(os.path.join(dataset_dir, "descriptions.txt"), "w") as f:
        for description in descriptions:
            f.write(" ".join(description.decode('utf-8').split()) + "\n")


def get_description(title: str) -> str:
    prompt = f"Write a short description of the movie '{title}' in one paragraph."
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def get_embedding(text: str) -> list:
    embedding = client.embeddings.create(
        model="text-embedding-v3",
        input=text,
        dimensions=64,
    )
    return embedding.data[0].embedding


def parse_food_com():
    download_path = os.path.join(
        "data", "food.com/shuyangli94/food-com-recipes-and-user-interactions"
    )
    interactions_test_csv = os.path.join(download_path, "interactions_test.csv")
    interactions_train_csv = os.path.join(download_path, "interactions_train.csv")
    interactions_validation_csv = os.path.join(
        download_path, "interactions_validation.csv"
    )
    raw_recipes_csv = os.path.join(download_path, "RAW_recipes.csv")
    manifest_file = os.path.join("data", "food.com", "manifest.json")

    user_dict = Dict()
    item_dict = Dict()
    labels_dict = Dict()
    manifest = {}

    with open(os.path.join("data", "food.com", "ratings.csv"), "w") as o:
        for file in [
            interactions_test_csv,
            interactions_train_csv,
            interactions_validation_csv,
        ]:
            with open(file, "r") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    user_id, item_id, _, rating, _, _ = line.strip().split(",")
                    user_index = user_dict.id(user_id)
                    item_index = item_dict.id(item_id)
                    o.write(f"{user_index},{item_index},{rating}\n")

    items = [[]] * len(item_dict)
    embeddings = [[]] * len(item_dict)
    cache = dbm.open(os.path.join("data", "food.com", "cache"), "c")
    with open(raw_recipes_csv, "r", encoding="unicode_escape") as f:
        spamreader = csv.reader(f, delimiter=",", quotechar='"')
        next(spamreader)
        for row in spamreader:
            if row[1] not in item_dict:
                continue
            item_id = item_dict.id(row[1])
            labels = ast.literal_eval(row[5])
            items[item_id] = [labels_dict.id(v) for v in labels]
            if row[0] in cache:
                data = json.loads(cache[row[0]])
                embeddings[item_id] = data["embedding"]
            else:
                description = row[0] + "\n\n"
                try:
                    for line in ast.literal_eval(row[8]):
                        description += "- " + line + ".\n"
                except SyntaxError:
                    description += row[8] + "\n"
                embedding = get_embedding(description)
                embeddings[item_id] = embedding
                cache[row[0]] = json.dumps(
                    {
                        "embedding": embedding,
                    }
                )
    with open(os.path.join("data", "food.com", "items.csv"), "w") as o:
        for item in items:
            o.write(",".join([str(i) for i in item]) + "\n")
    embeddings = np.array(embeddings)
    np.save(os.path.join("data", "food.com", "embeddings.npy"), embeddings)

    manifest["num_users"] = len(user_dict)
    manifest["num_items"] = len(item_dict)
    manifest["num_item_features"] = len(labels_dict)
    manifest["num_user_features"] = 0
    json.dump(manifest, open(manifest_file, "w"), indent=4)


if __name__ == "__main__":
    main()
