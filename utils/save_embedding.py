import os, sys
import json

sys.path.append("./")

from configs.main_config import EMBEDDING_NAME, CITY_PAIRS_MAPPING_FILE

def save_embeddings_file(model, filename=EMBEDDING_NAME):
    #Read City Pairs Json
    with open(CITY_PAIRS_MAPPING_FILE, 'r') as file:
        city_pairs_map = json.load(file)

    city_pairs_map_reverse = {v: k for k, v in city_pairs_map.items()}

    embeddings = model.embedding.weight.detach().cpu().numpy()  # Get embedding weights
    embeddings_dict = {city_pairs_map_reverse[i]: emb.tolist() for i, emb in enumerate(embeddings)}  # Convert to dict

    # Save as JSON
    with open(f"{filename}", "w") as f:
        json.dump(embeddings_dict, f, indent=4)



