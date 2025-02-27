from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client import models
from datasets import load_dataset
import torch
import configparser
import uuid
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)


configparser = configparser.ConfigParser()
configparser.read("config.ini")
qdrant_cloud_url = configparser["qdrant"]["cloud_url"]
qdrant_api = configparser["secrets"]["api_key"]


client = QdrantClient(url=qdrant_cloud_url, api_key=qdrant_api)
collection_name = "melanoma_main_collection"

# Create collection if not exists
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "patches": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0),  # switching off HNSW
            ),
            "cls": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            ),
            "pooled_patches": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            ),
        },
    )

# Function to get embeddings for a single batch
def get_embeddings_batch(batch):
    # Move inputs to GPU (if available)
    inputs = processor(images=batch["image"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
        outputs = outputs.cpu().numpy()

    patches = outputs[:, 1:, :]
    cls_outputs = outputs[:, 0, :]
    pooled_patches = patches.mean(axis=1)

    del outputs, inputs  # free GPU memory
    torch.cuda.empty_cache()

    return cls_outputs, pooled_patches, patches

# Embeddings batch upload
def upload_batch(batch, payloads_batch, collection_name="melanoma_collection"):
    cls, pooled_patches, patches = get_embeddings_batch(batch)

    client.upload_collection(
        collection_name=collection_name,
        vectors={
            "cls": cls,
            "pooled_patches": pooled_patches,
            "patches": patches,
        },
        payload=payloads_batch,
        ids=[str(uuid.uuid4()) for _ in range(len(batch))],
        batch_size=2, #too fat to do more
    )

    # Explicit cleanup of CPU memory
    del cls, pooled_patches, patches
    torch.cuda.empty_cache()


batch_size = 8
dataset = load_dataset("marmal88/skin_cancer", split="train")

def add_url(item):
    # Build the URL string for each sample
    url = f"https://storage.googleapis.com/demo-skin-cancer/train_{item['image_id']}_{item['lesion_id']}.jpeg"
    return {"url": url}

# Prepare payloads
dataset_with_url = dataset.map(add_url)

payloads = (
    dataset_with_url.select_columns(["image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization", "url"])
    .to_pandas()
    .fillna({"age": 0})
    .to_dict(orient="records")
)

# Loop over the dataset in batches
with tqdm(total=len(dataset), desc="Uploading progress") as pbar:
    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i : i + batch_size]
        payloads_batch = payloads[i : i + batch_size]

        upload_batch(batch_data, payloads_batch, collection_name)

        # Explicitly clear local references and empty the cache
        del batch_data, payloads_batch
        torch.cuda.empty_cache()

        pbar.update(batch_size)

print("Uploading complete!")
