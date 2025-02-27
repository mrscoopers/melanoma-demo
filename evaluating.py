from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client import models
from datasets import load_dataset
import torch
import configparser
from collections import Counter
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)

configparser = configparser.ConfigParser()
configparser.read("config.ini")
qdrant_cloud_url = configparser["qdrant"]["cloud_url"]
qdrant_api = configparser["secrets"]["api_key"]

client = QdrantClient(url=qdrant_cloud_url, api_key=qdrant_api)
collection_name = "melanoma"

test_dataset = load_dataset("marmal88/skin_cancer", split="test")


def get_embeddings_query(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
        outputs = outputs.cpu().numpy()  # 1st in a batch

    patches = outputs[0, 1:, :]  # 1 el in a batch
    cls_output = outputs[0, 0, :]  # 1 el in a batch
    pooled_patches = patches.mean(axis=0)

    del outputs, inputs  # free GPU memory
    torch.cuda.empty_cache()

    return cls_output, pooled_patches, patches


def search_knn(image_embedding, k=15, using_vector="pooled_patches", collection_name="melanoma"):
    for attempt in range(3):
        try:
            result = client.query_points(
                collection_name=collection_name,
                query=image_embedding,
                using=using_vector,
                limit=k,
                with_payload=True
            ).points
            return result
        except Exception as e:
            print(f"[search_knn] Attempt {attempt+1} failed with error: {e}")
            if attempt == 2:
                print("[search_knn] Returning empty result after 3 failed attempts.")
                return []
    return []


def search_rerank_knn(image_embedding, image_embedding_patches, 
                      prefetch_k=30, k=15, using_vector="pooled_patches", collection_name="melanoma"):
    for attempt in range(3):
        try:
            result = client.query_points(
                collection_name=collection_name,
                prefetch=models.Prefetch(
                    query=image_embedding,
                    using=using_vector,
                    limit=prefetch_k
                ),
                query=image_embedding_patches,
                using="patches",
                limit=k,
                with_payload=True
            ).points
            return result
        except Exception as e:
            print(f"[search_rerank_knn] Attempt {attempt+1} failed with error: {e}")
            if attempt == 2:
                print("[search_rerank_knn] Returning empty result after 3 failed attempts.")
                return []
    return []


def KNN_classifier_dx(image_embedding, k=10, using="pooled_patches", collection_name="melanoma"):
    points = search_knn(image_embedding, k, using, collection_name)

    if not points:
        return None, 0

    majority_vote_two_most_common = Counter([point.payload["dx"] for point in points]).most_common(2)

    # Resolve ties by increasing k
    if (len(majority_vote_two_most_common) > 1) and (k <= 50) and \
       (majority_vote_two_most_common[0][1] == majority_vote_two_most_common[1][1]):
        print("resolving a tie in KNN_classifier_dx")
        return KNN_classifier_dx(image_embedding, k + 5, using, collection_name)

    return majority_vote_two_most_common[0][0], majority_vote_two_most_common[0][1] / k


def KNN_classifier_dx_with_rerank(image_embedding, image_patches_embeddings, k=10, 
                                  using="pooled_patches", collection_name="melanoma"):
    points = search_rerank_knn(image_embedding, image_patches_embeddings, k * 2, k, using, collection_name)

    if not points:
        return None, 0

    majority_vote_two_most_common = Counter([point.payload["dx"] for point in points]).most_common(2)

    # Resolve ties by increasing k
    if (len(majority_vote_two_most_common) > 1) and (k <= 50) and \
       (majority_vote_two_most_common[0][1] == majority_vote_two_most_common[1][1]):
        print("resolving a tie in KNN_classifier_dx_with_rerank")
        return KNN_classifier_dx_with_rerank(image_embedding, image_patches_embeddings, k + 5, using, collection_name)
        
    return majority_vote_two_most_common[0][0], majority_vote_two_most_common[0][1] / k


cls_count = 0
pooled_patches_count = 0
cls_rerank_count = 0
pooled_patches_rerank_count = 0

total = len(test_dataset)
k = 10

for elem in tqdm(test_dataset, desc="Evaluating"):
    cls_output, pooled_patches, patches = get_embeddings_query(elem["image"])

    pred_cls, _ = KNN_classifier_dx(cls_output, k, using="cls")
    if pred_cls == elem['dx']:
        cls_count += 1

    pred_pooled, _ = KNN_classifier_dx(pooled_patches, k, using="pooled_patches")
    if pred_pooled == elem['dx']:
        pooled_patches_count += 1

    pred_pooled_rr, _ = KNN_classifier_dx_with_rerank(pooled_patches, patches, k, using="pooled_patches")
    if pred_pooled_rr == elem['dx']:
        pooled_patches_rerank_count += 1

    pred_cls_rr, _ = KNN_classifier_dx_with_rerank(cls_output, patches, k, using="cls")
    if pred_cls_rr == elem['dx']:
        cls_rerank_count += 1

print(f"Precision of CLS: {cls_count / total}")
print(f"Precision of Pooled Patches: {pooled_patches_count / total}")
print(f"Precision of CLS with rerank: {cls_rerank_count / total}")
print(f"Precision of Pooled Patches with rerank: {pooled_patches_rerank_count / total}")