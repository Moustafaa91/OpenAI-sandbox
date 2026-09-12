import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from openai_config import EMBEDDING_MODEL, get_openai_client


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector for each input string."""
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def main() -> None:
    embedding = get_embeddings(["life"])[0]
    print("Word embedding")
    print(f"Length = {len(embedding)}")
    print(embedding[:10])

    embedding = get_embeddings(["What is the meaning of life?"])[0]
    print("\nSentence embedding")
    print(f"Length = {len(embedding)}")
    print(embedding[:10])

    print("\nSimilarity")
    emb_1 = get_embeddings(["What is the meaning of life?"])[0]
    emb_2 = get_embeddings(["How does one spend their time well on Earth?"])[0]
    emb_3 = get_embeddings(["Would you like a salad?"])[0]

    vec_1 = [emb_1]
    vec_2 = [emb_2]
    vec_3 = [emb_3]

    print(cosine_similarity(vec_1, vec_2))
    print(cosine_similarity(vec_2, vec_3))
    print(cosine_similarity(vec_1, vec_3))

    print("\nFrom word to sentence embeddings")
    in_1 = "The kids play in the park."
    in_2 = "The play was for kids in the park."

    in_pp_1 = ["kids", "play", "park"]
    in_pp_2 = ["play", "kids", "park"]

    embeddings_1 = get_embeddings(in_pp_1)
    emb_array_1 = np.stack(embeddings_1)
    print(f"First word matrix shape: {emb_array_1.shape}")

    embeddings_2 = get_embeddings(in_pp_2)
    emb_array_2 = np.stack(embeddings_2)
    print(f"Second word matrix shape: {emb_array_2.shape}")

    emb_1_mean = emb_array_1.mean(axis=0)
    emb_2_mean = emb_array_2.mean(axis=0)

    print(f"Mean embedding shape: {emb_1_mean.shape}")
    print(f"Mean embeddings are equal: {np.allclose(emb_1_mean, emb_2_mean)}")
    print(emb_1_mean[:4])
    print(emb_2_mean[:4])

    print("\nSentence embeddings from the model")
    embedding_1 = get_embeddings([in_1])[0]
    embedding_2 = get_embeddings([in_2])[0]

    print(in_1)
    print(in_2)
    print(embedding_1[:4])
    print(embedding_2[:4])
    similarity = cosine_similarity([embedding_1], [embedding_2])[0, 0]
    print(f"Cosine similarity = {similarity:.4f}")


if __name__ == "__main__":
    main()
