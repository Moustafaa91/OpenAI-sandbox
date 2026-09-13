import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from openai_config import EMBEDDING_MODEL, get_openai_client
from utils import encode_text_to_embedding_batched, encode_texts_to_embeddings, generate_batches, get_embeddings, plot_2D, plot_heatmap, clusters_2D


def word_embeddings():
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


def visualize_embeddings():
    in_1 = "Missing flamingo discovered at swimming pool"
    in_2 = "Sea otter spotted on surfboard by beach"
    in_3 = "Baby panda enjoys boat ride"
    in_4 = "Breakfast themed food truck beloved by all!"
    in_5 = "New curry restaurant aims to please!"
    in_6 = "Python developers are wonderful people"
    in_7 = "TypeScript, C++ or Java? All are great!" 
    input_text_lst_news = [in_1, in_2, in_3, in_4, in_5, in_6, in_7]
    y_labels = input_text_lst_news
    
    embeddings = []
    for input_text in input_text_lst_news:
        emb = get_embeddings([input_text])[0]
        embeddings.append(emb)
    
    embeddings_array = np.array(embeddings) 
    
    print("Shape: " + str(embeddings_array.shape))
    print(embeddings_array)
    
    # Perform PCA for 2D visualization
    PCA_model = PCA(n_components = 2)
    PCA_model.fit(embeddings_array)
    new_values = PCA_model.transform(embeddings_array)
    print("Shape: " + str(new_values.shape))
    print(new_values)
    plot_2D(new_values[:,0], new_values[:,1], input_text_lst_news)
    

    # Plot the heatmap
    plot_heatmap(embeddings_array, y_labels = y_labels, title = "Embeddings Heatmap")
    
def example_usage_stackoverflow():
    """Visualize embeddings for a set of StackOverflow questions."""
    stackoverflow_df = pd.read_csv('so_database_app.csv')
    
    so_questions = stackoverflow_df.input_text.tolist() 
    #batches = generate_batches(sentences = so_questions)
    #batch = next(batches)
    #batch_embeddings = encode_texts_to_embeddings(batch)
    #print(f"{len(batch_embeddings)} embeddings of size {len(batch_embeddings[0])}")
    
    # To save time, we can load the embeddings from a file if they exist,
    # otherwise compute and save them for later use.
    embeddings_path = Path('question_embeddings_app.pkl')
    if embeddings_path.exists() and embeddings_path.stat().st_size > 0:
        with embeddings_path.open('rb') as embeddings_file:
            question_embeddings = pickle.load(embeddings_file)
        print(f"Loaded embeddings from {embeddings_path}")
    else:
        question_embeddings = encode_text_to_embedding_batched(
                                sentences=so_questions,
                                api_calls_per_second = 20/60, 
                                batch_size = 5)
        with embeddings_path.open('wb') as embeddings_file:
            pickle.dump(question_embeddings, embeddings_file)
        print(f"Saved embeddings to {embeddings_path}")
    
    
    
    clustering_dataset = question_embeddings
    n_clusters = 4 # This dataset contains questions related to 4 programming languages: ["python", "html", "r", "css"]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, 
                    n_init = 'auto').fit(clustering_dataset)
    kmeans_labels = kmeans.labels_
    PCA_model = PCA(n_components=2)
    PCA_model.fit(clustering_dataset)
    new_values = PCA_model.transform(clustering_dataset)
    #Clustering is able to identify distinct clusters of categories related questions, 
    # without being given the category labels ["python", "html", "r", "css"].
    clusters_2D(x_values = new_values[:,0], y_values = new_values[:,1], 
            labels = stackoverflow_df, kmeans_labels = kmeans_labels, displayed_column = "category")
    
    

if __name__ == "__main__":
    example_usage_stackoverflow()
