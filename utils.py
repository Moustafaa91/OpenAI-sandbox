from typing import Type
import mplcursors
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.auto import tqdm
import math
from concurrent.futures import ThreadPoolExecutor
import functools

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from openai_config import DEFAULT_MODEL, EMBEDDING_MODEL, get_openai_client


def call_llm(
    prompt: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    client = client or get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def validate_with_model(
    data_model: Type[BaseModel], llm_response: str
):
    try:
        validated_data = data_model.model_validate_json(llm_response)
        print("data validation successful!")
        print(validated_data.model_dump_json(indent=2))
        return validated_data, None
    except ValidationError as error:
        print(f"error validating data: {error}")
        return None, f"This response generated a validation error: {error}."


def create_retry_prompt(
    original_prompt: str,
    original_response: str,
    error_message: str,
) -> str:
    return f"""
This is a request to fix an error in the structure of an llm_response.
Here is the original request:
<original_prompt>
{original_prompt}
</original_prompt>

Here is the original llm_response:
<llm_response>
{original_response}
</llm_response>

This response generated an error:
<error_message>
{error_message}
</error_message>

Compare the error message and the llm_response and identify what
needs to be fixed or removed in the llm_response to resolve this error.

Respond ONLY with valid JSON. Do not include any explanations or
other text or formatting before or after the JSON string.
"""


def validate_llm_response(
    prompt: str,
    data_model: Type[BaseModel],
    n_retry: int = 5,
):
    response_content = call_llm(prompt)
    current_prompt = prompt

    for attempt in range(n_retry + 1):
        validated_data, validation_error = validate_with_model(
            data_model, response_content
        )
        if validation_error is None:
            return validated_data, None

        if attempt == n_retry:
            print(f"Max retries reached. Last error: {validation_error}")
            return None, f"Max retries reached. Last error: {validation_error}"

        print(f"retry {attempt} of {n_retry} failed, trying again...")
        current_prompt = create_retry_prompt(
            original_prompt=current_prompt,
            original_response=response_content,
            error_message=validation_error,
        )
        response_content = call_llm(current_prompt)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector for each input string."""
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]

def plot_heatmap(data, x_labels=None, y_labels=None, title=None):
    fig, ax = plt.subplots(figsize=(50, 3))
    heatmap = ax.pcolor(data, cmap='coolwarm', edgecolors='k', linewidths=0.1)

    # Add color bar to the right of the heatmap
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.remove()

    # Set labels for each axis
    if x_labels:
        ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    if y_labels:
        ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_yticklabels(y_labels, va="center")

    # Set title
    if title:
        ax.set_title(title)
        
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_2D(x_values, y_values, labels):

    # Create scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_values, 
                         y_values, 
                         alpha = 0.5, 
                         edgecolors='k',
                         s = 40) 

    # Create a mplcursors object to manage the data point interaction
    cursor = mplcursors.cursor(scatter, hover=True)

    #aes
    ax.set_title('Embedding visualization in 2D')  # Add a title
    ax.set_xlabel('X_1')  # Add x-axis label
    ax.set_ylabel('X_2')  # Add y-axis label

    # Define how each annotation should look
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(labels[int(sel.index)])
        sel.annotation.get_bbox_patch().set(facecolor='white', alpha=0.5) # Set annotation's background color
        sel.annotation.set_fontsize(12) 

    plt.show()
    
def clusters_2D(x_values, y_values, labels, kmeans_labels, displayed_column = None):
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_values, 
                         y_values, 
                         c = kmeans_labels, 
                         cmap='Set1', 
                         alpha=0.5, 
                         edgecolors='k', 
                         s = 40)  # Change the denominator as per n_clusters

    # Create a mplcursors object to manage the data point interaction
    cursor = mplcursors.cursor(scatter, hover=True)

    #axes
    ax.set_title('Embedding clusters visualization in 2D')  # Add a title
    ax.set_xlabel('X_1')  # Add x-axis label
    ax.set_ylabel('X_2')  # Add y-axis label

    # Define how each annotation should look
    @cursor.connect("add")
    def on_add(sel):
        if displayed_column:
            sel.annotation.set_text(labels.iloc[sel.index][displayed_column])
        else:
            sel.annotation.set_text(labels.iloc[sel.index])
        sel.annotation.get_bbox_patch().set(facecolor='white', alpha=0.95) # Set annotation's background color
        sel.annotation.set_fontsize(14) 

    plt.show()
    
def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

def encode_texts_to_embeddings(sentences):
    try:
        embeddings = get_embeddings(sentences)
        return embeddings
    except Exception:
        return [None for _ in range(len(sentences))]
        
def encode_text_to_embedding_batched(sentences, api_calls_per_second = 0.33, batch_size = 5):
    # Generates batches and calls embedding API
    
    embeddings_list = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total = math.ceil(len(sentences) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())

    is_successful = [
        embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    ]
    embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in embeddings_list if embedding is not None])
    )
    return embeddings_list_successful