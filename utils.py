from typing import Type
import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from openai_config import DEFAULT_MODEL, get_openai_client


def get_openai_api_key() -> str | None:
    """Load the project environment and return the OpenAI API key."""
    load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key() -> str | None:
    """Load the project environment and return the Hugging Face token."""
    load_dotenv(find_dotenv())
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")


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


def build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index",
):
    """Build or load an index that retrieves surrounding sentence windows."""
    from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
    from llama_index.core.node_parser import SentenceWindowNodeParser

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document],
            transformations=[node_parser],
            llm=llm,
            embed_model=embed_model,
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            llm=llm,
            embed_model=embed_model,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    """Create a query engine that restores windows and reranks results."""
    from llama_index.core.postprocessor import (
        MetadataReplacementPostProcessor,
        SentenceTransformerRerank,
    )

    postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n,
        model="BAAI/bge-reranker-base",
    )
    return sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postprocessor, rerank],
    )


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    """Build or load an index that merges parent context during retrieval."""
    from llama_index.core import (
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            llm=llm,
            embed_model=embed_model,
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            llm=llm,
            embed_model=embed_model,
        )

    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    """Create a query engine that merges retrieved child nodes with parents."""
    from llama_index.core.postprocessor import SentenceTransformerRerank
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import AutoMergingRetriever

    base_retriever = automerging_index.as_retriever(
        similarity_top_k=similarity_top_k
    )
    retriever = AutoMergingRetriever(
        base_retriever,
        automerging_index.storage_context,
        verbose=True,
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n,
        model="BAAI/bge-reranker-base",
    )
    return RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[rerank],
    )


def get_trulens_recorder(query_engine, feedbacks, app_id):
    """Create a TruLens recorder for a query engine."""
    from trulens_eval import TruLlama

    return TruLlama(query_engine, app_id=app_id, feedbacks=feedbacks)


def build_trulens_feedbacks():
    """Build the tutorial's TruLens relevance and groundedness feedbacks."""
    import numpy as np
    from trulens_eval import Feedback, OpenAI, TruLlama
    from trulens_eval.feedback import Groundedness

    openai_provider = OpenAI()
    answer_relevance = (
        Feedback(
            openai_provider.relevance_with_cot_reasons,
            name="Answer Relevance",
        )
        .on_input_output()
    )
    context_relevance = (
        Feedback(
            openai_provider.relevance_with_cot_reasons,
            name="Context Relevance",
        )
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )
    groundedness_provider = Groundedness(groundedness_provider=openai_provider)
    groundedness = (
        Feedback(
            groundedness_provider.groundedness_measure_with_cot_reasons,
            name="Groundedness",
        )
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(groundedness_provider.grounded_statements_aggregator)
    )
    return [answer_relevance, context_relevance, groundedness]


def get_prebuilt_trulens_recorder(query_engine, app_id, feedbacks=None):
    """Create a TruLens recorder using supplied or tutorial feedbacks."""
    return get_trulens_recorder(
        query_engine,
        feedbacks if feedbacks is not None else build_trulens_feedbacks(),
        app_id,
    )
