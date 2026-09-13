# OpenAI Sandbox

Small Python examples for OpenAI API calls, structured output, validation,
retries, tool calling, and text embeddings.
**This is not a vibe coding project; it is a demo/sandbox project for learning LLM API integrations.**

## Setup

Run from the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install openai python-dotenv "pydantic[email]" pydantic-ai numpy scikit-learn matplotlib mplcursors
```

Create `.env` in the project folder:

```text
OPENAI_API_KEY=your-api-key
```

All examples call the OpenAI API, so they require a valid API key and may incur
API usage charges.

## LLM examples: `main.py`

```powershell
python main.py single
```

Prints one free-form response to the sample prompt.

```powershell
python main.py retry --retries 5
```

Prints validated JSON for a `CustomerQuery`. Invalid responses are sent back to
the model for correction until validation succeeds or the retry limit is reached.

```powershell
python main.py reviews
```

Prints one `ReviewSummaryModel` per sample review, containing a short summary
and a score from 1 to 10.

```powershell
python main.py agent
```

Prints a validated `SupportTicket` JSON object. The agent may call the local FAQ
or order-status tools before creating the ticket.

These examples demonstrate chat completions, Pydantic model validation,
structured output, retry prompts, tool calling, and `pydantic-ai` agents.

## Embedding examples: `openai_embeddings.py`

To run an embedding example, edit the `__main__` block in
`openai_embeddings.py` and replace the function call:

```python
if __name__ == "__main__":
	word_embeddings()
```

Use `word_embeddings()` to print embedding vector lengths and sample values,
cosine similarities between sentences, averaged word-vector shapes, and
sentence similarity. Use `visualize_embeddings()` to create the visualization.
To run the Stack Overflow clustering example, replace the call in the
`__main__` block with:

```python
if __name__ == "__main__":
	example_usage_stackoverflow()
```

Then run:

```powershell
python openai_embeddings.py
```

With `visualize_embeddings()`, the script prints the embedding and PCA array
shapes, then opens a 2D PCA scatter plot and an embeddings heatmap. Hover over
scatter points to see their labels.

The Stack Overflow example processes roughly 2,000 questions from
`so_database_app.csv`. It sends the question text to the OpenAI text embedding
model, groups the resulting embedding vectors with KMeans, and uses PCA to
project them into two dimensions for visualization. Questions from the same
dataset category tend to appear together in the 2D cluster plot. KMeans is
unsupervised: it receives only the question embeddings, while the dataset
categories are shown afterward to inspect how well the clusters correspond to
the programming-language categories.

[TODO: Data set to be added later]

These examples demonstrate OpenAI embeddings, cosine similarity, NumPy array
operations, PCA dimensionality reduction, and Matplotlib visualization.
