# acre-rag

Ask questions about an SEC 10-Q filing and get answers **with citations**, from a pipeline that runs **entirely on your laptop**: free open-source models, no API keys, nothing sent to the cloud.

The example filing is Ares Commercial Real Estate Corporation's (ACRE) Form 10-Q for the quarter ended June 30, 2026.

```
BUILD ONCE (indexing)
 [SEC 10-Q] --> [Parse] --> [Chunk] --> [Embed + store]
  EDGAR         text +      fixed vs     all-MiniLM-L6-v2
  download      tables      semantic     into Chroma
                                              |
EVERY QUESTION (LangGraph)                    v
 [Question] --> [Retrieve] --> [Rerank] --> [Answer] --> [Verify] --(fail: retry once)--> answer
                 top 20         best 5       phi4-mini    citations &
                                             (Ollama)     numbers checked
                                                          in code
```

The model answers only from the 5 excerpts it is given and cites their ids. Code then checks the answer: every cited id must be one of the excerpts, every number must appear in them, and a "Not found" is questioned if an excerpt looked relevant. When a check fails, the model gets one retry with feedback on what was wrong, and the app tells you to double-check.

More detail: [ARCHITECTURE.md](ARCHITECTURE.md). Plain-language explanations of the concepts: [notes/](notes/).

## What you need

- **Python 3.12.** Some dependencies don't have wheels for newer versions yet.
- **[Ollama](https://ollama.com/download)**, to run the local LLM.
- **About 5 GB of free RAM** while answering (about 3–3.5 GB for the LLM, about 1 GB for the embedding and reranking models) and about 4 GB of disk space for the models.
- No GPU needed. Answers take about 10–60 seconds on a laptop CPU.

## Setup

```bash
# 1. Python environment
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. The local LLM (2.5 GB download)
ollama pull phi4-mini

# 3. Your SEC identity (the SEC requires a name and email on every download)
copy .env.example .env            # macOS/Linux: cp .env.example .env
# then edit .env
```

## Build the index (once)

Run the notebooks in `notebooks/` in order, with the `.venv` kernel:

| Notebook | Does | Time |
|---|---|---|
| `00_setup_check` | Checks packages, `.env` and Ollama | seconds |
| `01_download_10q` | Downloads the filing from SEC EDGAR | seconds |
| `02_parse` | Extracts text and tables, tagged by Part / Item | seconds |
| `03_chunk` | Cuts the text into chunks, two ways: fixed-size and semantic | ~1 min |
| `04_embed_store` | Embeds the chunks into a Chroma index at `data/chroma_db/` | 1–2 min |

The first runs download two small Hugging Face models (~90 MB each).

## Ask questions

```bash
streamlit run app.py
```

The very first time, Streamlit asks for an email address in the terminal; press Enter to skip it. Stop the app with `Ctrl+C` in the terminal.

This opens a page in your browser (or open http://localhost:8501). Type a question, or click one of the examples. You'll see the answer with its cited excerpt ids, the 5 excerpts the model read, and whether the checks passed. The sidebar lets you switch to semantic chunks or the quote-first prompt (v2) to compare.

The notebooks `05_retrieve_rerank` and `06_answer` show each step on its own, and `07_evaluate` scores the whole flow on the test set in `eval/questions.json`. Results are saved to `reports/eval_results.csv`.

## Results

From `notebooks/07_evaluate.ipynb`: 8 test questions whose answers were checked by hand against the filing, with `phi4-mini` on a laptop CPU.

| Setup | Correct | Right excerpt retrieved | Passed checks | Wrong but passed checks | Avg time |
|---|---|---|---|---|---|
| **fixed chunks · prompt v1 (default)** | **8 / 8** | 7 / 7 | 7 / 8 | 0 | 17 s |
| fixed chunks · prompt v2 (quote first) | 7 / 8 | 7 / 7 | 5 / 8 | 1 | 22 s |
| semantic chunks · prompt v2 | 6 / 8 | 6 / 7 | 2 / 8 | 0 | 31 s |

"Right excerpt retrieved" counts 7 questions, because the eighth (who the auditor is) correctly has no answer in the filing.

What this shows:

- **Fixed-size chunks beat semantic chunks** on every measure. Semantic chunking once failed to retrieve the right excerpt, and the model then answered with only part of the CECL reserve.
- **The retry matters.** Half the questions needed a second attempt. Without it, the same model and prompt answered "Not found" for questions whose answers were in the excerpts.
- **Prompt v2 asks for a word-for-word quote**, which a 3.8B model tends to paraphrase. Its answers are usually right, but its own quote check often flags them.

Caveats: 8 questions is a small test, and one question is 12.5 percentage points, so small differences are hints, not proof. "Correct" means the answer contains the key number (or says "Not found" when it should); it doesn't check the whole sentence. Results can change between runs and Ollama versions.

## Project layout

| Path | Holds |
|---|---|
| `notebooks/` | One notebook per stage, 00–07 |
| `src/pipeline.py` | The steps: retrieve, rerank, answer, verify |
| `src/graph.py` | The LangGraph flow, with one retry |
| `app.py` | The Streamlit app |
| `eval/questions.json` | Test questions, with answers checked by hand against the filing |
| `data/` | The filing, the parsed text and chunks, and the Chroma index |
| `notes/` | Plain-language explanations of the concepts |

## Models and licences

| Role | Model | Licence |
|---|---|---|
| Embeddings | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache 2.0 |
| Reranker | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Apache 2.0 |
| Answers | [phi4-mini](https://ollama.com/library/phi4-mini) (Microsoft, 3.8B) via Ollama | MIT |

To try another Ollama model, pull it and set the environment variable `ACRE_LLM_MODEL`, e.g. `ollama pull qwen3:4b`, then set `ACRE_LLM_MODEL=qwen3:4b` before starting the app or notebook.

## Limitations

- A 3.8B model is small. It sometimes misses an answer that is in the excerpts, or ignores an instruction. The checks catch some of these, but not all: an answer can pass every check and still be wrong (for example, a six-month figure given for a three-month question). See `07_evaluate` for the measured accuracy.
- One filing only. The pipeline is built so more can be added, but that isn't done yet.
- **Not investment advice.** Always check the quoted source.

## Licence

The code and notes in this repository are released under the [MIT License](LICENSE). The models are not included; each keeps its own licence (see above). The 10-Q is a public filing from SEC EDGAR.
