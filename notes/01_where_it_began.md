# Where it began

How a computer learned to turn text into "meaning" vectors, the idea behind the Embed + store and Retrieve stages of acre-rag.

No single person invented it. It was built up over about 80 years, each idea standing on the one before.

## Timeline

| Year | Who | Idea | Where it shows up in acre-rag |
|------|-----|------|-------------------------------|
| 1943 | Warren McCulloch & Walter Pitts | A math model of a brain cell: takes inputs, combines them, fires or not | The basic unit inside every model we use |
| 1954–57 | Zellig Harris, J.R. Firth (linguists) | "You shall know a word by the company it keeps": words in similar contexts have similar meanings | Why embeddings work at all |
| 1958 | Frank Rosenblatt | The **perceptron**: the first neuron that **learned its own weights** from examples | The 22 million weights in all-MiniLM |
| 1970–86 | Seppo Linnainmaa, Paul Werbos; made famous by David Rumelhart, Geoffrey Hinton & Ronald Williams | **Backpropagation**: tells every weight how to adjust after a mistake | How all-MiniLM was trained |
| 1993 | Jane Bromley, Yann LeCun & colleagues | **Siamese networks**: two inputs through the same network; matching pairs come out close | Question and chunk both go through all-MiniLM |
| 2003 | Yoshua Bengio's team | A network that predicts the next word learns a vector for each word along the way | Text as vectors |
| 2006 | Raia Hadsell, Sumit Chopra & Yann LeCun | **Contrastive loss**: pull matching pairs together, push unrelated ones apart | Exactly how all-MiniLM was trained |
| 2013 | Tomas Mikolov & colleagues (Google) | **word2vec**: word vectors learned from billions of words; *king − man + woman ≈ queen* | "Meaning as a direction" |
| 2017 | Ashish Vaswani & colleagues (Google) | **Transformer** (*"Attention Is All You Need"*): reads a whole sentence at once, works out which words relate | all-MiniLM, the reranker, and Claude are all Transformers |
| 2018 | Jacob Devlin & colleagues (Google) | **BERT**: a Transformer trained on Wikipedia and books that understands words in context | all-MiniLM descends from it |
| 2019 | Nils Reimers & Iryna Gurevych (TU Darmstadt) | **Sentence-BERT** and the `sentence-transformers` library: whole sentences become vectors you can compare with cosine distance | In `requirements.txt`; what `04_embed_store` uses |
| 2020 | Wenhui Wang & colleagues (Microsoft) | **MiniLM**: a small "student" model distilled from a large one | Why our model is only ~90 MB |
| 2021 | Hugging Face community project, with Google compute | MiniLM trained on **over 1 billion sentence pairs** → **all-MiniLM-L6-v2** | The model on this laptop |

## Recognition

- **2018 Turing Award** (computing's top prize): Geoffrey Hinton, Yann LeCun, Yoshua Bengio, for deep learning.
- **2024 Nobel Prize in Physics**: Geoffrey Hinton and John Hopfield, for foundational work on neural networks.

## How it connects to this project

- **The model is the mapmaker.** all-MiniLM learned, from a billion example pairs, where text belongs on a "map of meaning". Each chunk becomes one vector of 384 numbers.
- **Chroma is the ruler.** It has no intelligence; it measures cosine distance (the angle between two vectors) and uses its HNSW "road map" to find the closest chunks quickly.
- **Limits come from the training data.** all-MiniLM learned from general internet text, not SEC filings. That's why, for "What was the net interest margin this quarter?", it ranked a *definition* table above the chunk with the actual answer ($8.6 million). The reranker stage fixes some of this.

## In one line

Linguists had the idea that context reveals meaning. Rosenblatt and Hinton's generation made machines learn weights, LeCun's group taught them to pull matching pairs together, Google made it work at internet scale, and Reimers & Gurevych packaged it into the library this project uses.
