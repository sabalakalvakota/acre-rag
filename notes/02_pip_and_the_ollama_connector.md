# pip and the langchain-ollama connector

What `pip` is, and why the Answer stage (`06_answer`) needs the `langchain-ollama` package.

## pip: the app store for Python code

**pip** downloads and installs Python **packages**: ready-made code that other people wrote and shared.

- Packages live on **PyPI** ([pypi.org](https://pypi.org)), a free public library of about 600,000 Python packages.
- `pip install langchain-ollama` downloads that package, plus anything it depends on, and installs it in the project.
- `pip install -r requirements.txt` installs every package in the list at once. Anyone you share the project with sets it up this way.

### Why the long command?

```powershell
.venv\Scripts\python.exe -m pip install langchain-ollama
└──── which Python ─────┘ └ run pip ┘ └ what to install ┘
```

A laptop can have several versions of Python installed. A plain `pip install` might put the package into the wrong one, and then the notebook says "module not found". Starting the command with `.venv\Scripts\python.exe` makes sure the package goes into **this project's** Python, the same one the notebook kernel uses.

`.venv` is the project's **private set of packages**. It's kept separate so that different projects don't get in each other's way.

## The connector: a translator between Python and Ollama

The notebook and Ollama are **two separate programs**:

```
 Notebook (Python)              Ollama app (background)
 ─────────────────              ───────────────────────
 llm.invoke("question")  ──▶   runs phi4-mini
                         ◀──   sends back the answer
        (they talk over localhost:11434, a private "phone line" on your own laptop)
```

Ollama doesn't understand Python. It accepts web-style requests: small text messages in a format called **JSON**, such as:

```json
{"model": "phi4-mini", "messages": [{"role": "user", "content": "What is a 10-Q?"}]}
```

**`langchain-ollama` writes and reads those messages for you.** In the notebook you only write:

```python
llm = ChatOllama(model="phi4-mini", temperature=0)
llm.invoke("What is a 10-Q?")
```

and the connector:

1. turns the Python call into a JSON message,
2. sends it to Ollama at `localhost:11434`,
3. waits for the reply and turns it back into a Python object (`reply.content`).

### Why LangChain's connector, and not our own code?

- **The same interface for every model.** Switching from `phi4-mini` to another Ollama model is a one-word change.
- **It fits the rest of the project.** Chroma and HuggingFaceEmbeddings are already LangChain pieces, and LangGraph (coming later) plugs straight in.

## Things worth remembering

- The notebook doesn't need `ollama` on the PATH. It reaches the Ollama app over `localhost:11434`. The Ollama app only has to be **running** (look for the llama icon in the system tray).
- If `ollama` isn't recognised in a terminal right after installing it, the terminal is still using the old PATH. Restart VS Code, or use the full path: `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"`.

## In one line

**pip installs code packages into the project's `.venv`. `langchain-ollama` is one of those packages: it translates between the Python notebook and the Ollama app that runs phi4-mini.**
