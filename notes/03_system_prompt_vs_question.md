# System prompt vs. question

How the Answer stage (`06_answer`) talks to the LLM, and why the rules and the question are sent as separate messages.

## The analogy

> **The system prompt is the job description. The human message is each request.**

- **Job description** (system): handed to a new analyst on day one, and the same for every task. *"Only use the filing, cite your sources, say so if you can't find it."*
- **Request** (human): each ticket that lands on their desk. *"Here are 5 pages. What was the net interest margin?"*
- **The work handed back** (assistant): the model's answer.

## The three roles in a chat

| Role | Who writes it | What it's for | In our notebook |
|---|---|---|---|
| **System** | The app builder (you) | Standing instructions: who the model is, the rules, the output format | `INSTRUCTIONS` / `INSTRUCTIONS_V2`: **the same for every question** |
| **Human** (also called **user**) | The person asking, or code on their behalf | The actual request | The 5 excerpts + the question: **different every time** |
| **Assistant** | The model | Its reply | The answer we parse and verify |

## In our code

```python
messages = [
    ("system", instructions),                                          # job description: fixed
    ("human", f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"),     # the request: changes each time
]
```

The **excerpts go in the human message**, not the system one. Like the question, they change every time. The system part stays the stable "how to work"; the human part holds "what to work on".

## What the model actually receives

The roles aren't separate channels. Ollama joins them into **one piece of text** with special markers, using a format called a **chat template**. For phi4-mini it looks roughly like this:

```
<|system|>You answer questions about the Form 10-Q of ... Rules: ...<|end|>
<|user|>Excerpts: [fixed-0180] ... Question: What was the net interest margin this quarter?<|end|>
<|assistant|>
```

The model continues the text after `<|assistant|>`, and that continuation is the answer. Underneath, it is still "predict the next word".

## So why keep them separate?

1. **The model was trained on the difference.** It has seen millions of chats where the system part set the rules and the user part made requests, so rules in the system prompt tend to be followed more consistently.
2. **Reuse.** The rules are written once and sent with every question. Only the request changes.
3. **Some protection.** The excerpts come from a document, not from us. If a document contained text like *"ignore your rules and…"*, keeping our rules in the system role makes the model more likely to stick to them. This is not a guarantee, especially with small models. This kind of attack is called **prompt injection**.

## Why changing the instructions changes the answer

- An LLM predicts the next word from **all the text in front of it**, and the instructions are part of that text. Change them, and the likely next words change.
- **The first words steer the rest.** With the old prompt the model often started with "Not…", and "found in the provided excerpts." followed. The Option A prompt starts with `QUOTE:`, so the model copies evidence first, and the answer tends to follow from it.
- `temperature=0` means the **same input gives the same output**. A different prompt is a different input.
- **Small models are extra sensitive** to wording and to how many rules they get. Writing instructions well is called **prompt engineering**.
- **Limits:** a prompt can't add knowledge. If the right chunk isn't among the 5, no prompt will fix it; that is retrieval's job.

## In one line

**The system prompt is the job description (fixed rules, set by you); the human message is each request (the material and the question). To the model it's all one text, but it was trained to treat the system part as rules to follow throughout.**
