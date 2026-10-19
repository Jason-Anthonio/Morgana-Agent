"""Measure how fast the local LLM answers. Run from the project root:

    python -m scripts.benchmark_llm
"""

import statistics
import time

from morgana.llm import LocalLLM, SentenceSplitter

PROMPTS = [
    "Say hello in one short sentence.",
    "What is one useful thing I can do during a short break?",
    "Remember that I prefer concise answers. Confirm that in one sentence.",
]

llm = LocalLLM()
first_sentences, totals = [], []
for prompt in PROMPTS:
    messages = [
        {"role": "system", "content": "You are Morgana, a concise voice companion. Reply in one or two sentences."},
        {"role": "user", "content": prompt},
    ]
    splitter = SentenceSplitter()
    started = time.perf_counter()
    first_token = first_sentence = None
    reply = ""
    for piece in llm.stream(messages, max_tokens=80, temperature=0.2):
        elapsed = time.perf_counter() - started
        first_token = first_token or elapsed
        if splitter.feed(piece) and first_sentence is None:
            first_sentence = elapsed
        reply += piece
    total = time.perf_counter() - started
    first_sentences.append(first_sentence or total)
    totals.append(total)
    print(f"first token {first_token:.2f}s, first sentence {first_sentences[-1]:.2f}s, total {total:.2f}s: {reply!r}")

print(f"\nAverage first sentence {statistics.mean(first_sentences):.2f}s, average total {statistics.mean(totals):.2f}s")
