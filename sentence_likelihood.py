#!/usr/bin/env python3
import sys
from collections import Counter

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---- Model setup ----

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"

print(f"Loading {MODEL_NAME} on {DEVICE}...", file=sys.stderr)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Optional: cache sentence scores so we don't recompute the same prefix repeatedly
_score_cache = {}


def score_sentence(words):
    """
    Score a sequence of words using GPT-2.
    Returns total log-probability (higher is better).
    """
    key = tuple(words)
    if key in _score_cache:
        return _score_cache[key]

    text = " ".join(words)
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)

    with torch.no_grad():
        out = model(input_ids, labels=input_ids)

    # out.loss is mean NLL over tokens (except first token)
    seq_len = input_ids.shape[1]
    total_logprob = -out.loss.item() * (seq_len - 1)

    _score_cache[key] = total_logprob
    return total_logprob


def greedy_order(words):
    """
    Given a list of words, greedily build a sequence:
    start empty, and at each step choose the remaining word
    that yields the highest total log-probability when appended.
    """
    words = list(words)
    full_multiset = Counter(words)
    used = Counter()
    seq = []

    while len(seq) < len(words):
        best_word = None
        best_score = float("-inf")

        for w in full_multiset:
            if used[w] >= full_multiset[w]:
                continue

            candidate = seq + [w]
            s = score_sentence(candidate)

            if s > best_score:
                best_score = s
                best_word = w

        seq.append(best_word)
        used[best_word] += 1

    return seq


def main():
    """
    Read lines from stdin, each line = a sentence / bag of words.
    For each line:
      - split into words
      - run greedy ordering
      - print reordered sentence
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        words = line.split()
        reordered = greedy_order(words)
        print(" ".join(reordered))


if __name__ == "__main__":
    main()
