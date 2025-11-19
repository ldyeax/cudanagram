#!/usr/bin/env python3
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---- Model setup ----

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"

print(f"Loading {MODEL_NAME} on {DEVICE}...", file=sys.stderr)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

_score_cache = {}


def score_sentence(words):
    """
    Score a sequence of words using GPT-2.
    Returns total log-probability (higher is better).
    """
    # Normalize to strings
    words = [str(w) for w in words]
    key = tuple(words)

    if key in _score_cache:
        return _score_cache[key]

    text = " ".join(words)
    if not text.strip():
        _score_cache[key] = float("-inf")
        return _score_cache[key]

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)

    with torch.no_grad():
        out = model(input_ids, labels=input_ids)

    # out.loss is mean NLL over (seq_len-1) predicted tokens
    seq_len = input_ids.shape[1]
    total_logprob = -out.loss.item() * max(seq_len - 1, 1)

    _score_cache[key] = total_logprob
    return total_logprob


def greedy_order(words):
    """
    Given a list of words, greedily build a sequence:
      - For the *first* word, use a 2-word lookahead so GPT-2 actually has
        something to score.
      - Afterwards, at each step, append the remaining word that gives the
        best total log-probability.
    """
    remaining = list(words)
    seq = []

    # Special handling for the first word (and maybe second)
    if len(remaining) == 1:
        return remaining[:]  # nothing to reorder

    # Choose first word by looking at the best 2-word continuation
    best_first_idx = None
    best_first_score = float("-inf")

    for i, w1 in enumerate(remaining):
        best_for_w1 = float("-inf")
        # try pairing w1 with each possible w2 (including duplicates if present)
        for j, w2 in enumerate(remaining):
            if i == j and len(remaining) > 1:
                continue  # avoid w1==w2 if there are other options
            s = score_sentence([w1, w2])
            if s > best_for_w1:
                best_for_w1 = s

        if best_for_w1 > best_first_score:
            best_first_score = best_for_w1
            best_first_idx = i

    if best_first_idx is None:
        best_first_idx = 0  # fallback

    first_word = remaining.pop(best_first_idx)
    seq.append(first_word)

    # If there is only one left after choosing the first, we’re done
    if not remaining:
        return seq

    # Now continue greedy: append the remaining word that gives best score
    while remaining:
        best_idx = None
        best_score = float("-inf")

        for i, w in enumerate(remaining):
            candidate = seq + [w]
            s = score_sentence(candidate)
            if s > best_score:
                best_score = s
                best_idx = i

        if best_idx is None:
            best_idx = 0  # safety fallback

        best_word = remaining.pop(best_idx)
        seq.append(best_word)

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

