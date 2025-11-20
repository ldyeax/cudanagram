#!/usr/bin/env python3
import sys
import itertools

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
    # Ensure all are strings
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


def best_order_bruteforce(words):
    """
    Given a list of words, try all unique permutations and return
    the one with the highest GPT-2 log-probability.
    """
    words = list(words)

    # Short-circuit trivial cases
    if len(words) <= 1:
        return words

    if len(words) > 8:
        print(
            f"Warning: brute-force with {len(words)} words => {len(words)}! permutations; "
            f"this may be very slow.",
            file=sys.stderr,
        )

    best_perm = None
    best_score = float("-inf")

    # Use set() to avoid duplicate permutations when there are repeated words
    for perm in set(itertools.permutations(words)):
        seq = list(perm)
        s = score_sentence(seq)
        if s > best_score:
            best_score = s
            best_perm = seq

    return best_perm


def main():
    """
    Read lines from stdin, each line = a sentence / bag of words.
    For each line:
      - split into words
      - run brute-force ordering
      - print reordered sentence
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
		if 'THE' not in line:
			continue
		l2 = line.split(' ')
		if len(l2) != 9:
			continue


        words = line.split()
        best = best_order_bruteforce(words)
        print(" ".join(best))
        # Flush to ensure timely output
        sys.stdout.flush()


if __name__ == "__main__":
    main()

