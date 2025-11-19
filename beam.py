#!/usr/bin/env python3
import sys
import math

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---- Model setup ----

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"

print(f"Loading {MODEL_NAME} on {DEVICE}...", file=sys.stderr)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Cache: words tuple -> (total_logprob, avg_logprob, num_pred_tokens)
_score_cache = {}


def score_sentence_stats(words):
    """Return (total_logprob, avg_logprob, num_pred)."""
    words = [str(w) for w in words]
    key = tuple(words)

    if key in _score_cache:
        return _score_cache[key]

    text = " ".join(words)
    if not text.strip():
        stats = (float("-inf"), float("-inf"), 0)
        _score_cache[key] = stats
        return stats

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)

    with torch.no_grad():
        out = model(input_ids, labels=input_ids)

    seq_len = input_ids.shape[1]
    num_pred = max(seq_len - 1, 1)
    total_logprob = -out.loss.item() * num_pred
    avg_logprob = total_logprob / num_pred

    stats = (total_logprob, avg_logprob, num_pred)
    _score_cache[key] = stats
    return stats


def score_sentence(words):
    total_logprob, _, _ = score_sentence_stats(words)
    return total_logprob


def beam_search_order(words, beam_size=5):
    words = list(words)
    if len(words) <= 1:
        return words

    beam = [([], words, 0.0)]

    for _ in range(len(words)):
        new_candidates = []
        for seq, remaining, _sc in beam:
            for i, w in enumerate(remaining):
                new_seq = seq + [w]
                new_remaining = remaining[:i] + remaining[i + 1 :]
                s = score_sentence(new_seq)
                new_candidates.append((new_seq, new_remaining, s))

        new_candidates.sort(key=lambda x: x[2], reverse=True)
        beam = new_candidates[:beam_size]

    best_seq, _, _best = max(beam, key=lambda x: x[2])
    return best_seq


def main():
    beam_size = 5
    if len(sys.argv) >= 2:
        try:
            beam_size = int(sys.argv[1])
        except ValueError:
            pass

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        words = line.split()
        best = beam_search_order(words, beam_size)
        total_logprob, avg_logprob, num_pred = score_sentence_stats(best)

        ppl = math.exp(-avg_logprob) if num_pred > 0 else float("inf")

        # FIRST COLUMN: avg_logprob (bare number)
        #print(
        #    f"{avg_logprob:.6f}\t"
        #    f"{' '.join(best)}\t"
        #    f"logprob={total_logprob:.6f}\t"
        #    f"ppl={ppl:.6f}"
        #)
        print(
            f"{avg_logprob:.6f}\t"
            f"{' '.join(best)}"
        )
        # flush 
        sys.stdout.flush()

if __name__ == "__main__":
    main()
