"""Prepare the data for all experiments. Run once (the experiment scripts also do this if needed).

Extracts the GPT-2 Small input embedding table, counts GPT-2 token frequencies
in a seeded random sample of English Wikipedia, and prints diagnostics. Outputs go to
--data-dir: embeddings.npy, token_counts.npy and meta.json.
"""

import argparse

from sae_raw import data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--max-tokens", type=int, default=200_000_000,
                   help="stop counting after roughly this many Wikipedia tokens")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="seed for the order of Wikipedia row groups")
    args = p.parse_args()
    data.prepare(args.data_dir, args.max_tokens, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
