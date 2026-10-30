"""Prepare the data for all experiments. Run once (the experiment scripts also do this if needed).

Extracts the GPT-2 Small input embedding table to --data-dir/embeddings.npy and
prints diagnostics. The token counts that choose the subvocabularies
(token_counts.npy and meta.json: GPT-2 token frequencies in a seeded random
sample of about 200 million tokens of English Wikipedia) are shipped with the
repository and kept. --recount counts Wikipedia again (several minutes, mostly downloading).
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
    p.add_argument("--recount", action="store_true",
                   help="count Wikipedia tokens again instead of keeping the shipped counts")
    args = p.parse_args()
    data.prepare(args.data_dir, args.max_tokens, args.batch_size, args.seed, recount=args.recount)


if __name__ == "__main__":
    main()
