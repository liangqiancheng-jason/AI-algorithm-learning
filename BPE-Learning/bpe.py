"""
Minimal Byte Pair Encoding (BPE) tokenizer demonstration with verbose logging.

This script shows how one can learn merges from a small corpus and then apply
the learned rules to encode new text. BPE is widely used by large language
models because it provides a compact subword vocabulary that still captures
common patterns in natural language. Extra print statements are included so
that each key step in the algorithm is easy to follow.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Word = Tuple[str, ...]  # Represents a tokenized word as a tuple of symbols.
Pair = Tuple[str, str]


def preprocess(text: str, *, verbose: bool = False) -> List[str]:
    """
    Convert raw text into a list of word tokens.
    """
    tokens = [token.strip().lower() for token in text.split() if token.strip()]
    if verbose:
        print("=== Preprocess ===")
        print("Raw text:")
        print(text)
        print(f"Tokens ({len(tokens)}): {tokens}")
        print()
    return tokens


def initialize_vocab(
    words: Iterable[str],
    *,
    verbose: bool = False,
) -> Counter[Word]:
    """
    Build an initial vocabulary mapping from word-as-symbol-tuples to frequency.

    Each word is split into characters and we add a special </w> symbol so BPE
    can learn when a token should end the word.
    """
    vocab: Counter[Word] = Counter()
    if verbose:
        print("=== Initialize Vocabulary ===")
    for word in words:
        symbols = tuple(word) + ("</w>",)
        vocab[symbols] += 1
        if verbose:
            print(f"  {word!r} -> {symbols} (count={vocab[symbols]})")
    if verbose:
        print("\nFull vocabulary snapshot:")
        for entry, freq in vocab.items():
            print(f"  {entry}: {freq}")
        print()
    return vocab


def get_pair_counts(
    vocab: Counter[Word],
    *,
    verbose: bool = False,
) -> Dict[Pair, int]:
    """
    Count how frequently each adjacent symbol pair appears in the vocabulary.
    """
    pair_counts: Dict[Pair, int] = defaultdict(int)
    for word, frequency in vocab.items():
        for left, right in zip(word, word[1:]):
            pair_counts[(left, right)] += frequency
    if verbose:
        print("Pair counts:")
        for pair, count in sorted(
            pair_counts.items(), key=lambda item: item[1], reverse=True
        ):
            print(f"  {pair}: {count}")
        if not pair_counts:
            print("  (no pairs remain)")
        print()
    return pair_counts


def merge_pair(
    pair: Pair,
    vocab: Counter[Word],
    *,
    verbose: bool = False,
) -> Counter[Word]:
    """
    Replace every occurrence of the given pair with a merged symbol.
    """
    merged_vocab: Counter[Word] = Counter()
    merged_symbol = "".join(pair)

    if verbose:
        print(f"Merging pair: {pair} -> '{merged_symbol}'")

    for word, frequency in vocab.items():
        new_word: List[str] = []
        skip = False
        for idx, symbol in enumerate(word):
            if skip:
                skip = False
                continue

            if idx < len(word) - 1 and (symbol, word[idx + 1]) == pair:
                new_word.append(merged_symbol)
                skip = True  # Skip the next symbol because it is now merged.
            else:
                new_word.append(symbol)

        merged_vocab[tuple(new_word)] += frequency
        if verbose:
            print(f"  {word} (x{frequency}) -> {tuple(new_word)}")

    if verbose:
        print()
    return merged_vocab


@dataclass
class BPEResult:
    merges: List[Pair]
    vocab: Counter[Word]


def learn_bpe(
    corpus: str,
    num_merges: int = 30,
    *,
    verbose: bool = False,
) -> BPEResult:
    """
    Learn BPE merge operations from a text corpus.
    """
    if verbose:
        print("=== Learn BPE ===")
        print(f"Target number of merges: {num_merges}")
        print()

    words = preprocess(corpus, verbose=verbose)
    vocab = initialize_vocab(words, verbose=verbose)
    merges: List[Pair] = []

    for merge_idx in range(num_merges):
        if verbose:
            print(f"--- Merge step {merge_idx + 1} ---")

        pair_counts = get_pair_counts(vocab, verbose=verbose)
        if not pair_counts:
            if verbose:
                print("No more pairs to merge, stopping early.\n")
            break

        best_pair = max(pair_counts.items(), key=lambda item: item[1])[0]
        if verbose:
            print(f"Selected best pair: {best_pair}\n")

        vocab = merge_pair(best_pair, vocab, verbose=verbose)
        merges.append(best_pair)

    if verbose:
        print("=== Finished Learning ===")
        print(f"Total merges learned: {len(merges)}")
        if merges:
            print("Learned merges (in order):")
            for rank, pair in enumerate(merges):
                print(f"  {rank:02d}: {pair}")
        print()

    return BPEResult(merges=merges, vocab=vocab)


def apply_bpe_to_word(
    word: str,
    merges: Sequence[Pair],
    *,
    verbose: bool = False,
) -> List[str]:
    """
    Encode a single word using learned BPE merges.
    """
    symbols: List[str] = list(word) + ["</w>"]
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

    if verbose:
        print(f"Encoding word: {word!r}")
        print(f"  Start symbols: {symbols}")

    def get_pairs(symbol_seq: Sequence[str]) -> List[Pair]:
        return [
            (symbol_seq[i], symbol_seq[i + 1]) for i in range(len(symbol_seq) - 1)
        ]

    while True:
        candidate_pairs = get_pairs(symbols)
        ranked_pairs = [
            (merge_ranks[pair], pair)
            for pair in candidate_pairs
            if pair in merge_ranks
        ]
        if not ranked_pairs:
            if verbose:
                print("  No applicable merges remain.\n")
            break

        _, best_pair = min(ranked_pairs)  # Apply the earliest learned merge first.

        if verbose:
            print(f"  Candidate pairs: {candidate_pairs}")
            print(f"  Applying merge: {best_pair}")

        new_symbols: List[str] = []
        idx = 0
        while idx < len(symbols):
            if (
                idx < len(symbols) - 1
                and symbols[idx] == best_pair[0]
                and symbols[idx + 1] == best_pair[1]
            ):
                new_symbols.append("".join(best_pair))
                idx += 2
            else:
                new_symbols.append(symbols[idx])
                idx += 1
        symbols = new_symbols

        if verbose:
            print(f"  Symbols after merge: {symbols}")

    tokens = [symbol for symbol in symbols if symbol != "</w>"]
    if verbose:
        print(f"  Final tokens: {tokens}\n")
    return tokens


def encode(
    text: str,
    merges: Sequence[Pair],
    *,
    verbose: bool = False,
) -> List[List[str]]:
    """
    Encode every word in the input text using a fixed set of merges.
    """
    words = preprocess(text, verbose=verbose)
    outputs: List[List[str]] = []
    if verbose:
        print("=== Encode Text ===")
    for word in words:
        tokens = apply_bpe_to_word(word, merges, verbose=verbose)
        outputs.append(tokens)
        if verbose:
            print(f"Word '{word}' -> tokens {tokens}")
    if verbose:
        print()
    return outputs


def main() -> None:
    sample_corpus = (
       "大型语言模型通过对大量语料进行学习，从中捕捉语言的规律。"
    )

    log_path = Path(__file__).with_name("bpe_verbose_log.txt")
    with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(log_file):
        print("### Sample Corpus ###")
        print(sample_corpus)
        print()

        bpe_result = learn_bpe(sample_corpus, num_merges=25, verbose=True)
        merges_preview = ", ".join(
            [" ".join(pair) for pair in bpe_result.merges[:10]]
        )

        print("Top learned merges:")
        print(merges_preview or "(no merges learned)")
        print()

        demo_sentence = "大型语言模型通过对大量语料进行学习"
        print(f"Encoding: {demo_sentence!r}\n")
        encoded_words = encode(demo_sentence, bpe_result.merges, verbose=True)
        demo_words = preprocess(demo_sentence)
        for word, tokens in zip(demo_words, encoded_words):
            print(f"  {word:<12} -> {tokens}")


if __name__ == "__main__":
    main()
