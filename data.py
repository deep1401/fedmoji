import numpy as np
import spacy
import torch
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split

categories = [
    "Smileys & Emotion",
    "People & Body",
    "Animals & Nature",
    "Symbols",
    "Objects",
    "Activities",
    "Flags",
    "Unknown",
    "Food & Drink",
    "Travel & Places",
]
# create mapping of category to integers for ease
category_dict = {cat: idx for idx, cat in enumerate(categories)}


def create_data_iter(df, category_dict, col1="text", col2="category"):
    """Creates data iterator as list of tuple consisting of `text` and `category`."""
    # maps category to integer
    df[col2] = df[col2].apply(lambda x: category_dict[x])

    data_iters = []
    for i in range(len(df)):
        data_iters.append((df[col1].iloc[i], df[col2].iloc[i]))
    return data_iters


def build_vocab(data_iters, tokenizer):
    """Builds vocabulary with tokens from data iterator."""

    def yield_tokens(data_iter, tokenizer):
        for text, _ in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(data_iters, tokenizer), specials=["<pad>", "<unk>", "xxMENTIONxx", "xxHASHTAGxx", "xxURLxx"]
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def create_final_dataset(data_iters, vocab, tokenizer):
    """Pipelining for processing raw text into a tokenized text."""
    def text_pipeline(x): return vocab(tokenizer(x))
    def label_pipeline(x): return int(x)
    dataset = []
    for text, label in data_iters:
        tokens = text_pipeline(text)
        tokens_fin = torch.tensor(tokens, dtype=torch.int64)
        label_fin = label_pipeline(label)
        dataset.append((tokens_fin, label_fin))
    return dataset


def create_global_sharing_data_for_causalfedgsd(train_data, size=0.3):
    """Creating global sharing data for model pre-training"""
    x = [*range(len(train_data))]
    y = list(train_data["category"].copy())
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=size, random_state=42, shuffle=True, stratify=y
    )
    train_data_new = train_data.iloc[x_train].copy()
    shared_data = train_data.iloc[x_test].copy()
    return train_data_new, shared_data


def create_global_sharing_data(train_data, size=0.3):
    """Creating global sharing data for model pre-training"""
    x = [*range(len(train_data))]
    y = list(train_data["category"].copy())
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=size, random_state=42, shuffle=True, stratify=y
    )
    train_data_new = train_data.iloc[x_train].copy()
    shared_data = train_data.iloc[x_test].copy()
    train_data_new.reset_index(drop=True, inplace=True)
    shared_data.reset_index(drop=True, inplace=True)
    return train_data_new, shared_data
