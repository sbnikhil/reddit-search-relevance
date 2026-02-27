import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class ESCITripletDataset(Dataset):
    def __init__(self, parquet_path, model_name="bert-base-uncased", query_maxlen=32, doc_maxlen=128):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen

    def _tokenize(self, text, prefix, max_length):
        enc = self.tokenizer(
            f"{prefix} {text}",
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_ids, q_mask = self._tokenize(row["query"], "[Q]", self.query_maxlen)
        pos_text = row["positive_title"] + " " + (row["positive_description"] or "")
        neg_text = row["negative_title"] + " " + (row["negative_description"] or "")
        p_ids, p_mask = self._tokenize(pos_text, "[D]", self.doc_maxlen)
        n_ids, n_mask = self._tokenize(neg_text, "[D]", self.doc_maxlen)
        return q_ids, q_mask, p_ids, p_mask, n_ids, n_mask
