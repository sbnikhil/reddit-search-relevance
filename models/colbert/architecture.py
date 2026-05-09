import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from config import COLBERT_MODEL_NAME, COLBERT_DIM


class ColBERT(nn.Module):
    def __init__(self, model_name: str = COLBERT_MODEL_NAME, dim: int = COLBERT_DIM):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim)
        self.dim = dim

    def encode_query(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.linear(out.last_hidden_state), p=2, dim=-1)

    def encode_document(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embs = F.normalize(self.linear(out.last_hidden_state), p=2, dim=-1)
        return embs * attention_mask.unsqueeze(-1).float()

    def maxsim(self, query_embs, doc_embs):
        sim = torch.bmm(query_embs, doc_embs.transpose(1, 2))
        return sim.max(dim=2).values.sum(dim=1)

    def forward(self, query_ids, query_mask, doc_ids, doc_mask):
        return self.maxsim(
            self.encode_query(query_ids, query_mask),
            self.encode_document(doc_ids, doc_mask),
        )
