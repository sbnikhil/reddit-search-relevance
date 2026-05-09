import torch
import torch.nn.functional as F
from config import COLBERT_TEMPERATURE


def colbert_loss(query_embs, pos_embs, neg_embs, model, temperature: float = COLBERT_TEMPERATURE):
    B = query_embs.size(0)
    neg_scores = model.maxsim(query_embs, neg_embs)
    sim = torch.einsum('iqd,jld->ijql', query_embs, pos_embs)
    scores_matrix = sim.max(dim=3).values.sum(dim=2)
    logits = torch.cat([scores_matrix, neg_scores.unsqueeze(1)], dim=1) / temperature
    labels = torch.arange(B, device=query_embs.device)
    return F.cross_entropy(logits, labels)
