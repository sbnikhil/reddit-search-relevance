import torch
import torch.nn as nn
from transformers import AutoModel

LABEL_ORDER   = ["E", "S", "C", "I"]
LABEL_TO_IDX  = {l: i for i, l in enumerate(LABEL_ORDER)}
# Mirrors LambdaMART label_gain=[0,1,3,7]: E→gain 3, S→gain 1, C/I→gain 0
SCORE_WEIGHTS = torch.tensor([1.0, 0.1, 0.01, 0.0])


class CrossEncoder(nn.Module):
    """
    ModernBERT cross-encoder for query-product relevance scoring.

    Input:  [CLS] query [SEP] title description [SEP]
    Output: logits over {E, S, C, I}
    Score:  P(E) + 0.1·P(S) + 0.01·P(C)  — consistent with ESCI gain mapping

    Trained with ρ-conditioned rationale distillation (see models/rho_distillation.py).
    """
    MODEL_NAME = "answerdotai/ModernBERT-base"

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(LABEL_ORDER))

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]      # [batch, hidden]
        return self.classifier(cls)               # [batch, 4]

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
        """Scalar relevance score for ranking — higher = more relevant."""
        probs   = torch.softmax(self.forward(input_ids, attention_mask), dim=-1)
        weights = SCORE_WEIGHTS.to(probs.device)
        return (probs * weights).sum(dim=-1)      # [batch]
