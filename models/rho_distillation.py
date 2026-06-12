import torch
import torch.nn as nn
import torch.nn.functional as F


class RhoConditionedLoss(nn.Module):
    """
    ρ-conditioned rationale distillation loss for cross-encoder training.

    Standard cross-entropy on ESCI hard labels works well when ColBERT's
    ranking already aligns with human relevance judgments (high ρ). For
    queries where ColBERT is unreliable (low ρ), LLM-generated soft label
    distributions provide better supervision signal.

    Loss = (1 - λ) · CE(logits, hard_label)
         +      λ  · KL(softmax(logits/T) ‖ soft_label)

    λ(ρ) = clip(1 - ρ, 0, 1)
      → λ ≈ 1 when ρ ≈ 0 (ColBERT unreliable → trust LLM more)
      → λ ≈ 0 when ρ ≈ 1 (ColBERT reliable → hard labels sufficient)

    Args:
        temperature: KL distillation temperature (default 2.0 — softens predictions)
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits:      torch.Tensor,   # [batch, 4]  — raw model outputs
        hard_labels: torch.Tensor,   # [batch]     — int, 0=E 1=S 2=C 3=I
        soft_labels: torch.Tensor,   # [batch, 4]  — LLM probability distribution
        rho:         torch.Tensor,   # [batch]     — per-query Spearman ρ ∈ [-1, 1]
        has_soft:    torch.Tensor,   # [batch]     — bool mask: True if LLM label exists
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, hard_labels)                          # [batch]

        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        kl_loss   = F.kl_div(
            log_probs,
            soft_labels.clamp(min=1e-8),
            reduction="none",
        ).sum(dim=-1)                                                   # [batch]

        lam = (1.0 - rho).clamp(0.0, 1.0)                             # [batch]

        # Only apply KL term where LLM soft labels exist
        kl_masked = kl_loss * has_soft.float()
        lam_masked = lam    * has_soft.float()

        loss = (1.0 - lam_masked) * ce_loss + lam_masked * kl_masked
        return loss.mean()
