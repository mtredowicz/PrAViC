import torch
from torch import nn
from torch.nn import functional as F

class FastCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        ignore_index=-100,
        lambda_value=0,
        eps=1e-8,
    ) -> None:
        super(FastCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lambda_value = lambda_value
        self.eps = eps

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        proba = torch.softmax(outputs, dim=-1).clamp(min=self.eps) 
        max_probs, idx = torch.max(proba, dim=1)
        max_probs = max_probs.clamp(min=self.eps)
        cumulative_max_probs, _ = torch.cummax(proba, dim=1)
        sum_cumulative_max_probs = torch.sum(cumulative_max_probs, dim=1)

        e_x = outputs.size(1) - sum_cumulative_max_probs / max_probs
        i_x = e_x / outputs.size(1)

        log_term = torch.log(self.lambda_value + (1 - self.lambda_value) * i_x + self.eps)

        fast_loss = torch.gather(-torch.log(max_probs) + log_term, 1, labels.unsqueeze(1)).squeeze(1)

        if self.weight is not None:
            fast_loss = fast_loss * self.weight

        if self.reduction == "mean":
            fast_loss = fast_loss.mean()
        elif self.reduction == "sum":
            fast_loss = fast_loss.sum()

        return fast_loss, idx, max_probs, e_x
