import torch
from torch import nn, Tensor


class CustomLoss(nn.Module):
    """
    A custom PyTorch loss function combining cross-entropy loss with early detection loss.

    Parameters:
        alpha (float): Hyperparameter controlling the trade-off between cross-entropy loss and early detection loss.
    """

    def __init__(self, reduction="mean", alpha: float = 0.5, use_early_detection=True):
        super(CustomLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(
            reduction="none" if use_early_detection else reduction
        )

        assert (
            0 <= alpha <= 1
        ), f"Parameter 'alpha' must be between 0 and 1, got {alpha}"
        self.alpha = alpha

        assert reduction in [
            "mean",
            "sum",
            "none",
        ], f"Parameter 'reduction' must be 'mean' or 'sum', got {reduction}"
        self.reduction = reduction
        self.use_early_detection = use_early_detection

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the custom loss function given input and target tensors.

        Parameters:
            x (torch.Tensor): Input tensor, typically the output of a neural network.
            y (torch.Tensor): Target tensor, representing the ground truth labels.

        Returns:
            torch.Tensor: Scalar tensor representing the computed loss.
        """
        assert y.dim() == 1, f"Target tensor 'y' must be one-dimensional, got {y.dim()}"

        prob = torch.sigmoid(x)
        p = torch.max(prob, dim=1).values

        loos_ce = self.criterion(p, y)

        i_prob = 1 - torch.sum(torch.cummax(prob, dim=1).values, dim=1).div(
            p * x.shape[1]
        )
        i_prob = torch.gather(i_prob, 1, p.argmax(dim=1).view(-1, 1)).view(-1)

        if self.use_early_detection:
            loss_early_detection = torch.log(self.alpha + (1 - self.alpha) * i_prob)

            if self.reduction == "mean":
                return torch.mean(loos_ce + loss_early_detection), i_prob
            elif self.reduction == "sum":
                return torch.sum(loos_ce + loss_early_detection), i_prob
            else:
                return loos_ce + loss_early_detection, i_prob
        else:
            return loos_ce, i_prob
