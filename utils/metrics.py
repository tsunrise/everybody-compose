import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_pred.shape == y_true.shape
    return (y_pred == y_true).sum()/ len(y_true)

class Metrics:
    def __init__(self, label: str):
        self.label = label
        self.metrics_sum = {
            "loss": 0.,
            "accuracy": 0.
        }
        self.sample_count = 0

    def update(self, batch_size: int, loss: float, y_pred_one_hot: torch.Tensor, y_true: torch.Tensor):
        self.metrics_sum["loss"] += loss * batch_size

        y_pred = torch.argmax(y_pred_one_hot, dim=2).cpu().numpy().flatten()
        y_true = y_true.cpu().numpy().astype(float).flatten()

        self.metrics_sum["accuracy"] += accuracy(y_pred, y_true) * batch_size
        self.sample_count += batch_size

    def flush_and_reset(self, writer: SummaryWriter, global_step: int):
        for metric, value in self.metrics_sum.items():
            writer.add_scalar(f"{self.label}/{metric}", value / self.sample_count, global_step)
        summary = {metric: value / self.sample_count for metric, value in self.metrics_sum.items()}
        self.metrics_sum = {metric: 0 for metric in self.metrics_sum}
        self.sample_count = 0
        return summary
        
