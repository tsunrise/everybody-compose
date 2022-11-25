from abc import ABC, abstractmethod
from typing import Tuple
import torch
from models.lstm import DeepBeatsLSTM

class DistributionGenerator(ABC):
    @abstractmethod
    def __init__(self):
        """
        Implementor uses this method to store some constants during sampling process.
        """
        pass

    @abstractmethod
    def initial_state(self) -> dict:
        pass

    @abstractmethod
    def proceed(self, state: dict, prev_note: int, sampled_sequence: torch.Tensor) -> Tuple[dict, torch.Tensor]:
        """
        - `state`: a dictionary containing the state of the machine
        - `sampled_sequence`: a tensor of shape (seq_len, ), containing the sampled sequence
        Returns:
        - `state`: a dictionary containing the updated state of the machine
        - `distribution`: a tensor of shape (n_notes, ), containing the distribution of the next note
        """
        pass

class LSTMDistribution(DistributionGenerator):
    def __init__(self, model: DeepBeatsLSTM, x, device):
        """
        - `x` is the input sequence, shape: (seq_len, 2)
        """
        self.model = model
        self.device = device
        self.x = x

    def initial_state(self) -> dict:
        super().initial_state()
        return {
            "position": 0,
            "hidden": self.model._default_init_hidden(1),
        }

    def proceed(self, state: dict, prev_note: int, sampled_sequence: torch.Tensor) -> Tuple[dict, torch.Tensor]:
        super().proceed(state, prev_note, sampled_sequence)
        position = state["position"]
        hidden = state["hidden"]
        x_curr = self.x[position].reshape(1, 1, 2)
        y_prev = torch.tensor(prev_note).reshape(1, 1).to(self.device)
        scores, hidden = self.model.forward(x_curr, y_prev, hidden)
        scores = scores.squeeze(0)
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.squeeze(0)
        return {"position": position + 1, "hidden": hidden}, scores