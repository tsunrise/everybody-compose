from abc import ABC, abstractmethod
from typing import Tuple
import torch
from models.lstm import DeepBeatsLSTM
from models.lstm_local_attn import DeepBeatsLSTMLocalAttn
from models.transformer import DeepBeatsTransformer

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

class LocalAttnLSTMDistribution(DistributionGenerator):
    def __init__(self, model: DeepBeatsLSTMLocalAttn, x, device):
        """
        - `x` is the input sequence, shape: (seq_len, 2)
        """
        self.model = model
        self.device = device
        self.x = x
        self.context, self.encoder_state = self.model.encoder(x)
        self.starter = [60, 62, 64] # TODO: apply the same trick to all cases

    def initial_state(self) -> dict:
        super().initial_state()
        
        return {
            "position": 0,
            "memory": (self.encoder_state[0].reshape(1, 1, -1), self.encoder_state[1].reshape(1, 1, -1)),
        }

    def proceed(self, state: dict, prev_note: int, sampled_sequence: torch.Tensor) -> Tuple[dict, torch.Tensor]:
        super().proceed(state, prev_note, sampled_sequence)
        position = state["position"]
        memory = state["memory"]
        context_curr = self.context[position].reshape(1, 1, -1)
        if position < len(self.starter):
            y_prev = self.starter[position]
        y_prev = torch.tensor(prev_note).reshape(1, 1).to(self.device)
        scores, memory = self.model.decoder.forward(y_prev, context_curr, memory)
        if position < len(self.starter):
            scores = torch.zeros(128)
            scores[self.starter[position]] = 1
        else:
            scores = scores.squeeze(0)
            scores = torch.nn.functional.softmax(scores, dim=1)
            scores = scores.squeeze(0)
        return {"position": position + 1, "memory": memory}, scores


class TransformerDistribution(DistributionGenerator):

    def __init__(self, model: DeepBeatsTransformer, x, device):
        x = x.unsqueeze(1) # (seq_len, 1, 2)
        self.model = model.to(device)
        self.device = device
        self.x_mask = (torch.zeros(x.shape[0], x.shape[0])).type(torch.bool).to(device)
        self.x = x.to(device)
        self.memory = self.model.encode(self.x, self.x_mask).to(device)
    
    def initial_state(self) -> dict:
        super().initial_state()
        return {
            "ys": torch.ones(1, 1).fill_(0).type(torch.long).to(self.device)
        }
    
    def proceed(self, state: dict, prev_note: int, sampled_sequence: torch.Tensor) -> Tuple[dict, torch.Tensor]:
        super().proceed(state, prev_note, sampled_sequence)
        ys = state["ys"]
        tgt_mask = (self.model.generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(self.device)
        out = self.model.decode(ys, self.memory, tgt_mask)
        out = out.transpose(0, 1)
        scores = self.model.generator(out[:, -1]) # 1 * num_notes, we only care about the last one
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.transpose(0, 1).squeeze(1)
        return {"ys": torch.cat([ys, torch.ones(1, 1).type_as(self.x.data).fill_(prev_note)], dim=0)}, scores
