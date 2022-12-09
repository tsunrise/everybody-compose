from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
from models.lstm_local_attn import DeepBeatsLSTMLocalAttn
from models.attention_rnn import DeepBeatsAttentionRNN
from models.transformer import DeepBeatsTransformer
from models.vanilla_rnn import DeepBeatsVanillaRNN

from utils.constants import NOTE_START


class DistributionGenerator(ABC):
    @abstractmethod
    def __init__(self):
        """
        Implementor uses this method to store some constants during sampling process.
        """
        pass

    @abstractmethod
    def initial_state(self, hint: List[int]) -> dict:
        """
        Get initial state of the model for sampling, given that the initial sequence is `hint`.
        Return the state after hint.
        """
        pass

    @abstractmethod
    def proceed(self, state: dict, prev_note: int) -> Tuple[dict, torch.Tensor]:
        """
        - `state`: a dictionary containing the state of the machine
        - `sampled_sequence`: a tensor of shape (seq_len, ), containing the sampled sequence
        Returns:
        - `state`: a dictionary containing the updated state of the machine
        - `distribution`: a tensor of shape (n_notes, ), containing the distribution of the next note
        """
        pass



class LocalAttnLSTMDistribution(DistributionGenerator):
    def __init__(self, model: DeepBeatsLSTMLocalAttn, x, device):
        """
        - `x` is the input sequence, shape: (seq_len, 2)
        """
        self.model = model
        self.device = device
        self.x = x
        self.context, self.encoder_state = self.model.encoder(x)

    def initial_state(self, hint: List[int]) -> dict:
        super().initial_state(hint)
        hint_shifted = [NOTE_START] + hint[:-1]
        state = {
            "position": 0,
            "memory": (self.encoder_state[0].reshape(1, 1, -1), self.encoder_state[1].reshape(1, 1, -1)),
        }
        for i in range(len(hint)):
            state, _ = self.proceed(state, hint_shifted[i])
        return state

    def proceed(self, state: dict, prev_note: int) -> Tuple[dict, torch.Tensor]:
        super().proceed(state, prev_note)
        position = state["position"]
        memory = state["memory"]
        context_curr = self.context[position].reshape(1, 1, -1)

        y_prev = torch.tensor(prev_note).reshape(1, 1).to(self.device)
        scores, memory = self.model.decoder.forward(y_prev, context_curr, memory)

        scores = scores.squeeze(0)
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.squeeze(0)
        return {"position": position + 1, "memory": memory}, scores

class AttentionRNNDistribution(DistributionGenerator):
    def __init__(self, model: DeepBeatsAttentionRNN, x, device):
        """
        - `x` is the input sequence, shape: (seq_len, 2)
        """
        self.model = model
        self.device = device
        self.x = x
        self.encoder_output, _ = self.model.encoder(x)
        self.encoder_output = self.encoder_output.unsqueeze(0)

    def initial_state(self, hint: List[int]) -> dict:
        super().initial_state(hint)
        state = {
            "position": 0,
            "memory": None,
        }
        hint_shifted = [NOTE_START] + hint[:-1]
        for i in range(len(hint)):
            state, _ = self.proceed(state, hint_shifted[i])
        return state

    def proceed(self, state: dict, prev_note: int) -> Tuple[dict, torch.Tensor]:
        super().proceed(state, prev_note)
        position = state["position"]
        memory = state["memory"]
        y_prev = torch.tensor(prev_note).reshape(1, 1).to(self.device)
        scores, memory = self.model.decoder.forward(y_prev, self.encoder_output, memory)
        scores = scores.squeeze(0)
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.squeeze(0)
        return {"position": position + 1, "memory": memory}, scores

class TransformerDistribution(DistributionGenerator):

    def __init__(self, model: DeepBeatsTransformer, x, device):
        x = x.unsqueeze(1)  # (seq_len, 1, 2)
        self.model = model.to(device)
        self.device = device
        self.x_mask = (torch.zeros(x.shape[0], x.shape[0])).type(torch.bool).to(device)
        self.x = x.to(device)
        self.memory = self.model.encode(self.x, self.x_mask).to(device)
        self.max_seq = x.shape[0]

    def initial_state(self, hint: List[int]) -> dict:
        super().initial_state(hint)
        hint_shifted = [NOTE_START] + hint[:-1]
        ys = torch.tensor(hint_shifted).reshape(1, -1).permute(1, 0).to(self.device)
        return {
            "ys": ys,
        }

    def proceed(self, state: dict, prev_note: int) -> Tuple[dict, torch.Tensor]:
        super().proceed(state, prev_note)
        ys = state["ys"]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(self.x.data).fill_(prev_note)], dim=0)
        curr_i = ys.shape[0]
        filled_ys = torch.cat([ys, torch.ones(self.max_seq - curr_i, 1).type_as(self.x.data).fill_(0)]) # fill max_seq
        tgt_mask = (self.model.generate_square_subsequent_mask(filled_ys.shape[0])
                    .type(torch.bool)).to(self.device)
        out = self.model.decode(filled_ys, self.memory, tgt_mask) # max_seq * 1 * 128
        out = out.transpose(0, 1)  # 1 * max_seq * 128
        scores = self.model.generator(out[:, curr_i - 1])  # 1 * num_notes, we only care about the current one
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.transpose(0, 1).squeeze(1)
        return {"ys": ys}, scores

class VanillaRNNDistribution(DistributionGenerator):
    def __init__(self, model: DeepBeatsVanillaRNN, x, device):
        """
        - `x` is the input sequence, shape: (seq_len, 2)
        """
        self.model = model
        self.device = device
        self.x = x

    def initial_state(self, hint: List[int]) -> dict:
        super().initial_state(hint)
        state = {
            "position": 0,
            "hidden": self.model._default_init_hidden(1),
        }
        hint_shifted = [NOTE_START] + hint[:-1]
        for i in range(len(hint)):
            state, _ = self.proceed(state, hint_shifted[i])
        return state

    def proceed(self, state: dict, prev_note: int) -> Tuple[dict, torch.Tensor]:
        super().proceed(state, prev_note)
        position = state["position"]
        hidden = state["hidden"]
        x_curr = self.x[position].reshape(1, 1, 2)
        y_prev = torch.tensor(prev_note).reshape(1, 1).to(self.device)
        scores, hidden = self.model.forward(x_curr, y_prev, hidden)
        scores = scores.squeeze(0)
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.squeeze(0)
        return {"position": position + 1, "hidden": hidden}, scores
