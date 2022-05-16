import torch
from transformers import AutoTokenizer

class Mapping:
    def __init__(self, pretrained_language_model_name, stoi: dict):
        self.token_encoder = AutoTokenizer.from_pretrained(pretrained_language_model_name)

        self.stoi = stoi

        # create mapping
        self.mapping = {}
        for token, idx in self.stoi.values():
            ids = self.token_encoder.tokenize(token)[1:-1] # ignore the <s> and </s> token
            self.mapping[idx] = ids

    def map(self, input_tensor: torch.Tensor):
        inputs = input_tensor.tolist()
        mapped_inputs = []
        max_len = 0
        for input in inputs:
            mapped_input = []
            for idx in input:
                mapped_input += self.mapping[idx]
            if max_len < len(mapped_input):
                max_len = len(mapped_input)
            mapped_inputs.append(mapped_input)

        for ith in range(len(mapped_inputs)):
            mapped_input = mapped_inputs[ith]
            while len(mapped_input) < max_len:
                mapped_input.append(self.padding_token)
            mapped_inputs[ith] = mapped_input

        return torch.tensor(mapped_inputs, device=input_tensor.device)