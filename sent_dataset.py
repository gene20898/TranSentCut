from torch.utils.data import Dataset
import torch

class SentDataset(Dataset):
    def __init__(self, data, label, tokenizer, context_length) -> None:
        self.tokenizer = tokenizer
        self.data = data
        self.label = label
        self.inputs = []
        for i, example in enumerate(self.data): 
            tok_result = self.tokenizer(
                example[0], # A sequence
                example[1], # B sequence
                max_length=context_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            tok_result['input_ids'] = torch.squeeze(tok_result['input_ids'])
            tok_result['attention_mask'] = torch.squeeze(tok_result['attention_mask'])
            tok_result['labels'] = self.label[i]
            self.inputs.append(tok_result)

    def __len__(self) -> int:   
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index]
