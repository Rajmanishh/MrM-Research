import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class TextDataset(Dataset):
    def __init__(self, split="train", max_len=200):
        super().__init__()

        self.max_len = max_len

        # 🔥 Load DBpedia dataset
        dataset = load_dataset("dbpedia_14")
        self.data = dataset[split]

    def encode(self, text):
        text = text.lower()

        # 🔥 UTF-8 byte encoding
        byte_ids = list(text.encode("utf-8"))

        # 🔥 Padding / truncation
        if len(byte_ids) < self.max_len:
            byte_ids += [0] * (self.max_len - len(byte_ids))
        else:
            byte_ids = byte_ids[:self.max_len]

        return torch.tensor(byte_ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["content"]
        label = item["label"]

        x = self.encode(text)
        y = label

        return x, y