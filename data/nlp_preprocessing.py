import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        vocab = []

        for sentence in positive + negative:
            sentence_lst = sentence.split(' ')
            for word in sentence_lst:
                if word not in vocab:
                    vocab.append(word)

        vocab = sorted(vocab)
        vocab_dict = {}
        iD = 1.0
        for word in vocab:
            vocab_dict[word] = iD
            iD += 1.0

        enc = []
        for sentence in positive + negative:
            sentence_lst = sentence.split(' ')
            sent_enc = []
            for word in sentence_lst:
                sent_enc.append(vocab_dict[word])
            enc.append(torch.tensor(sent_enc))

        return torch.nn.utils.rnn.pad_sequence(enc, batch_first=True)
        
