"""Download and pre-process GloVe.
"""

import numpy as np
import torch

from codecs import open


class Embedding():
    def __init__(self, dim=200, vocab=None):
        """
        Initialize the Embedding class.

        Arguments
        ---------
        url: str
            Url to download the Embeddings file from

        vocab: Set[str]
            Set of tokens specifying the subset of embeddings to keep in memory

        Supports reading from glove-like file format and keeps a subset of embeddings in memory for fast-access
        """

        if vocab is not None:
            self.vocab = set(vocab)
        else:
            self.vocab = None
        self.emb_file = '/home/gaurangajitk/DL/data/image-caption-data/glove.6B.200d.txt'
        self.OOV = "--OOV--"
        self.dim = dim
        self.load()

    def load(self):
        """Load a subset (self.vocab) of embeddings into memory"""
        print("Pre-processing {} vectors...".format(self.emb_file))

        embedding_dict = {}
        with open(self.emb_file, "r", "utf-8") as fh:
            for line in fh:
                array = line.split()
                word = array[0]
                vector = torch.FloatTensor(list(map(float, array[1:])))
                if self.vocab is not None:
                    if word in self.vocab:
                        embedding_dict[word] = vector
                else:
                    embedding_dict[word] = vector
            if self.vocab is not None:
                print("{} / {} tokens have corresponding embedding vector".format(len(embedding_dict), len(self.vocab)))
            else:
                print("{} tokens have corresponding embedding vector".format(len(embedding_dict)))
        embedding_dict[self.OOV] = torch.FloatTensor(np.random.uniform(low=-1, high=1, size=self.dim))

        self.embeddings = embedding_dict

    def __getitem__(self, token):
        if token in self.embeddings:
            return self.embeddings[token]
        else:
            return self.embeddings[self.OOV]

def get_embedding_matrix(vocab: list):
    embedding_mat = []
    embedding = Embedding(vocab=vocab)
    for word in vocab:
        embedding_mat.append(embedding[word])
    return torch.vstack(embedding_mat)