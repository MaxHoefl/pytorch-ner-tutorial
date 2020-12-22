"""
data_loader.py

In this module, the following tasks are performed

- creating vocab dictionary
- creating tag dictionary

"""
import os
import random
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")

class DataLoader(object):
    def __init__(self):
        # load vocab indices
        vocab_path = os.path.join(DATA_DIR, 'words.txt')
        self.vocab = {}
        with open(vocab_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab[l] = i

        # setting indices for UNKown words and PADding symbols
        self.unk_ind = self.vocab["UNK"]
        self.pad_ind = self.vocab["PAD"]

        # loading tags
        tags_path = os.path.join(DATA_DIR, "tags.txt")
        self.tag_map = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag_map[t] = i

    def load_data(self, split="train"):
        data = {}
        sentence_file = os.path.join(DATA_DIR, split, "sentences.txt")
        labels_file = os.path.join(DATA_DIR, split, "labels.txt")
        data[split] = self.load_sentences_labels(
            sentence_file, 
            labels_file
        )
        return data

    def load_sentences_labels(self, sentence_file, labels_file):
        sentences = []
        labels = []
        with open(sentence_file) as f:
            for sentence in f.read().splitlines():
                # replace each token by its index if its in vocab
                # else use index of UNK token
                s = [self.vocab[token] if token in self.vocab
                        else self.unk_ind
                        for token in sentence.split()]
                sentences.append(s)

        with open(labels_file) as f:
            for sen_labels in f.read().splitlines():
                # replace each label by its index
                l = [self.tag_map[label] for label in sen_labels.split()]
                labels.append(l)

        res = {}
        assert len(sentences) == len(labels)
        for i in range(len(labels)):
            assert len(labels[i]) == len(sentences[i])
        res['data'] = sentences
        res['labels'] = labels
        res['size'] = len(sentences)
        return res


    def data_iterator(self, data, batch_size, shuffle=False, cuda=False):
        """
        Iterate over batches from `data`. 
        """
        indices = list(range(data['size']))
        if shuffle:
            random.seed(1992)
            random.shuffle(indices)
       
        num_batches = len(indices)//batch_size
        for batch_i in range(num_batches):
            start_idx = batch_i * batch_size
            if batch_i == num_batches - 1:
                end_idx = len(indices)
            else:
                end_idx = (batch_i + 1) * batch_size
            batch_sentences = [data['data'][idx] \
                for idx in indices[start_idx:end_idx]]
            batch_tags = [data['labels'][idx] \
                for idx in indices[start_idx:end_idx]]
            
            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in batch_sentences])
            # note that we only have to pad sentences within a batch
            # as pytorch's computational graph is dynamic.
            # on every training iteration the graph is rebuilt
            # i.e. variables within the graph only have a scope of the current
            # training iteration

            # prepare a numpy array with the data
            # the data array will have dimensions N x M
            # where N = # sentences in batch
            #       M = length of longest sentence in batch
            # initialise the data with PAD token index
            # the corresponding label index for the PAD token is -1
            batch_data = self.pad_ind * \
                np.ones((len(batch_sentences), batch_max_len)) 
            batch_labels = -1 * np.ones((len(batch_sentences), batch_max_len))
            for i, (sen, tags) in enumerate(zip(batch_sentences, batch_tags)):
                batch_data[i, :len(sen)] = sen
                batch_labels[i, :len(tags)] = tags

            # all data are indices, so conert to torch LongTensors
            batch_data = torch.LongTensor(batch_data)
            batch_labels = torch.LongTensor(batch_labels)

            # shift all tensors to GPU if available
            if cuda:
                batch_data = batch_data.cuda()
                batch_labels = batch_labels.cuda()

            # convert tensors to variables to record operations in 
            # the computational graph
            batch_data = Variable(batch_data)
            batch_labels = Variable(batch_labels)

            yield batch_data, batch_labels


if __name__ == '__main__':
    data_loader = DataLoader()
    train_data = data_loader.load_data(split="train")
    val_data = data_loader.load_data(split="val")
    print(train_data['train']['size'])
    print(val_data['val']['size'])
    print(train_data['train']['data'][-1])
    print(train_data['train']['labels'][-1])

    data_iter = data_loader.data_iterator(
            data=train_data['train'],
            batch_size=32
    )

    batch_data, batch_labels = next(data_iter)
    print(batch_data)
    print(batch_labels)

