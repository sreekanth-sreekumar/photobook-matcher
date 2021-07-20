import csv

class Vocab:
    def __init__(self, file=None):
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        if file:
            self.initialize_vocabulary(file)

    def initialize_vocabulary(self, file):
        for t in ['<pad>', '<unk>', '<sos>', '<eos>']:
            self.word2index[t] = len(self.index2word)
            self.index2word[len(self.index2word)] = t

        with open(file, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=',', quotechar='|')
            for row in reader:
                w, c = row[0], int(row[1])
                self.word2index[w] = len(self.index2word)
                self.index2word[len(self.index2word)] = w
                self.word2count[w] = c

    def __len__(self):
        return len(self.word2index)

    def __getitem__(self, q):
        if isinstance(q, str):
            return self.word2index.get(q, self.word2index['<unk>'])
        elif isinstance(q, int):
            return self.index2word.get(q, '<unk>')
        else:
            raise ValueError("Expected str or int but got {}".format(type(q)))