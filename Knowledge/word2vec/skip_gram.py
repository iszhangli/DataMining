from torch import nn
import torch
from vocabs import getvocabsOnlyIndex

class Skip_Gram(nn.Module):

    def __init__(self,vocabs,vector_size):
        super().__init__()
        self.vocabs = torch.LongTensor(vocabs)
        vocab_numbers = len(vocabs)
        self.word_embs = nn.Embedding(vocab_numbers,vector_size)  # number_embedding, embedding_dim
        self.bkp_word_embs = nn.Embedding(vocab_numbers,vector_size)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.word_embs(x)
        bkp = self.bkp_word_embs(self.vocabs)
        y = torch.matmul(x,bkp.T)
        y = self.softmax(y)
        return y

def word2vec( seqs, window_size = 1 ):
    vocabs = getvocabsOnlyIndex(seqs)  #
    net = Skip_Gram(vocabs,vector_size=16)
    criterion = torch.nn.CrossEntropyLoss()  #
    optimizer = torch.optim.SGD( net.parameters(), lr=0.01)
    net.train()
    for seq in seqs:
        ns  = len(seq)-(window_size*2)
        for i in range(0, ns):
            optimizer.zero_grad()
            window = seq[i:i+1+window_size*2]
            # [window*2]  [window[window_size] for _ in range(window_size*2)]
            x = torch.LongTensor([window[window_size] for _ in range(window_size*2)])
            y_pred = net(x)
            window.pop(window_size)
            # [window*2]
            y =  torch.LongTensor(window)
            loss = criterion(y_pred, y)
            print()
            print(loss)
            for name, parms in net.named_parameters():
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:', parms.requires_grad)
                print('-->grad_value:', parms.grad)

            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    s1 = [8,9,12,14,6]
    s2 = [4,11,13,3,6]
    s3 = [4,8,2,15,9]
    s4 = [1,7,5,10,0]
    seqs = [s1,s2,s3,s4]
    word2vec(seqs)

# embedding update dim and param every time