import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data_processing import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

parser = argparse.ArgumentParser()
#group = parser.add_mutually_exclusive_group()
parser.add_argument("-e","--embedding",
					help="prepare the weight of word embedding",
					action="store_true",
					)

args = parser.parse_args()

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_weight):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.embedding_dim = embedding_weight.size(1)
        self.gru = nn.GRU(self.embedding_dim, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, 1, self.embedding_dim) # 
        
        output = embedded
        output, hidden = self.gru(output, hidden) ##hidden(len, batch, dim), output(batch, len, dim)

        return output, hidden
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_weight, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size ## output dict size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding_dim = embedding_weight.size(1)
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.attn = nn.Linear(self.hidden_size + self.embedding_dim, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size +self.embedding_dim, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(-1, 1, self.embedding_dim) # -1, 1, embed_dim
        embedded = self.dropout(embedded)
        # hidden : (1, -1, dim )-> (-1, 1, h_dim)
        #print(torch.cat((embedded, hidden.view(-1, 1, self.hidden_size)), 2).size()) : (3, 1, 150)
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden.view(-1, 1, self.hidden_size)), 2).squeeze(1)), 
            dim=1)
        #print(attn_weights.size()) # (-1, len)

        #print(encoder_outputs.size()) # (-1, len, h_dim)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        #print(attn_applied.size()) (-1, 1, h_dim)

        
        output = torch.cat((embedded, attn_applied), 2)
    
        output = self.attn_combine(output.squeeze(1)).unsqueeze(1)

        #print(output.size()) (-1, 1, h_dim)

        
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        
        output = F.log_softmax(self.out(output.squeeze(1)), dim=1)

        print(output.size())

        return output, hidden
        '''
        return output, hidden, attn_weights
        '''





def main():

    if args.embedding:
        embedding_weight()
        exit(0)
    weight = np.load("weight.npy")

    weight = torch.FloatTensor(weight)

    encoder = EncoderRNN(50, weight)
    x = torch.LongTensor([[1, 2], [1, 2], [1, 2]])
    seq_len = x.size(1)
    batch_size=3
    hidden = None
    encoder_outputs=torch.zeros(batch_size, seq_len, encoder.hidden_size, device=device)
    for time in range(seq_len):
        x_in = x[:, time]
        o, hidden = encoder.forward(x_in, hidden)
        #print(o.size(), hidden.size())
        #print(encoder_outputs.size())
        encoder_outputs[:, time, :] = o[:,0,:]
    #print(encoder_outputs.size())


    max_length = seq_len
    decoder_dict_size = 5
    decoder = AttnDecoderRNN(50, decoder_dict_size, weight, 0.1, max_length)
    hidden_decoder = hidden
    for time in range(max_length):
        x_in = x[: ,time]
        o_decoder, h_decoder = decoder.forward(x_in, hidden_decoder, encoder_outputs)



if __name__ == '__main__':
    main()
    


