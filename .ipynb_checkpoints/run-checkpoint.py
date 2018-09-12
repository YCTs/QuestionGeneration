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
parser.add_argument("--prepare",
					help="prepare something we need",
					action="store_true",
					)
parser.add_argument("--testing",
					help="testing mode",
					action="store_true",
					)
parser.add_argument("--training",
					help="train mode",
					action="store_true",
					)

args = parser.parse_args()

class EncoderRNN_Document(nn.Module):
    def __init__(self, hidden_size, embedding_weight):
        super(EncoderRNN_Document, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        self.embedding_dim = embedding_weight.size(1)
        
        self.bi_gru = nn.GRU(self.embedding_dim, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, 1, self.embedding_dim) # 
        
        output = embedded
        output, hidden = self.bi_gru(output, hidden) ##hidden(1*num_dir, batch, h_dim), output(batch, 1, 2*h_dim)

        return output, hidden

class EncoderRNN_Answer(nn.Module):
    def __init__(self, hidden_size, embedding_weight):
        super(EncoderRNN_Answer, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        self.embedding_dim = embedding_weight.size(1)

        self.bi_gru = nn.GRU(self.embedding_dim + 2*self.hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input, annotation_seq, hidden):
        # annotaion_seq = (-1, 1, 2*hidden_size)
        embedded = self.embedding(input).view(-1, 1, self.embedding_dim) # 
        
        output = torch.cat((embedded, annotation_seq.unsqueeze(1)), 2)
        
        output, hidden = self.bi_gru(output, hidden) ##hidden(1*num_dir, batch, h_dim), output(batch, 1, 2*h_dim)

        return output, hidden
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_weight, dropout_p=0.1, len_q=20, len_c=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size ## output dict size
        self.dropout_p = dropout_p
        self.max_length = len_q
        self.embedding_dim = embedding_weight.size(1)
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        self.attn = nn.Linear(self.hidden_size*(len_c + 2) + self.embedding_dim, len_c)
        self.attn_combine = nn.Linear(self.hidden_size +self.embedding_dim, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size*2 +self.embedding_dim, self.hidden_size, batch_first=True)
        self.mlp = nn.Linear(3*self.hidden_size + self.embedding_dim, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.pointer_weight = nn.Parameter(torch.zeros((1), requires_grad=True))
        self.L = nn.Parameter(torch.zeros((self.hidden_size, self.hidden_size), requires_grad=True))
        self.Wb_0 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, annot_vector, answer_encoding):
        embedded = self.embedding(input).view(-1, 1, self.embedding_dim) # -1, 1, embed_dim
        embedded = self.dropout(embedded)        
                
        #print(embedded.size())         #-1, 1, embed_dim
        #print(annot_vector.size())     #-1, len_c, hidden_size
        #print(answer_encoding.size())  #-1, 1, hidden_size
        len_c = annot_vector.size(1)
        
        in_attn = torch.cat((hidden.squeeze(0), embedded.squeeze(1), answer_encoding.squeeze(1), annot_vector.view(-1, len_c*self.hidden_size)), 1) # -1, 13156
        
    
        tanh = nn.Tanh()
        attn_weights = F.softmax(tanh(self.attn(in_attn)), dim=1) # -1, len_c
        
        c_t = torch.bmm(attn_weights.unsqueeze(1), annot_vector) # -1, 1, hidden_size
        
        v_t = torch.cat((c_t, answer_encoding), 2) # -1, 1, 2*hidden_size
        
        gru_input = torch.cat((v_t, embedded), 2)
        
        output, hidden = self.gru(gru_input, hidden)
        
        input_mlp = torch.cat((output, v_t, embedded), 2).squeeze(1)

        e_t = self.mlp(input_mlp)
        o_t = F.softmax(self.out(e_t), dim=1) # -1, output_size
        z_t = torch.sigmoid(self.pointer_weight)
        #print(z_t.item())
        p_t = torch.cat((o_t*z_t, attn_weights*(1-z_t)), 1)
        p_t = torch.log(p_t+1e-20)
        return output, hidden, p_t
    def inintHidden(self, h_a, h_d):

        r = torch.bmm(h_a, self.L.unsqueeze(0)) + torch.sum(h_d.view(-1, h_d.size(1), h_d.size(2)))/h_d.size(1)
        
        return self.Wb_0(r.view(-1, 1, self.hidden_size)).view(1, -1, self.hidden_size)
        

teacher_forcing_ratio = 0.5

def train(input_d, d_words, input_a, answer_pointer, target, q_words, encoder_d, encoder_a, decoder, encoder_d_optimizer, encoder_a_optimizer, decoder_optimizer,len_d, len_a, len_q): #return loss
    
    batch_size = 1
    
    encoder_d_optimizer.zero_grad()
    encoder_a_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    annotation_vector = torch.zeros(1, len_d, 2*64, device=device)
    len_d_actual = input_d.size(1)
    h_d = None
    for t_d in range(len_d_actual):
        w = input_d[:, t_d].view(1, 1)
        o, h = encoder_d.forward(w, h_d)
        annotation_vector[:, t_d, :] = o[:, 0, :]
    
    h_a = None
    answer_encoding = None
    len_a = input_a.size(1)
    annotation_seq = torch.zeros(1, len_a, 2*64, device=device) 
    for j in range(batch_size):
        actual_len =  answer_pointer[j][1] - answer_pointer[j][0] + 1
        annotation_seq[j, 0:actual_len, :] = annotation_vector[j, answer_pointer[j][0]:answer_pointer[j][1]+1,:]
    for t_a in range(len_a):
        input = input_a[:, t_a].view(1, 1)
        answer_encoding, h_a = encoder_a.forward(input, annotation_seq[:, t_a, :], h_a)
        
    #s = answer_encoding.view(1, -1, 128)
    s = decoder.inintHidden(answer_encoding, annotation_vector)
    SOS = [1]
    decoder_input = torch.tensor(SOS, dtype=torch.long, device=device).view(1, 1) # SOS token  
    loss = 0
    using_teacher_forcing = True
    
    if using_teacher_forcing:

        len_q = target.size(1)

        for t_q in range(len_q):

            o, s, p_t = decoder.forward(decoder_input,s,annotation_vector, answer_encoding)
            decoder_input = target[:, t_q].view(1, 1)
            y_t = target[:, t_q].view(1,)
            if y_t.item() == 3:
                
                word = q_words[t_q]
                for j in range(len(d_words)):
                    if word == d_words[j]:
                        y_t = torch.tensor([j+2004], device=device)
                        break
            
            loss+=nn.NLLLoss()(p_t, y_t)
            
            
    else:
        len_q = target.size(1)
        
        for t_q in range(len_q):

            o, s, p_t = decoder.forward(decoder_input,s,annotation_vector, answer_encoding)

            decoder_input = target[:, t_q].view(1, 1)
    
    loss.backward()
    encoder_d_optimizer.step()
    encoder_a_optimizer.step()
    decoder_optimizer.step()

            
            
    return loss.item()/len_q


def main():
    if args.training:
        
        document, question, answer, answer_pointer = data_reduction() #100, 20, 20
        
        d = dynamic_id_sentence(document)
        a = dynamic_id_sentence(answer)
        q = dynamic_id_sentence(question, shortlist= True)


        weight = np.load("weight.npy")
        weight = torch.FloatTensor(weight) 
        weight_shortlist = np.load("weight_shortlist.npy")
        weight_shortlist = torch.FloatTensor(weight_shortlist)
        
        encoder_d = EncoderRNN_Document(64, weight).to(device)
        encoder_a = EncoderRNN_Answer(64, weight).to(device)
        decoder = AttnDecoderRNN(128, weight_shortlist.size(0), weight_shortlist, 0.1, 20, 100).to(device)
        
        encoder_d_optimizer = optim.Adam(encoder_d.parameters(), lr=0.0001)
        encoder_a_optimizer = optim.Adam(encoder_a.parameters(), lr=0.0001)        
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)

        for i in range(1000):
            d_in = torch.tensor(d[i], dtype=torch.long, device=device).view(1, -1) # batch, len_d
            a_in = torch.tensor(a[i], dtype=torch.long, device=device).view(1, -1)
            q_in = torch.tensor(q[i], dtype=torch.long, device=device).view(1, -1)
            a_p = [answer_pointer[i]]        


            loss = train(d_in, document[i], a_in, a_p, q_in, question[i], encoder_d, encoder_a, decoder,
                         encoder_d_optimizer, encoder_a_optimizer, decoder_optimizer, 100, 20, 20)
            if i%10 == 0:
                print("i =", i, " ,loss=", loss)
        
        
        #print(d_in.size())
        
        
        #loss = train()

    if args.testing:    
        
        exit(0)
    
    
    if args.prepare:
        #embedding_weight()
        shortlist()
        exit(0)


if __name__ == '__main__':
    main()
    


