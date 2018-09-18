import numpy as np
import random
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

embedding_freeze = True

class EncoderRNN_Document(nn.Module):
    def __init__(self, hidden_size, embedding_weight):
        super(EncoderRNN_Document, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=embedding_freeze)
        self.embedding_dim = embedding_weight.size(1)+1
        self.dropout = nn.Dropout(0.2)
        self.bi_gru = nn.GRU(self.embedding_dim, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input, hidden, is_in_a):
        embedded = self.embedding(input).view(-1, 1, self.embedding_dim-1) #
        embedded = self.dropout(embedded)
        b_feature = torch.tensor(is_in_a, dtype=torch.float, device=device).view(-1, 1, 1)
        output = torch.cat((embedded, b_feature), 2)
        output, hidden = self.bi_gru(output, hidden) ##hidden(1*num_dir, batch, h_dim), output(batch, 1, 2*h_dim)
        output = self.dropout(output)
        hidden = self.dropout(hidden)

        return output, hidden

class EncoderRNN_Answer(nn.Module):
    def __init__(self, hidden_size, embedding_weight):
        super(EncoderRNN_Answer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.2)
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=embedding_freeze)
        self.embedding_dim = embedding_weight.size(1)

        self.bi_gru = nn.GRU(self.embedding_dim + 2*self.hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input, annotation_seq, hidden):
        # annotaion_seq = (-1, 1, 2*hidden_size)
        embedded = self.embedding(input).view(-1, 1, self.embedding_dim) # 
        embedded = self.dropout(embedded)
        output = torch.cat((embedded, annotation_seq.unsqueeze(1)), 2)
        output, hidden = self.bi_gru(output, hidden) ##hidden(1*num_dir, batch, h_dim), output(batch, 1, 2*h_dim)
        output = self.dropout(output)
        hidden = self.dropout(hidden)

        return output, hidden
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_weight, embedding_weight_d, dropout_p=0.3, len_q=20, len_c=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size ## output dict size
        self.dropout_p = dropout_p
        self.max_length = len_q
        self.embedding_dim = embedding_weight.size(1)
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=embedding_freeze)
        self.embedding_d = nn.Embedding.from_pretrained(embedding_weight_d, freeze=embedding_freeze)
        self.attn = nn.Linear(self.hidden_size*(len_c + 2) + self.embedding_dim, len_c)
        self.attn_combine = nn.Linear(self.hidden_size +self.embedding_dim, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size*2 +self.embedding_dim, self.hidden_size, batch_first=True)
        self.mlp = nn.Linear(3*self.hidden_size + self.embedding_dim, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        #self.pointer_weight = nn.Parameter(torch.zeros((1), requires_grad=True))
        self.MLP_zt = nn.Sequential(
            nn.Linear(3*self.hidden_size + self.embedding_dim, self.hidden_size*2),
            #nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
            #nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        self.L = nn.Parameter(torch.zeros((self.hidden_size, self.hidden_size), requires_grad=True))
        self.Wb_0 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, annot_vector, answer_encoding, input_d):
        batch_size = input.size(0)
        embedded = torch.zeros(batch_size, 1, self.embedding_dim, device=device)
        input_d = input_d.view(batch_size, -1)
        for i in range(batch_size):
            if input[i].item() >2003:
                i_d = input[i].item() - 2004
                embedded[i] = self.embedding_d(input_d[i][i_d]).view(-1, 1, self.embedding_dim)
            else:
                embedded[i] = self.embedding(input[i]).view(-1, 1, self.embedding_dim) # -1, 1, embed_dim
                
        embedded = self.dropout(embedded)
        hidden = self.dropout(hidden)
                
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
        gru_input = self.dropout(gru_input)
        
        output, hidden = self.gru(gru_input, hidden)

        input_mlp = torch.cat((output, v_t, embedded), 2).squeeze(1)
        #input_mlp = self.dropout(input_mlp)
        e_t = self.mlp(input_mlp)
        e_t = self.dropout(e_t)
        o_t = F.softmax(self.out(e_t), dim=1) # -1, output_size
        z_t = torch.sigmoid(self.MLP_zt(input_mlp))
        #z_t = torch.sigmoid(self.pointer_weight)
        #print(z_t.item())
        p_t = torch.cat((o_t*z_t, attn_weights*(1-z_t)), 1)
        p_t = torch.log(p_t+1e-20)
        return output, hidden, p_t
    def inintHidden(self, h_a, h_d):

        r = torch.matmul(h_a, self.L) + torch.sum(h_d.view(-1, h_d.size(1), h_d.size(2)), dim=(1, 2), keepdim=True)/h_d.size(1)
        
        return self.Wb_0(r.view(-1, 1, self.hidden_size)).view(1, -1, self.hidden_size)
        

teacher_forcing_ratio = 1

def train(input_d, d_words, input_a, answer_pointer, target, q_words, encoder_d, encoder_a, decoder, encoder_d_optimizer, encoder_a_optimizer, decoder_optimizer, len_d, len_a, len_q): #return loss
    
    batch_size = input_d.size(0)
    
    encoder_d_optimizer.zero_grad()
    encoder_a_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    annotation_vector = torch.zeros(batch_size, len_d, 2*encoder_d.hidden_size, device=device)
    h_d = None
    for t_d in range(len_d):
        w = input_d[:, t_d].view(-1, 1)
        is_in_a = []
        for i in range(batch_size):           
            if t_d>=answer_pointer[i][0] and t_d<=answer_pointer[i][1]:
                is_in_a.append(1)
            else:
                is_in_a.append(0)
        o, h = encoder_d.forward(w, h_d, is_in_a)
        annotation_vector[:, t_d, :] = o[:, 0, :]
    
    h_a = None
    answer_encoding = None
    annotation_seq = torch.zeros(batch_size, len_a, 2*encoder_a.hidden_size, device=device) 
    for j in range(batch_size):
        actual_len =  answer_pointer[j][1] - answer_pointer[j][0] + 1
        annotation_seq[j, 0:actual_len, :] = annotation_vector[j, answer_pointer[j][0]:answer_pointer[j][1]+1,:]
    
    for t_a in range(len_a):
        input = input_a[:, t_a].view(-1, 1)
        answer_encoding, h_a = encoder_a.forward(input, annotation_seq[:, t_a, :], h_a)
        
       
    #s = answer_encoding.view(1, -1, hidden_size)
    s = decoder.inintHidden(answer_encoding, annotation_vector)
    SOS = np.ones(batch_size).astype(np.int32)
    decoder_input = torch.tensor(SOS, dtype=torch.long, device=device).view(-1, 1, 1) # SOS token  
    loss = 0
    
    
    using_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    

    predict_id = torch.zeros(batch_size, len_q, device=device)
    
    if using_teacher_forcing:

        #len_q = target.size(1)

        for t_q in range(len_q):

            o, s, p_t = decoder.forward(decoder_input, s, annotation_vector, answer_encoding, input_d)
            

    
            decoder_input = target[:, t_q].view(-1, 1, 1)
            
            #print(decoder_input.size())
            for i in range(batch_size):
                if target[i, t_q].item()== 3:

                    for j in range(input_d.size(1)):
                        if target[i, t_q].item() == input_d[i][j].item() and input_d[i][j].item() != 3:
                            decoder_input[i][0] = j+2004 #torch.tensor([j+2004], device=device).view(-1, 1)
                            break


            y_t = decoder_input.view(batch_size,)
            #predict_id[t_q] = torch.argmax(p_t, dim=1)
            loss+=nn.NLLLoss()(p_t, y_t)
    
    else:
        
        for t_q in range(len_q):

            o, s, p_t = decoder.forward(decoder_input,s,annotation_vector, answer_encoding, input_d)
            topi = torch.argmax(p_t, dim=1).detach()
            x_feed = topi.item()
            if x_feed > 2003+input_d.size(1):
                x_feed = 3
            
            decoder_input = torch.tensor(x_feed, dtype=torch.long, device=device).view(1, 1)
            y_t = target[:, t_q].view(1,)
            if y_t.item() == 3:
                for j in range(input_d.size(1)):
                    if target[:, t_q].item() == input_d[0][j].item() and input_d[0][j].item() != 3:
                        y_t = torch.tensor([j+2004], device=device).view(1,)
                        break
                        
            predict_id[t_q] = x_feed
            loss+=nn.NLLLoss()(p_t, y_t)
            
            if decoder_input.item() == 2: #EOS
                break
      
    
    
    
    loss.backward()
    encoder_d_optimizer.step()
    encoder_a_optimizer.step()
    decoder_optimizer.step()
    
         
    predict_id = 0        
    return loss.item(), predict_id

def evaluate(input_d, d_words, input_a, answer_pointer, target, q_words, encoder_d, encoder_a, decoder, len_d, len_a, len_q):
    batch_size = input_d.size(0)
    
    annotation_vector = torch.zeros(batch_size, len_d, 2*encoder_d.hidden_size, device=device)
    h_d = None
    for t_d in range(len_d):
        w = input_d[:, t_d].view(-1, 1)
        is_in_a = []
        for i in range(batch_size):           
            if t_d>=answer_pointer[i][0] and t_d<=answer_pointer[i][1]:
                is_in_a.append(1)
            else:
                is_in_a.append(0)
        o, h = encoder_d.forward(w, h_d, is_in_a)
        annotation_vector[:, t_d, :] = o[:, 0, :]
    
    h_a = None
    answer_encoding = None
    annotation_seq = torch.zeros(batch_size, len_a, 2*encoder_a.hidden_size, device=device) 
    for j in range(batch_size):
        actual_len =  answer_pointer[j][1] - answer_pointer[j][0] + 1
        annotation_seq[j, 0:actual_len, :] = annotation_vector[j, answer_pointer[j][0]:answer_pointer[j][1]+1,:]
    
    for t_a in range(len_a):
        input = input_a[:, t_a].view(-1, 1)
        answer_encoding, h_a = encoder_a.forward(input, annotation_seq[:, t_a, :], h_a)
        
       
    #s = answer_encoding.view(1, -1, hidden_size)
    s = decoder.inintHidden(answer_encoding, annotation_vector)
    SOS = np.ones(batch_size).astype(np.int32)
    decoder_input = torch.tensor(SOS, dtype=torch.long, device=device).view(-1, 1, 1) # SOS token  
    loss = 0
    
    
    using_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    

    predict_id = torch.zeros(batch_size, len_q, device=device)
    
    if using_teacher_forcing:

        #len_q = target.size(1)

        for t_q in range(len_q):

            o, s, p_t = decoder.forward(decoder_input, s, annotation_vector, answer_encoding, input_d)
            

    
            decoder_input = target[:, t_q].view(-1, 1, 1)
            
            #print(decoder_input.size())
            for i in range(batch_size):
                if target[i, t_q].item()== 3:

                    for j in range(input_d.size(1)):
                        if target[i, t_q].item() == input_d[i][j].item() and input_d[i][j].item() != 3:
                            decoder_input[i][0] = j+2004 #torch.tensor([j+2004], device=device).view(-1, 1)
                            break


            y_t = decoder_input.view(batch_size,)
            #predict_id[t_q] = torch.argmax(p_t, dim=1)
            loss+=nn.NLLLoss()(p_t, y_t)           
            
    return loss.item(), predict_id   

def main():
    HIDDEN_SIZE = 512

    document, question, answer, answer_pointer = data_reduction() #100, 20, 20
    d = id_sentence(document, 101)
    a = id_sentence(answer, 21)
    q = id_sentence(question, 21, shortlist= True)
    print(len(d))

    weight = np.load("weight.npy")
    weight = torch.FloatTensor(weight) 
    weight_shortlist = np.load("weight_shortlist.npy")
    weight_shortlist = torch.FloatTensor(weight_shortlist)

    id2word_shortlist = np.load("id2word_shortlist.npy")

    encoder_d = EncoderRNN_Document(int(HIDDEN_SIZE/2), weight).to(device)
    encoder_a = EncoderRNN_Answer(int(HIDDEN_SIZE/2), weight).to(device)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, weight_shortlist.size(0), weight_shortlist, weight, 0.1, 21, 101).to(device)
    if args.training:
        
        
        encoder_d.load_state_dict(torch.load("ckpt/encoder_d.pkl"))
        encoder_a.load_state_dict(torch.load("ckpt/encoder_a.pkl"))
        decoder.load_state_dict(torch.load("ckpt/decoder.pkl"))
        
        
        encoder_d_optimizer = optim.Adam(encoder_d.parameters(), lr=0.001)
        encoder_a_optimizer = optim.Adam(encoder_a.parameters(), lr=0.001)        
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
        
        epoch = 0
        loss_total = 0
        num_batch = 48
        predict_word = []
        batch_size = 512
        num = num_batch*batch_size
        print("total: ",num, " data, print every epoch")
        for iter in range(100000):
            i = iter%num_batch
            j = i%2000
            d_in = torch.tensor(d[i*batch_size:(i+1)*batch_size], dtype=torch.long, device=device).view(batch_size, -1) # batch, len_d
            a_in = torch.tensor(a[i*batch_size:(i+1)*batch_size], dtype=torch.long, device=device).view(batch_size, -1)
            q_in = torch.tensor(q[i*batch_size:(i+1)*batch_size], dtype=torch.long, device=device).view(batch_size, -1)
            a_p = answer_pointer[i*batch_size:(i+1)*batch_size]        

            loss_, predict_id_ = train(d_in, document[i*batch_size:(i+1)*batch_size], a_in, a_p, q_in, question[i*batch_size:(i+1)*batch_size], encoder_d, encoder_a, decoder,
                         encoder_d_optimizer, encoder_a_optimizer, decoder_optimizer, 101, 21, 21)
            loss_total+=loss_
            if i == num_batch-1:
                val_size=1000
                val_d_in = torch.tensor(d[num:num+val_size], dtype=torch.long, device=device).view(val_size, -1) # batch, len_d
                val_a_in = torch.tensor(a[num:num+val_size], dtype=torch.long, device=device).view(val_size, -1)
                val_q_in = torch.tensor(q[num:num+val_size], dtype=torch.long, device=device).view(val_size, -1)
                val_a_p = answer_pointer[num:num+val_size]        

                val_loss_, val_predict_id_ = evaluate(val_d_in, document[num:num+val_size], val_a_in, val_a_p, val_q_in, question[num:num+val_size], encoder_d, encoder_a, decoder, 101, 21, 21)
                print(epoch, loss_total/num_batch, val_loss_)
                loss_total = 0
                epoch+=1
                torch.save(encoder_d.state_dict(), 'ckpt/encoder_d.pkl')
                torch.save(encoder_a.state_dict(), 'ckpt/encoder_a.pkl')
                torch.save(decoder.state_dict(), 'ckpt/decoder.pkl')               
            
            
            if j == 999:
                print("epoch =", epoch, " ,loss =", loss_total / 1000)
                loss_total = 0

                '''
                for k in range(predict_id_.size(0)):
                    if predict_id_[k].item() <=2003:
                        predict_word.append(id2word_shortlist[int(predict_id_[k].item())])
                    elif int(predict_id_[k].item()-2004) < len(document[i]) :
                        predict_word.append(document[i][int(predict_id_[k].item()-2004)])
                    else:
                        predict_word.append('x')
                
                print('target: ', question[i])
                print('predict: ', predict_word) 
                predict_word = []
                '''

        
        

    if args.testing:    
        '''
        encoder_d.load_state_dict(torch.load("ckpt/encoder_d.pkl"))
        encoder_a.load_state_dict(torch.load("ckpt/encoder_a.pkl"))
        decoder.load_state_dict(torch.load("ckpt/decoder.pkl"))
        '''


        loss_total = 0
        num = 1000
        print("total: ",num, " data, print every 100 data")
        predict_word = []
        for iter in range(10000, 11000):
            i = iter
            j = i%100
            d_in = torch.tensor(d[i], dtype=torch.long, device=device).view(1, -1) # batch, len_d
            a_in = torch.tensor(a[i], dtype=torch.long, device=device).view(1, -1)
            q_in = torch.tensor(q[i], dtype=torch.long, device=device).view(1, -1)
            a_p = [answer_pointer[i]]        


            loss_, predict_id_ = evaluate(d_in, document[i], a_in, a_p, q_in, question[i], encoder_d, encoder_a, decoder,100, 20, 20)
            loss_total+=loss_
            
            
            
            if j == 99:
                print("loss=", loss_total / 100)
                loss_total = 0
                
                for k in range(predict_id_.size(0)):
                    if predict_id_[k].item() <=2003:
                        predict_word.append(id2word_shortlist[int(predict_id_[k].item())])
                    elif int(predict_id_[k].item()-2004) < len(document[i]) :
                        predict_word.append(document[i][int(predict_id_[k].item()-2004)])
                    else:
                        predict_word.append('x')
                
                print('target: ', question[i])
                print('predict: ', predict_word) 
                predict_word = []
        exit(0)
    
    
    if args.prepare:
        #embedding_weight()
        shortlist()
        exit(0)


if __name__ == '__main__':
    main()
    


