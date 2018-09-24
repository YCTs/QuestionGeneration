import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data_processing import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]
multi_gpu = False
if len(device_ids) >1:
    multi_gpu = True
    
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
        self.bn = nn.BatchNorm1d(self.embedding_dim-1)
        self.dropout = nn.Dropout(0.1)
        #self.bi_gru = nn.GRU(self.embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.bi_LSTM = nn.LSTM(self.embedding_dim, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input, hidden, c, is_in_a, b_feature):
        embedded = self.embedding(input).view(-1, 1, self.embedding_dim-1) #
        #embedded = self.bn(embedded.view(-1, self.embedding_dim-1)).view(-1, 1, self.embedding_dim-1)
        #embedded = self.dropout(embedded)
        #b_feature = torch.tensor(is_in_a, dtype=torch.float, device=device).view(-1, 1, 1)
        #print(b_feature.size())
        #b_feature = nn.D
        output = torch.cat((embedded, b_feature), 2)
        #output, hidden = self.bi_gru(output, hidden) ##hidden(1*num_dir, batch, h_dim), output(batch, 1, 2*h_dim)
        if hidden is not None:
            output, (hidden, c) = self.bi_LSTM(output, (hidden, c))
        else:
            output, (hidden, c) = self.bi_LSTM(output)

        output = self.dropout(output)
        hidden = self.dropout(hidden)

        return output, (hidden, c)

class EncoderRNN_Answer(nn.Module):
    def __init__(self, hidden_size, embedding_weight):
        super(EncoderRNN_Answer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.1)
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=embedding_freeze)
        self.embedding_dim = embedding_weight.size(1)

        #self.bi_gru = nn.GRU(self.embedding_dim + 2*self.hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.bi_LSTM = nn.LSTM(self.embedding_dim + 2*self.hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input, annotation_seq, hidden, c):
        # annotaion_seq = (-1, 1, 2*hidden_size)
        embedded = self.embedding(input).view(-1, 1, self.embedding_dim) # 
        #embedded = self.dropout(embedded)
        output = torch.cat((embedded, annotation_seq.unsqueeze(1)), 2)
        if hidden is not None:
            output, (hidden, c) = self.bi_LSTM(output, (hidden, c)) ##hidden(1*num_dir, batch, h_dim), output(batch, 1, 2*h_dim)
        else:
            output, (hidden, c) = self.bi_LSTM(output)
            
        output = self.dropout(output)
        hidden = self.dropout(hidden)

        return output, (hidden, c)
        
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
        #self.gru = nn.GRU(self.hidden_size*2 +self.embedding_dim, self.hidden_size, batch_first=True)
        self.LSTM = nn.LSTM(self.hidden_size*2 +self.embedding_dim, self.hidden_size, batch_first=True)
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

    def forward(self, input, hidden, c, annot_vector, answer_encoding, input_d):
        batch_size = input.size(0)
        hidden = hidden.view(1, batch_size, -1)
        embedded = torch.zeros(batch_size, 1, self.embedding_dim, device=device)
        input_d = input_d.view(batch_size, -1)
        for i in range(batch_size):
            if input[i].item() >2003:
                i_d = input[i].item() - 2004
                #print("copy")
                embedded[i] = self.embedding_d(input_d[i][i_d]).view(-1, 1, self.embedding_dim)
            else:
                embedded[i] = self.embedding(input[i]).view(-1, 1, self.embedding_dim) # -1, 1, embed_dim
        
        #embedded = self.dropout(embedded)
        hidden = self.dropout(hidden)
                
        #print(embedded.size())         #-1, 1, embed_dim
        #print(annot_vector.size())     #-1, len_c, hidden_size
        #print(answer_encoding.size())  #-1, 1, hidden_size
        len_c = annot_vector.size(1)
        #print(hidden.size(), embedded.size())
        in_attn = torch.cat((hidden.squeeze(0), embedded.squeeze(1), answer_encoding.squeeze(1), annot_vector.view(-1, len_c*self.hidden_size)), 1) # -1, 13156
        
    
        tanh = nn.Tanh()
        attn_weights = F.softmax(tanh(self.attn(in_attn)), dim=1) # -1, len_c
        
        c_t = torch.bmm(attn_weights.unsqueeze(1), annot_vector) # -1, 1, hidden_size
        
        v_t = torch.cat((c_t, answer_encoding), 2) # -1, 1, 2*hidden_size
        
        gru_input = torch.cat((v_t, embedded), 2)
        gru_input = self.dropout(gru_input)
        
        if c is not None:
            output, (hidden, c) = self.LSTM(gru_input, (hidden, c))
        else:
            output, (hidden, c) = self.LSTM(gru_input)

        input_mlp = torch.cat((output, v_t, embedded), 2).squeeze(1)
        #input_mlp = self.dropout(input_mlp)
        e_t = self.mlp(input_mlp)
        e_t = self.dropout(e_t)
        o_t = F.softmax(self.out(e_t), dim=1) # -1, output_size
        z_t = torch.sigmoid(self.MLP_zt(input_mlp))
        #z_t = torch.sigmoid(self.pointer_weight)

        p_t = torch.cat((o_t*(1 - z_t), attn_weights*(z_t)), 1)
        p_t = torch.log(p_t+1e-20)
        return output, hidden, c, p_t
    def inintHidden(self, h_a, h_d):
        #print(h_a.size())
        r = torch.matmul(h_a, self.L) + torch.sum(h_d.view(-1, h_d.size(1), h_d.size(2)), dim=(1, 2), keepdim=True)/h_d.size(1)
        
        return self.Wb_0(r.view(-1, 1, self.hidden_size)).view(1, -1, self.hidden_size)
        

teacher_forcing_ratio = 1

def train(input_d, d_words, input_a, answer_pointer, target, q_words, encoder_d, encoder_a, decoder, encoder_d_optimizer, encoder_a_optimizer, decoder_optimizer, len_d, len_a, len_q): #return loss
    
    batch_size = input_d.size(0)
    
    encoder_d_optimizer.zero_grad()
    encoder_a_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    annotation_vector = None
    if multi_gpu:
        annotation_vector = torch.zeros(batch_size, len_d, 2*encoder_d.module.hidden_size, device=device)
    else:
        annotation_vector = torch.zeros(batch_size, len_d, 2*encoder_d.hidden_size, device=device)
    h_d = None
    c_d = None
    for t_d in range(len_d):
        w = input_d[:, t_d].view(-1, 1)
        is_in_a = []
        for i in range(batch_size):           
            if t_d>=answer_pointer[i][0] and t_d<=answer_pointer[i][1]:
                is_in_a.append(1)
            else:
                is_in_a.append(0)
        b_feature = torch.tensor(is_in_a, dtype=torch.float, device=device).view(-1, 1, 1)
        o, (h_d, c_d) = encoder_d.forward(w, h_d, c_d, is_in_a, b_feature)
        
        annotation_vector[:, t_d, :] = o[:, 0, :]
    
    h_a = None
    c_a = None
    answer_encoding = None
    annotation_seq = None 
    if multi_gpu:
        annotation_seq = torch.zeros(batch_size, len_a, 2*encoder_a.module.hidden_size, device=device) 
    else:
        annotation_seq = torch.zeros(batch_size, len_a, 2*encoder_a.hidden_size, device=device)
    for j in range(batch_size):
        actual_len =  answer_pointer[j][1] - answer_pointer[j][0] + 1
        annotation_seq[j, 0:actual_len, :] = annotation_vector[j, answer_pointer[j][0]:answer_pointer[j][1]+1,:]
    
    for t_a in range(len_a):
        input = input_a[:, t_a].view(-1, 1)
        answer_encoding, (h_a, c_a) = encoder_a.forward(input, annotation_seq[:, t_a, :], h_a, c_a)
        #h_a = h.view(batch_size, 2, -1)
        
       
    #s = answer_encoding.view(1, batch_size, -1)
    s =None
    c_q = None
    if multi_gpu:
        s = decoder.module.inintHidden(answer_encoding, annotation_vector).view(batch_size, 1, -1)
    else:
        s = decoder.inintHidden(answer_encoding, annotation_vector).view(batch_size, 1, -1)
    SOS = np.ones(batch_size).astype(np.int32)
    decoder_input = torch.tensor(SOS, dtype=torch.long, device=device).view(-1, 1, 1) # SOS token  
    loss = 0
    
    
    using_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    

    predict_id = torch.zeros(batch_size, len_q, device=device)
    
    if using_teacher_forcing:

        #len_q = target.size(1)

        for t_q in range(len_q):
            
            o, s, c_q, p_t = decoder.forward(decoder_input, s, c_q, annotation_vector, answer_encoding, input_d)
            

    
            decoder_input = target[:, t_q].view(-1, 1, 1)
            '''
            print(decoder_input.requires_grad)
            for i in range(batch_size):
                if target[i][t_q].item()== 3:
                    word = q_words[i][t_q]
                    for j in range(len(d_words[i])):
                        if word == d_words[i][j] and input_d[i][j].item() != 3:
                            print("copy")
                            #decoder_input[i] = torch.tensor([j+2004], device=device).view(1, 1)
                            break
            '''
            #print(p_t.size())
            y_t = decoder_input.view(batch_size,)
            predict_id[:, t_q] = torch.argmax(p_t, dim=1)
            loss+=nn.NLLLoss(ignore_index=0)(p_t, y_t)

    
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
    
    return loss.item(), predict_id

def evaluate(input_d, d_words, input_a, answer_pointer, target, q_words, encoder_d, encoder_a, decoder, len_d, len_a, len_q):
    batch_size = input_d.size(0)

    annotation_vector = None
    if multi_gpu:
        annotation_vector = torch.zeros(batch_size, len_d, 2*encoder_d.module.hidden_size, device=device)
    else:
        annotation_vector = torch.zeros(batch_size, len_d, 2*encoder_d.hidden_size, device=device)
    h_d = None
    c_d = None
    for t_d in range(len_d):
        w = input_d[:, t_d].view(-1, 1)
        is_in_a = []
        for i in range(batch_size):           
            if t_d>=answer_pointer[i][0] and t_d<=answer_pointer[i][1]:
                is_in_a.append(1)
            else:
                is_in_a.append(0)
        b_feature = torch.tensor(is_in_a, dtype=torch.float, device=device).view(-1, 1, 1)
        o, (h_d, c_d) = encoder_d.forward(w, h_d, c_d, is_in_a, b_feature)
        
        annotation_vector[:, t_d, :] = o[:, 0, :]
    
    h_a = None
    c_a = None
    answer_encoding = None
    annotation_seq = None 
    if multi_gpu:
        annotation_seq = torch.zeros(batch_size, len_a, 2*encoder_a.module.hidden_size, device=device) 
    else:
        annotation_seq = torch.zeros(batch_size, len_a, 2*encoder_a.hidden_size, device=device)
    for j in range(batch_size):
        actual_len =  answer_pointer[j][1] - answer_pointer[j][0] + 1
        annotation_seq[j, 0:actual_len, :] = annotation_vector[j, answer_pointer[j][0]:answer_pointer[j][1]+1,:]
    
    for t_a in range(len_a):
        input = input_a[:, t_a].view(-1, 1)
        answer_encoding, (h_a, c_a) = encoder_a.forward(input, annotation_seq[:, t_a, :], h_a, c_a)
        #h_a = h.view(batch_size, 2, -1)
        
       
    #s = answer_encoding.view(1, batch_size, -1)
    s =None
    c_q = None
    if multi_gpu:
        s = decoder.module.inintHidden(answer_encoding, annotation_vector).view(batch_size, 1, -1)
    else:
        s = decoder.inintHidden(answer_encoding, annotation_vector).view(batch_size, 1, -1)
    SOS = np.ones(batch_size).astype(np.int32)
    decoder_input = torch.tensor(SOS, dtype=torch.long, device=device).view(-1, 1, 1) # SOS token  
    loss = 0
    
    
    using_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    

    predict_id = torch.zeros(batch_size, len_q, device=device)
    
    if using_teacher_forcing:

        #len_q = target.size(1)

        for t_q in range(len_q):
            
            o, s, c_q, p_t = decoder.forward(decoder_input, s, c_q, annotation_vector, answer_encoding, input_d)
            

    
            decoder_input = target[:, t_q].view(-1, 1, 1)
            '''
            print(decoder_input.requires_grad)
            for i in range(batch_size):
                if target[i][t_q].item()== 3:
                    word = q_words[i][t_q]
                    for j in range(len(d_words[i])):
                        if word == d_words[i][j] and input_d[i][j].item() != 3:
                            print("copy")
                            #decoder_input[i] = torch.tensor([j+2004], device=device).view(1, 1)
                            break
            '''
            #print(p_t.size())
            y_t = decoder_input.view(batch_size,)
            predict_id[:, t_q] = torch.argmax(p_t, dim=1)
            loss+=nn.NLLLoss(ignore_index=0)(p_t, y_t)        
            
    return loss.item(), predict_id   

def main():
    HIDDEN_SIZE = 700

    document, question, answer, answer_pointer = data_reduction() #100, 20, 20
    d = id_sentence(document, 101)
    a = id_sentence(answer, 21)
    q = id_sentence(question, 21, shortlist= True)
    
    val_document, val_question, val_answer, val_answer_pointer = data_reduction(val=True) #100, 20, 20
    val_d = id_sentence(val_document, 101)
    val_a = id_sentence(val_answer, 21)
    val_q = id_sentence(val_question, 21, shortlist= True)
    
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            if q[i][j] == 3:
                word = question[i][j]
                for k in range(len(document[i])):
                    if word == document[i][k] and d[i][k] !=3 :
                        q[i][j] = 2004+k
                        break
    for i in range(val_q.shape[0]):
        for j in range(val_q.shape[1]):
            if val_q[i][j] == 3:
                word = val_question[i][j]
                for k in range(len(val_document[i])):
                    if word == val_document[i][k] and val_d[i][k] !=3 :
                        val_q[i][j] = 2004+k
                        break

    weight = np.load("weight.npy")
    weight = torch.FloatTensor(weight) 
    weight_shortlist = np.load("weight_shortlist.npy")
    weight_shortlist = torch.FloatTensor(weight_shortlist)

    id2word_shortlist = np.load("id2word_shortlist.npy")
    
    encoder_d = EncoderRNN_Document(int(HIDDEN_SIZE/2), weight).to(device)
    encoder_a = EncoderRNN_Answer(int(HIDDEN_SIZE/2), weight).to(device)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, weight_shortlist.size(0), weight_shortlist, weight, 0.3, 21, 101).to(device)
    
    if multi_gpu:
        encoder_d = nn.DataParallel(EncoderRNN_Document(int(HIDDEN_SIZE/2), weight), device_ids=device_ids).cuda()
        encoder_a = nn.DataParallel(EncoderRNN_Answer(int(HIDDEN_SIZE/2), weight), device_ids=device_ids).cuda()
        decoder = nn.DataParallel(AttnDecoderRNN(HIDDEN_SIZE, weight_shortlist.size(0), weight_shortlist, weight, 0.3, 21, 101), device_ids=device_ids).cuda()

    
    

    
    if args.training:
        
        '''
        encoder_d.load_state_dict(torch.load("ckpt/encoder_d.pkl"))
        encoder_a.load_state_dict(torch.load("ckpt/encoder_a.pkl"))
        decoder.load_state_dict(torch.load("ckpt/decoder.pkl"))
        '''
        
        encoder_d_optimizer = optim.Adam(encoder_d.parameters(), lr=0.0001)
        encoder_a_optimizer = optim.Adam(encoder_a.parameters(), lr=0.0001)        
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)
        
        epoch = 0
        loss_total = 0
        num_batch = 624  #192
        val_predict_word = []
        predict_word = []
        batch_size = 128
        num = num_batch*batch_size
        print("total: ",num, " data, print every epoch")
        loss_min = 99999
        for iter in range(1000000):
            i = iter%num_batch

            d_in = torch.tensor(d[i*batch_size:(i+1)*batch_size], dtype=torch.long, device=device).view(batch_size, -1) # batch, len_d
            a_in = torch.tensor(a[i*batch_size:(i+1)*batch_size], dtype=torch.long, device=device).view(batch_size, -1)
            q_in = torch.tensor(q[i*batch_size:(i+1)*batch_size], dtype=torch.long, device=device).view(batch_size, -1)
            a_p = answer_pointer[i*batch_size:(i+1)*batch_size]        

            loss_, predict_id_ = train(d_in, document[i*batch_size:(i+1)*batch_size], a_in, a_p, q_in, question[i*batch_size:(i+1)*batch_size], encoder_d, encoder_a, decoder,
                         encoder_d_optimizer, encoder_a_optimizer, decoder_optimizer, 101, 21, 21)
            loss_total+=loss_
            if i == num_batch-1:
                val_size=500
                val_d_in = torch.tensor(val_d[:val_size], dtype=torch.long, device=device).view(val_size, -1) # batch, len_d
                val_a_in = torch.tensor(val_a[:val_size], dtype=torch.long, device=device).view(val_size, -1)
                val_q_in = torch.tensor(val_q[:val_size], dtype=torch.long, device=device).view(val_size, -1)
                val_a_p = val_answer_pointer[:val_size]        

                val_loss_, val_predict_id_ = evaluate(val_d_in, val_document[:val_size], val_a_in, val_a_p, val_q_in, val_question[:val_size], encoder_d, encoder_a, decoder, 101, 21, 21)
                print(epoch, loss_total/num_batch, val_loss_)

                for j in range(predict_id_.size(1)):
                    if predict_id_[0][j].item()<2004:
                        predict_word.append(id2word_shortlist[int(predict_id_[0][j].item())])
                    elif int(predict_id_[0][j].item()-2004) < len(document[num-batch_size]) :
                        predict_word.append(document[num-batch_size][int(predict_id_[0][j].item()-2004)])
                    else:
                        predict_word.append("x")
                        
                for j in range(val_predict_id_.size(1)):
                    if val_predict_id_[1][j].item()<2004:
                        val_predict_word.append(id2word_shortlist[int(val_predict_id_[1][j].item())])
                    elif int(val_predict_id_[1][j].item()-2004) < len(val_document[1]) :
                        val_predict_word.append(val_document[1][int(val_predict_id_[1][j].item()-2004)])
                    else:
                        val_predict_word.append("x")
                
                loss_total = 0
                epoch+=1
                if loss_min > val_loss_:
                    loss_min = val_loss_
                    torch.save(encoder_d.state_dict(), 'ckpt/encoder_d.pkl')
                    torch.save(encoder_a.state_dict(), 'ckpt/encoder_a.pkl')
                    torch.save(decoder.state_dict(), 'ckpt/decoder.pkl')               
            
            

                print("=================Result====================:")
                print("-------------Traning Set--------------")
                print('ground truth: ', question[num-batch_size])
                print('predict: ', predict_word)                             
                print("-------------Validataion--------------")
                print('ground truth: ', val_question[1])
                print('predict: ', val_predict_word) 
                print("===========================================")
                predict_word = []
                val_predict_word = []
                

        
        

    if args.testing:    
        '''
        encoder_d.load_state_dict(torch.load("ckpt/encoder_d.pkl"))
        encoder_a.load_state_dict(torch.load("ckpt/encoder_a.pkl"))
        decoder.load_state_dict(torch.load("ckpt/decoder.pkl"))
        '''


        loss_total = 0
        num = 1000

        exit(0)
    
    
    if args.prepare:
        #embedding_weight()
        shortlist()
        exit(0)


if __name__ == '__main__':
    main()
    


