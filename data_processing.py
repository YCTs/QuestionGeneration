import numpy as np
import torch

def train_data():
			
    context = open("./data/train/train.txt.source.txt").readlines()
    for i in range(len(context)):
        context[i] = context[i].split()

    question = open('./data/train/train.txt.target.txt').readlines()
    for i in range(len(question)):
        question[i] = question[i].split()

    ans_pos  = open('./data/train/train.txt.bio').readlines()
    answer = []
    answer_pointer = []
    for i in range(len(ans_pos)):
        answer.append([])
        answer_pointer.append([])
        ans_pos[i] = ans_pos[i].split()
        ans_s = -1
        ans_e = -1
        for j in range(len(ans_pos[i])):
    
            if ans_pos[i][j] != 'O' :
                answer[i].append(context[i][j])
                ans_e = j
                if ans_s == -1:
                    ans_s = j
                    answer_pointer[i].append(j)
        answer_pointer[i].append(ans_e)        

    return context, question, answer, answer_pointer

def embedding_weight():
	f = open('glove.6B.100d.txt').readlines()
	dim = len(f[0].split()) - 1
	weight = []
	word2id = {}
	id2word = []

	np.random.seed(7)
	# pad, sos, eos, unk----------

	word2id['PAD'] = 0
	word2id['SOS'] = 1
	word2id['EOS'] = 2
	word2id['UNK'] = 3
	id2word.append('PAD')
	id2word.append('S0S')
	id2word.append('EOS')
	id2word.append('UNK')

	weight.append(np.random.randn(dim))
	weight.append(np.random.randn(dim))
	weight.append(np.random.randn(dim))
	weight.append(np.random.randn(dim))
	# pad, sos, eos, unk --------


	
	for i in range(len(f)):
		f[i] = f[i].split()
		vector_element = np.array(f[i][1:]).astype(np.float32)
		word2id[f[i][0]] = int(i+4)
		id2word.append(f[i][0])
		weight.append(vector_element)
	weight = np.array(weight).reshape(-1, dim)
	#print(weight.shape)
	
	np.save('weight.npy', weight)
	np.save('word2id.npy', word2id)
	np.save('id2word.npy', id2word)
    
def max_length(x):
    max_len = 0
    for i, sentence in enumerate(x):
        if len(sentence) > max_len:
            max_len = len(sentence)
    return max_len

def data_reduction():
    context, question, answer, answer_pointer = train_data()
    count = 0
    index_c = []
    for i in range(len(context)):
        if len(context[i]) <= 100:
            count+=1
            index_c.append(i)
    #print(count/len(context)) 
    count = 0
    index_q = []
    for i in range(len(question)):
        if len(question[i]) <= 20:
            count+=1
            index_q.append(i)
    #print(count/len(question))
    
    count = 0
    index_a = []
    for i in range(len(answer)):
        if len(answer[i]) <= 20:
            count+=1
            index_a.append(i)
    #print(count/len(answer))
    
    print(len(set(index_c)&set(index_q)&set(index_a))/len(context))
    set_i = set(index_c)&set(index_q)&set(index_a)
    index = [i for i in set_i]
    context_new = []
    question_new = []
    answer_new = []
    answer_pointer_new = []
    
    for i in range(len(index)):
        context_new.append(context[index[i]])
        question_new.append(question[index[i]])
        answer_new.append(answer[index[i]])
        answer_pointer_new.append(answer_pointer[index[i]])
        
        
    return context_new, question_new, answer_new, answer_pointer_new

def id_sentence(sentences, max_len, shortlist=False):
    ids= []
    if False == shortlist:
        

        word2id = np.load("word2id.npy").item()

        for i, sentence in enumerate(sentences):
            ids.append(np.zeros(max_len))
            for j, word in enumerate(sentence):
                if word in word2id:
                    ids[i][j] = word2id[word]
                else:
                    ids[i][j] = word2id["UNK"]
            ids[i][len(sentence)] = int(2)
            
        ids = np.array(ids).reshape(-1, max_len).astype(np.int32)
    else:
        

        word2id = np.load("word2id_shortlist.npy").item()

        for i, sentence in enumerate(sentences):
            ids.append(np.zeros(max_len))
            for j, word in enumerate(sentence):
                if word in word2id:
                    ids[i][j] = word2id[word]
                else:
                    ids[i][j] = word2id["UNK"]
            ids[i][len(sentence)] = int(2)

        ids = np.array(ids).reshape(-1, max_len).astype(np.int32)
    
    return ids
def dynamic_id_sentence(sentences, shortlist=False): # different length
    ids = []
    word2id = None
    if shortlist:
        word2id = np.load("word2id_shortlist.npy").item()
    else:
        word2id = np.load("word2id.npy").item()
     
    for i, sentence in enumerate(sentences):
        ids.append([])
        for j, word in enumerate(sentence):
            if word in word2id:
                ids[i].append(word2id[word])
            else:
                ids[i].append(word2id["UNK"])
        ids[i].append(int(2))       
    return ids
    
def shortlist():
    
    context, question, answer, answer_pointer = train_data()
    
    count = {}

    for i, sentence in enumerate(question):
        for j , word in enumerate(sentence):
            if word not in count:
                count[word] = 0
            else:
                count[word] += 1
                
    x = sorted(count.items(), key = lambda item:item[1]) #(word, num)
    
    x_top2000 = x[-2000:]
    
    word2id = {}
    word2id['PAD'] = 0
    word2id['SOS'] = 1
    word2id['EOS'] = 2
    word2id['UNK'] = 3
    id2word=['PAD', 'SOS', 'EOS', 'UNK']
    
    weight = np.zeros((2004, 100))

    
    word2id_encoder = np.load("word2id.npy").item()
    weight_encoder = np.load("weight.npy")
    
    for i in range(4):
        weight[i] = weight_encoder[i]
    print(weight.shape)
    np.random.seed(0)
    i = 3
    for word, item in x_top2000:
        i+=1
        word2id[word] = i
        id2word.append(word)
        
        if word in word2id_encoder:
            i_encoder = word2id_encoder[word]
            weight[i] = weight_encoder[i_encoder]
        else:
            weight[i] = np.random.randn(100)
    
    np.save('word2id_shortlist.npy', word2id)
    np.save('weight_shortlist.npy', weight)
    np.save('id2word_shortlist.npy', id2word)