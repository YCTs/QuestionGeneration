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
	for i in range(len(ans_pos)):
		answer.append([])
		ans_pos[i] = ans_pos[i].split()
		for j in range(len(ans_pos[i])):
			if ans_pos[i][j] != 'O' :
				answer[i].append(context[i][j])

	return context, question, answer

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


def id_sentence(sentences):

    ids = []
    
    word2id = np.load("word2id.npy").item()
    
    for i, sentence in enumerate(sentences):
        ids.append([])
        for j, word in enumerate(sentence):
            if word in word2id:
                ids[i].append(word2id[word])
            else:
                ids[i].append(word2id["UNK"])
        

        if i == 1:
            print(sentence)
            print(ids[i])
    
    
    
    