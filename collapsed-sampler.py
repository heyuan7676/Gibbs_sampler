import os
import sys
import pandas as pd
pd.set_option('display.precision', 13)
import numpy as np
from itertools import chain
from collections import Counter
from numpy import random
import time
import bisect
from copy import copy


class literature:
    def __init__(self, filename_train, filename_test, K):
        ## all topics
        self.K = K
        ## all documents: training + test
        self.documents_train, self.documents_test = list(), list()
        ## train
        datafile = open(filename_train)
        for l in datafile:
            self.documents_train.append(document(l,K))
        ## test
        datafile = open(filename_test)
        for l in datafile:
            self.documents_test.append(document(l,K))
        self.D = len(self.documents_train) + len(self.documents_test)
        ## all words
        all_words = list(chain(*[d.words for d in (self.documents_train + self.documents_test)]))
        all_words = list(set(all_words))
        ## length of all words
        V = len(all_words)
        self.V = V
        ## words index in all documents
        self.words_index = dict(zip(all_words, range(V)))
        ## fill in index for all documents
        for d in self.documents_train:
            d.word_index(self.words_index)
        for d in self.documents_test:
            d.word_index(self.words_index)
        ## initialize Nwk for training data
        self.cal_Nwk()

    def cal_Nwk(self):
        ## Nwk
        V = self.V
        K = self.K
        self.Nwk, self.Nwk_0, self.Nwk_1 = np.zeros([V,K]), np.zeros([V,K]), np.zeros([V,K])     ### V x K
        ## Nwk(s): V x K. ordered lists of ordered lists
        for d in self.documents_train:
            d.cal_nwk(self.V, self.K)
            self.Nwk = self.Nwk + d.nwk
            self.Nwk_0 = self.Nwk_0 + d.nwk_0
            self.Nwk_1 = self.Nwk_1 + d.nwk_1
        self.Nk = np.sum(self.Nwk, axis = 0)
        self.Nk_0 = np.sum(self.Nwk_0, axis = 0)
        self.Nk_1 = np.sum(self.Nwk_1, axis = 0)
    
    def gibbs_sampler(self, datatype):
        if datatype == 'train':
            for d in self.documents_train:
                d.update_z_x_train(self.Nk, self.Nk_0, self.Nk_1, self.Nwk, self.Nwk_0, self.Nwk_1, self.V, self.K)
        elif datatype == 'test':
            for d in self.documents_test:
                d.update_z_x_test(self.Nk, self.Nk_0, self.Nk_1, self.Nwk, self.Nwk_0, self.Nwk_1, self.V, self.K)
        
    def MAP_estimate(self, datatype):
        if datatype == 'train':
            for d in self.documents_train:
                d.estimate_theta(self.K)
            ### equation 6 - 7
            self.phi   = np.divide((self.Nwk + beta), (self.Nk + self.V * beta))     ### V x K
            self.phi_0 = np.divide((self.Nwk_0 + beta), (self.Nk_0 + self.V * beta))
            self.phi_1 = np.divide((self.Nwk_1 + beta), (self.Nk_1 + self.V * beta))
            self.phi_sum = np.sum(self.phi, axis = 0)
            self.phi_0_sum = np.sum(self.phi_0, axis = 0)
            self.phi_1_sum = np.sum(self.phi_1, axis = 0)
        elif datatype == 'test':
            ### theta for each document
            for d in self.documents_test:
                d.estimate_theta(self.K)
    
    def cal_llk(self, datatype):
        ### equation 8
        llk = 0.0
        if datatype == 'train':
            documents = self.documents_train
        else:
            documents = self.documents_test
        for d in documents:
            each_d = 0.0
            if d.corpus == 1:
                phi_c = self.phi_1
            else:
                phi_c = self.phi_0
            for w in d.words:
                current_v = w
                most_inner = (1-lmda)*self.phi[current_v,] + lmda*(phi_c[current_v,])
                unit = np.multiply(d.theta, most_inner)  ##  1 x K
                sum_z = sum(unit)
                log_sum_z = np.log(sum_z)
                each_d += log_sum_z
            llk += each_d
        return llk
                
        
        
        
        
class document:
    
    def __init__(self, l, K):
        line = l.rstrip().split(' ')
        self.corpus = int(line[0])
        self.words = line[1:]
        ### randomly assign z and x when creating the object
        self.v = len(self.words)   # total number of words in document d
        random.seed(0)
        self.z = random.choice(range(K), size = self.v)   # z: 1 x v. the index of topics for all words.
        self.x = random.choice([0,1], size = self.v)      # x: 1 x v. which phi the word is drawn from.  
        self.ndk = np.zeros(K)
        self.cal_ndk(K)
        
    def word_index(self, words_index):
        for t in range(self.v):
            self.words[t] = words_index[self.words[t]]
        
    def cal_nwk(self, V, K):
        ### append assignments (z) to words. nwk: V x K. ordered lists of ordered lists
        self.nwk, self.nwk_0, self.nwk_1 = np.zeros([V,K]), np.zeros([V,K]), np.zeros([V,K])
        for t in range(self.v):
            w_idx = self.words[t]
            w_z  = self.z[t]
            w_x = self.x[t]
            if w_x == 0:
                self.nwk[w_idx, w_z] += 1 
            elif w_x == 1 and self.corpus == 0:
                self.nwk_0[w_idx, w_z] += 1
            elif w_x == 1 and self.corpus == 1:
                self.nwk_1[w_idx, w_z] += 1
        
    def cal_ndk(self, K):
        ### number of words assigned in each topic
        ndk = Counter(self.z)
        for k in range(K):
            self.ndk[k] = ndk[k]      ## 1 x K

    def update_z_x_train(self, Nk, Nk_0, Nk_1, Nwk, Nwk_0, Nwk_1, V, K):
        ### update z and x word by word      
        Nd = self.v - 1 
        for t in range(self.v):
            current_k = self.z[t]                     # current topic / k
            current_v = self.words[t]                 # current word location
            current_x = self.x[t]                     # current x
            ### update z: equation (3)
            ## for numerator
            self.ndk[current_k] -= 1
            ## first term
            first_term = np.divide((self.ndk + alpha), (Nd + K * alpha))    ## 1 x K
            ## second term
            if current_x == 0:
                temp_Nwk = Nwk
                temp_Nk = Nk
            elif current_x == 1 and self.corpus == 0:
                temp_Nwk = Nwk_0
                temp_Nk = Nk_0
            elif current_x == 1 and self.corpus == 1:
                temp_Nwk = Nwk_1
                temp_Nk = Nk_1
            ## delete the current assignment
            temp_Nk[current_k] -= 1
            temp_Nwk[current_v, current_k] -= 1
            ## calcualte
            second_term = np.divide((temp_Nwk[current_v,] + beta), (temp_Nk + V * beta))   ## 1 x K
            if (second_term<0).any():
                print current_v
            ## pdf
            prop = np.multiply(first_term, second_term)   ## 1 x K
            prop_sum = sum(prop) 
            pdf = prop / prop_sum
            ## sample z
            assert (pdf>=0).all()
            random_z = np.random.random_sample()
            self.z[t] = bisect.bisect_left(np.cumsum(pdf), random_z)
            ## update Nwk, Nk and ndk
            temp_Nwk[current_v, self.z[t]] += 1
            temp_Nk[self.z[t]] += 1
            self.ndk[self.z[t]] += 1
            current_k = self.z[t]               ### use the new z!

            ### update x: equation (4)
            if self.corpus == 0:
                temp_Nwk_c = Nwk_0
                temp_Nk_c = Nk_0
            else:
                temp_Nwk_c = Nwk_1
                temp_Nk_c = Nk_1
            ## delete the current assignment
            if current_x == 0:
                Nwk[current_v,current_k] -= 1      
                Nk[current_k] -= 1
            else:
                temp_Nwk_c[current_v, current_k] -= 1
                temp_Nk_c[current_k] -= 1
            # equation (4)
            prob_0 = (1-lmda) * (Nwk[current_v,current_k] + beta) / (Nk[current_k] + V * beta)
            prob_1 = lmda * (temp_Nwk_c[current_v,current_k] + beta) / (temp_Nk_c[current_k] + V * beta)
            ## pdf
            prop = list([prob_0, prob_1])   ## 1 x 2
            prop_sum = prob_0 + prob_1
            pdf = prop / prop_sum
            ## sample x
            assert (pdf>=0).all()
            self.x[t] = np.random.choice(2, p = pdf)
            # random_x = np.random.random_sample()
            # self.x[t] = bisect.bisect_left(np.cumsum(pdf), random_x)
            
            ## update Nwk and Nwk_c
            if self.x[t] == 1:
                temp_Nwk_c[current_v, self.z[t]] += 1
                temp_Nk_c[current_k] += 1
            else:
                Nwk[current_v, self.z[t]] += 1
                Nk[current_k] += 1

    def update_z_x_test(self, Nk, Nk_0, Nk_1, Nwk, Nwk_0, Nwk_1, V, K):
        ### update z and x word by word      
        Nd = self.v - 1 
        for t in range(self.v):
            current_k = self.z[t]                     # current topic / k
            current_v = self.words[t]                 # current word location
            current_x = self.x[t]                     # current x 
            ### update z: equation (3)
            ## for numerator
            self.ndk[current_k] -= 1
            ## first term
            first_term = np.divide((self.ndk + alpha), (Nd + K * alpha))    ## 1 x K
            ## second term
            if current_x == 0:
                temp_Nwk = Nwk 
                temp_Nk = Nk
            elif current_x == 1 and self.corpus == 0:
                temp_Nwk = Nwk_0
                temp_Nk = Nk_0
            elif current_x == 1 and self.corpus == 1:
                temp_Nwk = Nwk_1
                temp_Nk = Nk_1
            ## calcualte
            second_term = np.divide((temp_Nwk[current_v,] + beta), (temp_Nk + V * beta))   ## 1 x K
            ## pdf
            prop = np.multiply(first_term, second_term)   ## 1 x K
            prop_sum = sum(prop) 
            pdf = prop / prop_sum
            ## sample z
            assert (pdf>=0).all()
            random_z = np.random.random_sample()
            self.z[t] = bisect.bisect_left(np.cumsum(pdf), random_z)
            ## update ndk
            self.ndk[self.z[t]] += 1
            
            ### update x: equation (4)
            current_k = self.z[t]
            if self.corpus == 0:
                temp_Nwk_c = Nwk_0
                temp_Nk_c = Nk_0
            else:
                temp_Nwk_c = Nwk_1
                temp_Nk_c = Nk_1
            # equation (4)
            prob_0 = (1-lmda) * (Nwk[current_v,current_k] + beta) / (Nk[current_k] + V * beta)
            prob_1 = lmda * (temp_Nwk_c[current_v,current_k] + beta) / (temp_Nk_c[current_k] + V * beta)
            ## pdf
            prop = list([prob_0, prob_1])   ## 1 x 2
            prop_sum = prob_0 + prob_1
            pdf = prop / prop_sum
            ## sample x
            assert (pdf>=0).all()
            random_x = np.random.random_sample()
            self.x[t] = bisect.bisect_left(np.cumsum(pdf), random_x)

    def estimate_theta(self, K):
        ### equation 5
        self.theta = np.divide((self.ndk + alpha), (self.v + K * alpha))    ### D x K



def phi_addindex(phi, words_index):
    keys = words_index.keys()
    idx = words_index.values()
    words = []
    for t in range(len(phi)):
        words.append(keys[idx.index(t)])
    phi_df = pd.DataFrame(phi)
    phi_df['words'] = words
    df = phi_df.set_index(['words'])
    return df



def main(train='input-train.txt', test='input-test.txt', output='output.txt', K=10, lmda=.5, alpha=.1, beta=.01, iter_max=1100, burn_in=1000):
    filename_train = os.path.join(train)
    filename_test = os.path.join(test)
    data = literature(filename_train, filename_test, K)

    theta = np.zeros([len(data.documents_train), data.K])
    phi, phi_0, phi_1 = np.zeros([data.V, data.K]), np.zeros([data.V, data.K]), np.zeros([data.V, data.K])
    llk_train, llk_test = [], []
    t = []

    for it in range(iter_max):
        START = time.time()
        ### train
        data.gibbs_sampler('train')            ## (a)
        data.MAP_estimate('train')             ## (b)
        ### test
        data.gibbs_sampler('test')
        data.MAP_estimate('test')
        if it > burn_in:                       ## (c)
            phi += data.phi
            phi_0 += data.phi_0
            phi_1 += data.phi_1
            for d in xrange(len(data.documents_train)):
                theta[d] = data.documents_train[d].theta
        ### llk
        llk_train.append(data.cal_llk('train'))
        llk_test.append(data.cal_llk('test'))
        t.append((time.time() - START))
    
    ### overall llk
    data.phi = phi / (iter_max - burn_in)
    data.phi_0 = phi_0 / (iter_max - burn_in)
    data.phi_1 = phi_1 / (iter_max - burn_in)
    print 'overall llk for test data:', data.cal_llk('test')

    pd.DataFrame(llk_train).to_csv('%s-trainll' % output, sep=' ', index=False , header=False)
    pd.DataFrame(llk_test).to_csv('%s-testll' % output, sep=' ', index=False, header=False)
    pd.DataFrame(t).to_csv('%s-t' % output, sep=' ', index=False, header=False)
    pd.DataFrame(theta / (iter_max - burn_in)).to_csv('%s-theta' % output, sep=' ', index=False, header=False)
    phi = phi_addindex(data.phi, data.words_index)
    phi.to_csv('%s-phi' % output, sep=' ', index=True, header=False)
    phi0 = phi_addindex(data.phi_0, data.words_index)
    phi0.to_csv('%s-phi0' % output, sep=' ', index=True, header=False)
    phi1 = phi_addindex(data.phi_1, data.words_index)
    phi1.to_csv('%s-phi1' % output, sep=' ', index=True, header=False)



if __name__ == '__main__':
    para = sys.argv[1:]
    train, test, output = para[:3]
    K = int(para[3])
    lmda, alpha, beta = [float(x) for x in para[4:7]]
    iter_max, burn_in = [int(x) for x in para[7:]]
    main(train, test, output, K, lmda, alpha, beta, iter_max, burn_in)

