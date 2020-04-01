# Restricted Boltzmann Machines (Probabilistic graphical model)

# Importing the libraries
import numpy as np
import pandas as pd
import torch     #
import torch.nn as nn   #to implement NN
import torch.nn.parallel   #for parallel computation
import torch.optim as optim   #for optimizer
import torch.utils.data
from torch.autograd import Variable   #for stochastic gradient descent

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')   #convert to array
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')        ##convert to array

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))   #to get total number of users
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))  

# Converting the data into an array with users(observations) in lines and movies(features) in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]    #to get all the movies for a particular user 
        id_ratings = data[:,2][data[:,0] == id_users]   #to get all the ratings by a particular user
        ratings = np.zeros(nb_movies)          #for those users who didnt rated a movie = 0         
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)
#each list corresponds to one user

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
#Tensors are simply arrays that contain elements of a single data type.
#torch tensor -  more efficient than numpy array

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1  # considerd as -ve rating (not liked)
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1   #3 and >3 will be considered as liked
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():                           #nv = number of visible nodes, nh = no. of hidden nodes
    def __init__(self, nv, nh):             #initialize the RBM 
        self.W = torch.randn(nh, nv)        #initialize weights acco. to ND
        self.a = torch.randn(1, nh)         #----||---- bias of nh
        self.b = torch.randn(1, nv)         #--||-- for nv
    def sample_h(self, x):    #sample the probabilities of the hidden nodes given the visible nodes
        wx = torch.mm(x, self.W.t())                               # = sigmoid activation function
        activation = wx + self.a.expand_as(wx)   #i/p * weight+bias
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):  #sample the probabilities of the visible nodes given the hidden nodes
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):              #contrastive divergence
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100            #number of features
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))