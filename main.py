# -*- coding: utf-8 -*-
"""
The work is based on neoDTI, please visit the https://github.com/FangpingWan/NeoDTI for more details, including datasets.
Thanks to Fangping Wan and others for their contributions.
"""
import numpy as np
import pickle
import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split,StratifiedKFold
import sys
from optparse import OptionParser
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from tensorflow.python.framework import ops
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#===================================#

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def a_layer(x,units):
    W = weight_variable([x.get_shape().as_list()[1],units])
    b = bias_variable([units])
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W))
    return tf.nn.relu(tf.matmul(x, W) + b)

def meta_layer(A,B,units):
    b = bias_variable([units])
    return tf.nn.relu(tf.matmul(A,B) + b)


def bi_layer(x0,x1,sym,dim_pred):
    if sym == False:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        W1p = weight_variable([x1.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1p))
        return tf.matmul(tf.matmul(x0, W0p), 
                            tf.matmul(x1, W1p),transpose_b=True)
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        return tf.matmul(tf.matmul(x0, W0p), 
                            tf.matmul(x1, W0p),transpose_b=True)

#==================================#

parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d")
parser.add_option("-n","--n",default=1, help="global norm to be clipped")
parser.add_option("-k","--k",default=512,help="The dimension of project matrices k")
parser.add_option("-t","--t",default = "o",help="Test scenario")
parser.add_option("-r","--r",default = "ten",help="positive negative ratio")

(opts, args) = parser.parse_args()


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix,0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


def node_shuffle(x):

    list_idx = [i for i in range(x.get_shape().as_list()[0])]
    random.shuffle(list_idx)
    node_shuffle_tensor = x[list_idx[0]]
    flag = 0
    for idx in list_idx[1:]:
        if flag == 0:
            node_shuffle_tensor = tf.concat([tf.reshape(node_shuffle_tensor,[1,x.shape[1]]),tf.reshape(x[idx],[1,x.shape[1]])],axis=0)
            flag = 1
        else:
            node_shuffle_tensor = tf.concat([node_shuffle_tensor,tf.reshape(x[idx],[1,x.shape[1]])],axis=0)
    
    return node_shuffle_tensor

def feature_shuffle(x):
    
    list_idx = [i for i in range(x.get_shape().as_list()[1])]
    random.shuffle(list_idx)
    feature_shuffle_tensor = x[:,list_idx[0]]
    flag = 0
    for index in list_idx[1:]:
        if flag == 0:
            feature_shuffle_tensor = tf.concat([tf.reshape(feature_shuffle_tensor,[x.shape[0],1]),\
                                                tf.reshape(x[:,index],[x.shape[0],1])],axis=1)
            flag = 1
        else:
            feature_shuffle_tensor = tf.concat([feature_shuffle_tensor,tf.reshape(x[:,index],\
                                                                                [x.shape[0],1])],axis=1)
    
    return feature_shuffle_tensor
    
def reparameter(args):
    z_mean,z_log_var = args
    u = tf.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(z_log_var / 2) * u
    
    
    
#load network
network_path = '/home/data/'

drug_drug = np.loadtxt(network_path+'mat_drug_drug.txt')
#print 'loaded drug drug', check_symmetric(drug_drug), np.shape(drug_drug)
true_drug = 708 # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database
drug_chemical = np.loadtxt(network_path+'Similarity_Matrix_Drugs.txt')
drug_chemical=drug_chemical[:true_drug,:true_drug]
#print 'loaded drug chemical', check_symmetric(drug_chemical), np.shape(drug_chemical)
drug_disease = np.loadtxt(network_path+'mat_drug_disease.txt')
#print 'loaded drug disease', np.shape(drug_disease)
drug_sideeffect = np.loadtxt(network_path+'mat_drug_se.txt')
#print 'loaded drug sideffect', np.shape(drug_sideeffect)
disease_drug = drug_disease.T
sideeffect_drug = drug_sideeffect.T

protein_protein = np.loadtxt(network_path+'mat_protein_protein.txt')
#print 'loaded protein protein', check_symmetric(protein_protein), np.shape(protein_protein)
protein_sequence = np.loadtxt(network_path+'Similarity_Matrix_Proteins.txt')
#print 'loaded protein sequence', check_symmetric(protein_sequence), np.shape(protein_sequence)
protein_disease = np.loadtxt(network_path+'mat_protein_disease.txt')
#print 'loaded protein disease', np.shape(protein_disease)
disease_protein = protein_disease.T

file1 = open('/home/res.txt','w',encoding='utf-8')
#normalize network for mean pooling aggregation
drug_drug_normalize = row_normalize(drug_drug,True)
drug_chemical_normalize = row_normalize(drug_chemical,True)
drug_disease_normalize = row_normalize(drug_disease,False)
drug_sideeffect_normalize = row_normalize(drug_sideeffect,False)

protein_protein_normalize = row_normalize(protein_protein,True)
protein_sequence_normalize = row_normalize(protein_sequence,True)
protein_disease_normalize = row_normalize(protein_disease,False)

disease_drug_normalize = row_normalize(disease_drug,False)
disease_protein_normalize = row_normalize(disease_protein,False)
sideeffect_drug_normalize = row_normalize(sideeffect_drug,False)



#define computation graph
num_drug = len(drug_drug_normalize)
num_protein = len(protein_protein_normalize)
num_disease = len(disease_protein_normalize)
num_sideeffect = len(sideeffect_drug_normalize)

dim_drug = int(opts.d)
dim_protein = int(opts.d)
dim_disease = int(opts.d)
dim_sideeffect = int(opts.d)
dim_pred = int(opts.k)#512
dim_pass = int(opts.d) #1024




class Model_(object):
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        #inputs
        self.drug_drug = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_drug_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.drug_chemical = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_chemical_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.drug_disease = tf.placeholder(tf.float32, [num_drug, num_disease])
        self.drug_disease_normalize = tf.placeholder(tf.float32, [num_drug, num_disease])

        self.drug_sideeffect = tf.placeholder(tf.float32, [num_drug, num_sideeffect])
        self.drug_sideeffect_normalize = tf.placeholder(tf.float32, [num_drug, num_sideeffect])

        
        self.protein_protein = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_protein_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])

        self.protein_sequence = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_sequence_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])

        self.protein_disease = tf.placeholder(tf.float32, [num_protein, num_disease])
        self.protein_disease_normalize = tf.placeholder(tf.float32, [num_protein, num_disease])
        
        self.disease_drug = tf.placeholder(tf.float32, [num_disease, num_drug])
        self.disease_drug_normalize = tf.placeholder(tf.float32, [num_disease, num_drug])

        self.disease_protein = tf.placeholder(tf.float32, [num_disease, num_protein])
        self.disease_protein_normalize = tf.placeholder(tf.float32, [num_disease, num_protein])

        self.sideeffect_drug = tf.placeholder(tf.float32, [num_sideeffect, num_drug])
        self.sideeffect_drug_normalize = tf.placeholder(tf.float32, [num_sideeffect, num_drug])

        self.drug_protein = tf.placeholder(tf.float32, [num_drug, num_protein])
        self.drug_protein_normalize = tf.placeholder(tf.float32, [num_drug, num_protein])

        self.protein_drug = tf.placeholder(tf.float32, [num_protein, num_drug])
        self.protein_drug_normalize = tf.placeholder(tf.float32, [num_protein, num_drug])

        self.drug_protein_mask = tf.placeholder(tf.float32, [num_drug, num_protein])

        #features
        self.drug_embedding = weight_variable([num_drug,dim_drug])           
        self.protein_embedding = weight_variable([num_protein,dim_protein])             
        self.disease_embedding = weight_variable([num_disease,dim_disease])              
        self.sideeffect_embedding = weight_variable([num_sideeffect,dim_sideeffect])     

        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.drug_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.protein_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.disease_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.sideeffect_embedding))



        #feature passing weights (maybe different types of nodes can use different weights)
        W0 = weight_variable([dim_pass+dim_drug, dim_drug])
        b0 = bias_variable([dim_drug])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0))

        #passing 1 times (can be easily extended to multiple passes)
        drug_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([tf.matmul(self.drug_drug_normalize, a_layer(self.drug_embedding, dim_pass)) + \
            tf.matmul(self.drug_chemical_normalize, a_layer(self.drug_embedding, dim_pass)) + \
            tf.matmul(self.drug_disease_normalize, a_layer(self.disease_embedding, dim_pass)) + \
            tf.matmul(self.drug_sideeffect_normalize, a_layer(self.sideeffect_embedding, dim_pass)) + \
            tf.matmul(self.drug_protein_normalize, a_layer(self.protein_embedding, dim_pass)), \
            self.drug_embedding], axis=1), W0)+b0),dim=1)


        protein_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([tf.matmul(self.protein_protein_normalize, a_layer(self.protein_embedding, dim_pass)) + \
            tf.matmul(self.protein_sequence_normalize, a_layer(self.protein_embedding, dim_pass)) + \
            tf.matmul(self.protein_disease_normalize, a_layer(self.disease_embedding, dim_pass)) + \
            tf.matmul(self.protein_drug_normalize, a_layer(self.drug_embedding, dim_pass)), \
            self.protein_embedding], axis=1), W0)+b0),dim=1)


        disease_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([tf.matmul(self.disease_drug_normalize, a_layer(self.drug_embedding, dim_pass)) + \
            tf.matmul(self.disease_protein_normalize, a_layer(self.protein_embedding, dim_pass)), \
            self.disease_embedding], axis=1), W0)+b0),dim=1)


        sideeffect_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([tf.matmul(self.sideeffect_drug_normalize, a_layer(self.drug_embedding, dim_pass)), \
            self.sideeffect_embedding], axis=1), W0)+b0),dim=1)

        # ===================meta encoder==========================#
        self.meta_drug_drug = tf.nn.l2_normalize(meta_layer(self.drug_protein, self.protein_drug, num_drug), dim=1)
        self.meta_protein_protein = tf.nn.l2_normalize(meta_layer(self.protein_drug, self.drug_protein, num_protein),dim=1)
        # ===================meta encoder===========================#


        self.drug_representation = drug_vector1                      
        self.protein_representation = protein_vector1                
        self.disease_representation = disease_vector1                
        self.sideeffect_representation = sideeffect_vector1          

        # self.drug_representation = tf.transpose(self.drug_representation,(1,0))
        # self.protein_representation = tf.transpose(self.protein_representation,(1,0))
        # self.disease_representation = tf.transpose(self.disease_representation,(1,0))
        # self.sideeffect_representation = tf.transpose(self.sideeffect_representation,(1,0))


        self.global_representation = K.concatenate([self.drug_representation,self.protein_representation],axis=0) 
        # self.global_representation = tf.transpose(self.global_representation,(1,0))      
        
        alpha = 1     #global_loss_weight
        beta = 1.5    #local_loss_weight
        gamma = 0.01  #prior_loss_weight
        
        #====================VAE==========================
        z_mean = Dense(512)(self.global_representation)   
        z_log_var = Dense(512)(self.global_representation)
        z_samples = Lambda(reparameter)([z_mean,z_log_var])  
        
        self.prior_kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        #=================================================
        
        z_shuffle = Lambda(feature_shuffle)(z_samples)    
        
        #===================global=========================
        z_z_1 = Concatenate()([z_samples,z_samples])     
        z_z_2 = Concatenate()([z_samples,z_shuffle])     
        
        z_in = Input(shape=(2220,1024))
        z = z_in
        z = Dense(512,activation='relu')(z)
        z = Dense(512,activation='relu')(z)
        z = Dense(512,activation='relu')(z)
        z = Dense(1,activation='sigmoid')(z)
        GlobalDiscriminator = Model(z_in,z)
        z_z_1_scores = GlobalDiscriminator(z_z_1)
        z_z_2_scores = GlobalDiscriminator(z_z_2)
        self.global_info_loss = -K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))
        
        #===================================================
        
        
        #=====================local=========================

        feature_map_shuffle_drug = Lambda(node_shuffle)(self.drug_representation) 
        feature_map_shuffle_protein = Lambda(node_shuffle)(self.protein_representation) 
        # feature_map_shuffle_disease = Lambda(node_shuffle)(self.disease_representation) 
        # feature_map_shuffle_sideeffect = Lambda(node_shuffle)(self.sideeffect_representation) 

        z_samples_repeat = K.concatenate([z_samples,z_samples],axis=1)  
        z_samples_map = tf.transpose(z_samples_repeat,(1,0))  
        z_f_1_drug = Concatenate()([z_samples_map,tf.transpose(self.drug_representation,(1,0))]) 
        z_f_2_drug = Concatenate()([z_samples_map,tf.transpose(feature_map_shuffle_drug,(1,0))])
        z_f_1_protein = Concatenate()([z_samples_map,tf.transpose(self.protein_representation,(1,0))]) 
        z_f_2_protein = Concatenate()([z_samples_map,tf.transpose(feature_map_shuffle_protein,(1,0))])
        # z_f_1_disease = Concatenate()([z_samples_map,tf.transpose(self.disease_representation,(1,0))]) 
        # z_f_2_disease = Concatenate()([z_samples_map,tf.transpose(feature_map_shuffle_disease,(1,0))])
        # z_f_1_sideeffect = Concatenate()([z_samples_map,tf.transpose(self.sideeffect_representation,(1,0))]) 
        # z_f_2_sideeffect = Concatenate()([z_samples_map,tf.transpose(feature_map_shuffle_sideeffect,(1,0))])


        z_in_drug = Input(shape=(2928,1024))
        z = z_in_drug
        z = Dense(512,activation='relu')(z)
        z = Dense(512,activation='relu')(z)
        z = Dense(512,activation='relu')(z)
        z = Dense(1,activation='sigmoid')(z)
        LocalDiscriminator = Model(z_in_drug,z)
        z_f_1_scores_drug = LocalDiscriminator(tf.transpose(z_f_1_drug,(1,0)))
        z_f_2_scores_drug = LocalDiscriminator(tf.transpose(z_f_2_drug,(1,0)))
        local_info_loss_drug = -K.mean(K.log(z_f_1_scores_drug + 1e-6) + K.log(1 - z_f_2_scores_drug + 1e-6))

        z_in_protein = Input(shape=(3732,1024))
        z = z_in_protein
        z = Dense(512,activation='relu')(z)
        z = Dense(512,activation='relu')(z)
        z = Dense(512,activation='relu')(z)
        z = Dense(1,activation='sigmoid')(z)
        LocalDiscriminator = Model(z_in_protein,z)
        z_f_1_scores_protein = LocalDiscriminator(tf.transpose(z_f_1_protein,(1,0)))
        z_f_2_scores_protein = LocalDiscriminator(tf.transpose(z_f_2_protein,(1,0)))
        local_info_loss_protein = -K.mean(K.log(z_f_1_scores_protein + 1e-6) + K.log(1 - z_f_2_scores_protein + 1e-6))

        # z_in_disease = Input(shape=(17618,1024))
        # z = z_in_disease
        # z = Dense(512,activation='relu')(z)
        # z = Dense(512,activation='relu')(z)
        # z = Dense(512,activation='relu')(z)
        # z = Dense(1,activation='sigmoid')(z)
        # LocalDiscriminator = Model(z_in_disease,z)
        # z_f_1_scores_disease = LocalDiscriminator(tf.transpose(z_f_1_disease,(1,0)))
        # z_f_2_scores_disease = LocalDiscriminator(tf.transpose(z_f_2_disease,(1,0)))
        # local_info_loss_disease = -K.mean(K.log(z_f_1_scores_disease + 1e-6) + K.log(1 - z_f_2_scores_disease + 1e-6))

        # z_in_sideeffect = Input(shape=(16207,1024))
        # z = z_in_sideeffect
        # z = Dense(512,activation='relu')(z)
        # z = Dense(512,activation='relu')(z)
        # z = Dense(512,activation='relu')(z)
        # z = Dense(1,activation='sigmoid')(z)
        # LocalDiscriminator = Model(z_in_sideeffect,z)
        # z_f_1_scores_sideeffect = LocalDiscriminator(tf.transpose(z_f_1_sideeffect,(1,0)))
        # z_f_2_scores_sideeffect = LocalDiscriminator(tf.transpose(z_f_2_sideeffect,(1,0)))
        # local_info_loss_sideeffect = -K.mean(K.log(z_f_1_scores_sideeffect + 1e-6) + K.log(1 - z_f_2_scores_sideeffect + 1e-6))

        self.local_info_loss = (local_info_loss_drug + local_info_loss_protein) / 2
        #==============================================================

        #reconstructing networks
        self.drug_drug_reconstruct = bi_layer(self.drug_representation,self.drug_representation, sym=True, dim_pred=dim_pred)
        self.drug_drug_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_drug_reconstruct-self.drug_drug), (self.drug_drug_reconstruct-self.drug_drug)))

        self.drug_chemical_reconstruct = bi_layer(self.drug_representation,self.drug_representation, sym=True, dim_pred=dim_pred)
        self.drug_chemical_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_chemical_reconstruct-self.drug_chemical), (self.drug_chemical_reconstruct-self.drug_chemical)))


        self.drug_disease_reconstruct = bi_layer(self.drug_representation,self.disease_representation, sym=False, dim_pred=dim_pred)
        self.drug_disease_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_disease_reconstruct-self.drug_disease), (self.drug_disease_reconstruct-self.drug_disease)))


        self.drug_sideeffect_reconstruct = bi_layer(self.drug_representation,self.sideeffect_representation, sym=False, dim_pred=dim_pred)
        self.drug_sideeffect_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_sideeffect_reconstruct-self.drug_sideeffect), (self.drug_sideeffect_reconstruct-self.drug_sideeffect)))


        self.protein_protein_reconstruct = bi_layer(self.protein_representation,self.protein_representation, sym=True, dim_pred=dim_pred)
        self.protein_protein_reconstruct_loss = tf.reduce_sum(tf.multiply((self.protein_protein_reconstruct-self.protein_protein), (self.protein_protein_reconstruct-self.protein_protein)))

        self.protein_sequence_reconstruct = bi_layer(self.protein_representation,self.protein_representation, sym=True, dim_pred=dim_pred)
        self.protein_sequence_reconstruct_loss = tf.reduce_sum(tf.multiply((self.protein_sequence_reconstruct-self.protein_sequence), (self.protein_sequence_reconstruct-self.protein_sequence)))


        self.protein_disease_reconstruct = bi_layer(self.protein_representation,self.disease_representation, sym=False, dim_pred=dim_pred)
        self.protein_disease_reconstruct_loss = tf.reduce_sum(tf.multiply((self.protein_disease_reconstruct-self.protein_disease), (self.protein_disease_reconstruct-self.protein_disease)))


        self.drug_protein_reconstruct = bi_layer(self.drug_representation,self.protein_representation, sym=False, dim_pred=dim_pred)
        tmp = tf.multiply(self.drug_protein_mask, (self.drug_protein_reconstruct-self.drug_protein))
        self.drug_protein_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp))

        # ===================meta decoder===================#
        self.meta_drug_drug_reconstruct = Dense(708,activation = 'relu')(self.meta_drug_drug)
        self.meta_drug_drug_reconstruct_loss = tf.reduce_sum(tf.multiply((drug_drug - self.meta_drug_drug_reconstruct), (drug_drug - self.meta_drug_drug_reconstruct)))
        self.meta_protein_protein_reconstruct = Dense(1512,activation = 'relu')(self.meta_protein_protein)
        self.meta_protein_protein_reconstruct_loss = tf.reduce_sum(tf.multiply((protein_protein - self.meta_protein_protein_reconstruct),(protein_protein - self.meta_protein_protein_reconstruct)))
        # ===================meta decoder===================#

        self.l2_loss = tf.add_n(tf.get_collection("l2_reg"))        

        self.loss = self.drug_protein_reconstruct_loss + 1.0*(self.drug_drug_reconstruct_loss+self.drug_chemical_reconstruct_loss+
                                                            self.drug_disease_reconstruct_loss+self.drug_sideeffect_reconstruct_loss+
                                                            self.protein_protein_reconstruct_loss+self.protein_sequence_reconstruct_loss+
                                                            self.protein_disease_reconstruct_loss) + self.l2_loss + \
                                                            gamma * self.prior_kl_loss + alpha * self.global_info_loss +beta * self.local_info_loss +\
                                                            0.2*self.meta_drug_drug_reconstruct_loss+0.2*self.meta_protein_protein_reconstruct_loss
                                                            
                                                            
                                                            




graph = tf.get_default_graph()
with graph.as_default():
    model = Model_()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss
    dti_loss = model.drug_protein_reconstruct_loss

    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, int(opts.n))
    optimizer = optimize.apply_gradients(zip(gradients, variables))

    eval_pred = model.drug_protein_reconstruct

def train_and_evaluate(DTItrain, DTIvalid, DTItest, graph, verbose=True, num_steps = 4000):
    drug_protein = np.zeros((num_drug,num_protein))
    mask = np.zeros((num_drug,num_protein))     
    
    for ele in DTItrain:
        drug_protein[ele[0],ele[1]] = ele[2]
        mask[ele[0],ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein,False)
    protein_drug_normalize = row_normalize(protein_drug,False)

    lr = 0.001

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        for i in range(num_steps):
            _, tloss, dtiloss, results = sess.run([optimizer,total_loss,dti_loss,eval_pred], \
                                        feed_dict={model.drug_drug:drug_drug, model.drug_drug_normalize:drug_drug_normalize,\
                                        model.drug_chemical:drug_chemical, model.drug_chemical_normalize:drug_chemical_normalize,\
                                        model.drug_disease:drug_disease, model.drug_disease_normalize:drug_disease_normalize,\
                                        model.drug_sideeffect:drug_sideeffect, model.drug_sideeffect_normalize:drug_sideeffect_normalize,\
                                        model.protein_protein:protein_protein, model.protein_protein_normalize:protein_protein_normalize,\
                                        model.protein_sequence:protein_sequence, model.protein_sequence_normalize:protein_sequence_normalize,\
                                        model.protein_disease:protein_disease, model.protein_disease_normalize:protein_disease_normalize,\
                                        model.disease_drug:disease_drug, model.disease_drug_normalize:disease_drug_normalize,\
                                        model.disease_protein:disease_protein, model.disease_protein_normalize:disease_protein_normalize,\
                                        model.sideeffect_drug:sideeffect_drug, model.sideeffect_drug_normalize:sideeffect_drug_normalize,\
                                        model.drug_protein:drug_protein, model.drug_protein_normalize:drug_protein_normalize,\
                                        model.protein_drug:protein_drug, model.protein_drug_normalize:protein_drug_normalize,\
                                        model.drug_protein_mask:mask,\
                                        learning_rate: lr})
            
            #every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
            if i % 25 == 0 and verbose == True:
                print('step',i,'total and dtiloss',tloss, dtiloss)

                pred_list = []
                ground_truth = []
                for ele in DTIvalid:
                    pred_list.append(results[ele[0],ele[1]])
                    ground_truth.append(ele[2])
                valid_auc = roc_auc_score(ground_truth, pred_list)
                valid_aupr = average_precision_score(ground_truth, pred_list)
                if valid_aupr >= best_valid_aupr:
                    best_valid_aupr = valid_aupr
                    best_valid_auc = valid_auc
                    pred_list = []
                    ground_truth = []
                    for ele in DTItest:
                        pred_list.append(results[ele[0],ele[1]])
                        ground_truth.append(ele[2])
                    test_auc = roc_auc_score(ground_truth, pred_list)
                    test_aupr = average_precision_score(ground_truth, pred_list)
                print('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)
    return best_valid_auc, best_valid_aupr, test_auc, test_aupr

test_auc_round = []
test_aupr_round = []
for r in range(10):
    print('sample round',r+1)
    if opts.t == 'o':
        dti_o = np.loadtxt(network_path+'mat_drug_protein.txt')
    else:
        dti_o = np.loadtxt(network_path+'mat_drug_protein_'+opts.t+'.txt')

    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i,j])
            elif int(dti_o[i][j]) == 0:
                whole_negative_index.append([i,j])


    if opts.r == 'ten':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=10*len(whole_positive_index),replace=False)
    elif opts.r == 'all':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=len(whole_negative_index),replace=False)
    else:
        print('wrong positive negative ratio')
        break

    data_set = np.zeros((len(negative_sample_index)+len(whole_positive_index),3),dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1



    if opts.t == 'unique':
        
        whole_positive_index_test = []
        whole_negative_index_test = []
        for i in range(np.shape(dti_o)[0]):
            for j in range(np.shape(dti_o)[1]):
                if int(dti_o[i][j]) == 3:
                    whole_positive_index_test.append([i,j])
                elif int(dti_o[i][j]) == 2:
                    whole_negative_index_test.append([i,j])

        if opts.r == 'ten':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),size=10*len(whole_positive_index_test),replace=False)
        elif opts.r == 'all':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),size=whole_negative_index_test,replace=False)
        else:
            print('wrong positive negative ratio')
            break
        data_set_test = np.zeros((len(negative_sample_index_test)+len(whole_positive_index_test),3),dtype=int)
        count = 0
        for i in whole_positive_index_test:
            data_set_test[count][0] = i[0]
            data_set_test[count][1] = i[1]
            data_set_test[count][2] = 1
            count += 1
        for i in negative_sample_index_test:
            data_set_test[count][0] = whole_negative_index_test[i][0]
            data_set_test[count][1] = whole_negative_index_test[i][1]
            data_set_test[count][2] = 0
            count += 1

        DTItrain = data_set
        DTItest = data_set_test
        rs = np.random.randint(0,1000,1)[0]
        DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=0.05, random_state=rs)
        v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, graph=graph, num_steps=3000)

        test_auc_round.append(t_auc)
        test_aupr_round.append(t_aupr)
        np.savetxt('test_auc', test_auc_round)
        np.savetxt('test_aupr', test_aupr_round)

    else:
        
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0,1000,1)[0]
        kf = StratifiedKFold(data_set[:,2], n_folds=10, shuffle=True, random_state=rs)

        for train_index, test_index in kf:
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=0.05, random_state=rs)

            v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, graph=graph, num_steps=3000)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)

        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        
for i in range(len(test_auc_round)):
    file1.write(str(test_auc_round[i]) + " " + str(test_aupr_round[i]) + "\n")

file1.close()
        #np.savetxt('/home/res/test_auc', test_auc_round)
        #np.savetxt('/home/res/test_aupr', test_aupr_round)





