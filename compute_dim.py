########3
# Computing interpretable dimensions/axes 
# Katrin Erk Fall 2023

import os
from scipy import stats
import numpy as np 
import pandas as pd
import zipfile
import math
import sklearn
import torch
import torch.optim as optim
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import random

######################3
# methods for computing dimensions

####
# seed-based method
# averaging over seed pair vectors
def dimension_seedbased(seeds_pos, seeds_neg, space, paired = False):
    diffvectors = [ ]
    
    for negword, posword in _make_seedpairs(seeds_pos, seeds_neg, paired = paired):
        diffvectors.append(space[posword] - space[negword])

    # average
    dimvec = np.mean(diffvectors, axis = 0)
    return dimvec

####
# fitted, from ratings
#  computing a fitted dimension based on a list of word vectors and matching list of gold ratings.
# feature_dim is dimensionality of the vectors. 
# This fits a dimension using an objective function based on the Jameel and Schockaert idea
def dimension_fitted_fromratings(word_vectors_list, gold_ratings, feature_dim, random_seed = 123):
    torch.manual_seed(random_seed) 

    # we compute: a vector of same dimensionality as our embeddings,
    # and a weight and a bias constant
    feature_vector = torch.randn(feature_dim, requires_grad=True) # dtype=torch.float32)
    weight_constant = torch.randn(1, requires_grad=True) 
    bias_constant = torch.randn(1, requires_grad=True)    

    
    optimizer = optim.Adam([feature_vector, weight_constant, bias_constant], lr=0.01)
    # optimizer = optim.SGD([feature_vector, weight_constant, bias_constant], lr=0.001)

    # Number of optimization steps
    num_steps = 1000

    losses = []

    # Gradient clipping threshold
    max_norm = 1.0  

    for step in range(num_steps):
        total_loss = 0

        # for i in range(len(X_train)):
        # 	word_embedding = torch.tensor(X_train[i]) 
        # 	gold_rating = y_train[i]

        for i in range(len(word_vectors_list)):
            word_embedding = torch.tensor(word_vectors_list[i])
            gold_rating = gold_ratings[i]

            dot_product = torch.dot(word_embedding, feature_vector)
            weighted_gold = gold_rating * weight_constant
            loss = ((dot_product - weighted_gold - bias_constant) ** 2)

            total_loss += loss

        # Average loss over all words in X_train or in the whole human dic (2,801 annotated single words in total)
        avg_loss = total_loss / len(word_vectors_list)

        avg_loss.backward()

        # Compute the gradient norms and monitor them during training

        torch.nn.utils.clip_grad_norm_([feature_vector, weight_constant, bias_constant], max_norm)
        optimizer.step()

        losses.append(avg_loss.item())
    
    return (feature_vector.detach().numpy(), weight_constant.item(), bias_constant.item())

####
# fitted, from ratings and seed words
# 
# combined method: fitted dimension with seeds
# thisdata_vectors: vectors for category words
# thisdata_gold: gold ratings for category words
# feature_dim: dimensionality
# pos_seedwords: list of positive seedwords
# neg_seedwords: list of negative seedwords
# word_vectors: mapping word-> vector
# offset: synthetic rating for seed words should be this far beyond
#   the rating of the most positive/most negative word
# jitter: if true: add a bit of random variation to seed word ratings
def dimension_fitted_fromratings_seedwords(data_vectors, data_gold, feature_dim, 
                                           pos_seedwords, neg_seedwords, word_vectors,
                                           offset = 0.5, jitter = False, random_seed = 123):

    random.seed(random_seed)
    # adding seed words
    
    lowvalue = min(data_gold) - offset
    highvalue = max(data_gold) + offset

    thisdata_vectors = data_vectors.copy()
    thisdata_gold = data_gold.copy()
        
    for seed in pos_seedwords:
        thisdata_vectors.append( word_vectors [ seed ])
        j = random.uniform(0.001, 0.005) if jitter else 0
        thisdata_gold.append( highvalue + j)

    for seed in neg_seedwords:
        thisdata_vectors.append( word_vectors [ seed ])
        j = random.uniform(0.001, 0.005) if jitter else 0
        thisdata_gold.append( lowvalue - j)
            
            
    return dimension_fitted_fromratings(thisdata_vectors, thisdata_gold, feature_dim, random_seed = random_seed)

###########
# fitted dimension, based on ratings and seed dimensions
# 
# Weight/bias defined as trainable variables to optimize during backpropagation along with the feature vector
#
# parameters:
# - word vectors list: list of vectors for the words in our category
# - gold ratings: gold ratings on the dimension of interest, 
#    in the same order as the word vectors list
# - feature_dim: dimensionality of the vectors
# - random seed
#
# returns: 
# - computed vector for the ideal dimension
# - weight and bias such that vector * idealvec \approx weight * goldrating + bias
#
def dimension_fitted_fromratings_seeddims(word_vectors_list, gold_ratings, feature_dim,
                                          pos_seedwords, neg_seedwords, word_vectors,
                                          do_average = True,
                                          alpha = 0.1, random_seed = 123, paired = False):

    diffvectors = [ ]

    for negword, posword in _make_seedpairs(pos_seedwords, neg_seedwords, paired = paired):
        diffvectors.append(word_vectors[posword] - word_vectors[negword])

    if do_average:
        seed_dims = [ torch.from_numpy(np.mean(diffvectors, axis = 0)) ]
    else:
        seed_dims = [ torch.from_numpy(v) for v in diffvectors]
        
    torch.manual_seed(random_seed) 

    # we compute: a vector of same dimensionality as our embeddings,
    # and a weight and a bias constant
    feature_vector = torch.randn(feature_dim, requires_grad=True) # dtype=torch.float32)
    weight_constant = torch.randn(1, requires_grad=True) 
    bias_constant = torch.randn(1, requires_grad=True)    

    
    optimizer = optim.Adam([feature_vector, weight_constant, bias_constant], lr=0.01)
    # optimizer = optim.SGD([feature_vector, weight_constant, bias_constant], lr=0.001)

    # Number of optimization steps
    num_steps = 1000

    losses = []
    
    criterion2 = torch.nn.CosineEmbeddingLoss()

    # Gradient clipping threshold
    max_norm = 1.0  

    for step in range(num_steps):
        total_loss1 = 0

        # for i in range(len(X_train)):
        # 	word_embedding = torch.tensor(X_train[i]) 
        # 	gold_rating = y_train[i]

        for i in range(len(word_vectors_list)):
            word_embedding = torch.tensor(word_vectors_list[i])
            gold_rating = gold_ratings[i]

            dot_product = torch.dot(word_embedding, feature_vector)
            weighted_gold = gold_rating * weight_constant
            loss1 = ((dot_product - weighted_gold - bias_constant) ** 2)

            total_loss1 += loss1
            
        
        avg_loss1 = total_loss1 / len(word_vectors_list)
        loss2 = sum([criterion2( feature_vector, d, torch.tensor(1.0)) for d in seed_dims]) / len(seed_dims)
        total_loss = alpha*avg_loss1 + (1-alpha) * loss2

        total_loss.backward()

        # Compute the gradient norms and monitor them during training

        # feature_vector_grad_norm = torch.norm(feature_vector.grad)
        # print(f"Step {step+1}, Feature Vector Gradient Norm: {feature_vector_grad_norm.item()}")

        torch.nn.utils.clip_grad_norm_([feature_vector, weight_constant, bias_constant], max_norm)
        optimizer.step()

        losses.append(total_loss.item())

    
    return (feature_vector.detach().numpy(), weight_constant.item(), bias_constant.item())

####
# fitted, from ratings and seed words and seed dimensions
# 
# combined method
# data_vectors: vectors for category words
# data_gold: gold ratings for category words
# feature_dim: dimensionality
# pos_seedwords: list of positive seedwords
# neg_seedwords: list of negative seedwords
# word_vectors: mapping word-> vector
# offset: synthetic rating for seed words should be this far beyond
#   the rating of the most positive/most negative word
# jitter: if true: add a bit of random variation to seed word ratings
def dimension_fitted_fromratings_combined(data_vectors, data_gold, feature_dim, 
                                          pos_seedwords, neg_seedwords, word_vectors,
                                          offset = 0.5, jitter = False, do_average = True,
                                          alpha = 0.1, random_seed = 123, paired = False):
    # adding seed words
    
    lowvalue = min(data_gold) - offset
    highvalue = max(data_gold) + offset

    thisdata_vectors = data_vectors.copy()
    thisdata_gold = data_gold.copy()
        
    for seed in pos_seedwords:
        thisdata_vectors.append( word_vectors [ seed ])
        j = random.uniform(0.001, 0.005) if jitter else 0
        thisdata_gold.append( highvalue + j)

    for seed in neg_seedwords:
        thisdata_vectors.append( word_vectors [ seed ])
        j = random.uniform(0.001, 0.005) if jitter else 0
        thisdata_gold.append( lowvalue - j)
            
            
    return dimension_fitted_fromratings_seeddims(thisdata_vectors, thisdata_gold, feature_dim,
                                                 pos_seedwords, neg_seedwords, word_vectors,
                                                 do_average = do_average, alpha = alpha, random_seed = random_seed, paired = paired)

###########
# fitted dimension, based on ratings and seed dimensions
# 
# Weight/bias defined as trainable variables to optimize during backpropagation along with the feature vector
#
# parameters:
# - word vectors list: list of vectors for the words in our category
# - gold ratings: gold ratings on the dimension of interest, 
#    in the same order as the word vectors list
# - feature_dim: dimensionality of the vectors
# - random seed
#
# returns: 
# - computed vector for the ideal dimension
# - weight and bias such that vector * idealvec \approx weight * goldrating + bias
#
def dimension_fitted_fromratings_attseeddims(word_vectors_list, gold_ratings, feature_dim,
                                          pos_seedwords, neg_seedwords, word_vectors,
                                          alpha = 0.1, random_seed = 123, paired = False):

    diffvectors = [ ]
    
    for negword, posword in _make_seedpairs(pos_seedwords, neg_seedwords, paired = paired):
        diffvectors.append(word_vectors[posword] - word_vectors[negword])

    seed_dims = [ torch.from_numpy(v) for v in diffvectors]
        
    torch.manual_seed(random_seed) 

    # we compute: a vector of same dimensionality as our embeddings,
    # a weight and a bias constant,
    # and an attention weight for each dimension
    feature_vector = torch.randn(feature_dim, requires_grad=True) # dtype=torch.float32)
    weight_constant = torch.randn(1, requires_grad=True) 
    bias_constant = torch.randn(1, requires_grad=True)
    seed_attn_raw = torch.randn(len(seed_dims), requires_grad = True)
    
    optimizer = optim.Adam([feature_vector, weight_constant, bias_constant], lr=0.01)

    # Number of optimization steps
    num_steps = 1000

    losses = []
    
    criterion2 = torch.nn.CosineEmbeddingLoss()

    # Gradient clipping threshold
    max_norm = 1.0  

    for step in range(num_steps):
        total_loss1 = 0

        # for i in range(len(X_train)):
        # 	word_embedding = torch.tensor(X_train[i]) 
        # 	gold_rating = y_train[i]

        for i in range(len(word_vectors_list)):
            word_embedding = torch.tensor(word_vectors_list[i])
            gold_rating = gold_ratings[i]

            dot_product = torch.dot(word_embedding, feature_vector)
            weighted_gold = gold_rating * weight_constant
            loss1 = ((dot_product - weighted_gold - bias_constant) ** 2)

            total_loss1 += loss1

        # loss 1: from match to human ratings. average over number of items we rated.
        avg_loss1 = total_loss1 / len(word_vectors_list)

        # loss 2: from similarity with seed dimensions, which have weights
        # normalize weights
        attn_weights = torch.nn.functional.softmax(seed_attn_raw, dim = 0)
        # cosine distances between feature vector and all seed dimensions
        cs = torch.Tensor([criterion2( feature_vector, d, torch.tensor(1.0)) for d in seed_dims])
        # loss 2 is normalized weights DOT cosine distances
        loss2 = torch.dot(attn_weights, cs)
        
        # combine the two losses with weight alpha
        total_loss = alpha*avg_loss1 + (1-alpha) * loss2

        total_loss.backward()

        # Compute the gradient norms and monitor them during training

        # feature_vector_grad_norm = torch.norm(feature_vector.grad)
        # print(f"Step {step+1}, Feature Vector Gradient Norm: {feature_vector_grad_norm.item()}")

        torch.nn.utils.clip_grad_norm_([feature_vector, weight_constant, bias_constant], max_norm)
        optimizer.step()

        losses.append(total_loss.item())

    attn_weights = torch.nn.functional.softmax(seed_attn_raw, dim = 0)
    
    return (feature_vector.detach().numpy(), weight_constant.item(), bias_constant.item(), attn_weights.detach().numpy())

def dimension_fitted_fromratings_attcombined(data_vectors, data_gold, feature_dim, 
                                          pos_seedwords, neg_seedwords, word_vectors,
                                          offset = 0.5, jitter = False, 
                                          alpha = 0.1, random_seed = 123, paired = False):
    # adding seed words
    
    lowvalue = min(data_gold) - offset
    highvalue = max(data_gold) + offset

    thisdata_vectors = data_vectors.copy()
    thisdata_gold = data_gold.copy()
        
    for seed in pos_seedwords:
        thisdata_vectors.append( word_vectors [ seed ])
        j = random.uniform(0.001, 0.005) if jitter else 0
        thisdata_gold.append( highvalue + j)

    for seed in neg_seedwords:
        thisdata_vectors.append( word_vectors [ seed ])
        j = random.uniform(0.001, 0.005) if jitter else 0
        thisdata_gold.append( lowvalue - j)
            
            
    return dimension_fitted_fromratings_attseeddims(thisdata_vectors, thisdata_gold, feature_dim,
                                                 pos_seedwords, neg_seedwords, word_vectors,
                                                 alpha = alpha, random_seed = random_seed, paired = paired)

###################################
#########
# predicting ratings on a dimension

# ...
# when we only have the dimension:
# vector scalar projection
def predict_scalarproj(veclist, dimension):
    dir_veclen = math.sqrt(np.dot(dimension, dimension))
    return [np.dot(v, dimension) / dir_veclen for v in veclist]

# ... when we have the dimension,
# as well as weight and bias
# such that scalar_projection = weight * prediction + bias
def predict_coord_fromline(veclist, dimension, weight, bias):
    return [(np.dot(v, dimension) - bias) / weight for v in veclist]

# predict a rating given scalar projection, and given
# weight, and bias fitted to produce actuall ratings
def predict_coord_fromtrain(train_veclist, train_ratings, dimension, test_veclist):
    train_predict = predict_scalarproj( train_veclist, dimension)
    weight, bias = fit_dimension_coef(train_ratings, train_predict)
    return predict_coord_fromline(test_veclist, dimension, weight, bias)

############
# computing weight and bias
# for direct prediction of ratings
#  fitted dimensions come with a weight and bias for prediction. 
# we can compute those also for seed-based dimensions
# to make predictions on the same order of magnitude as the original ratings.
# We need that in order to do a Mean Squared Error (MSE) evaluation.
# 
# This function uses linear regression to compute weight and bias. formula:
#
# model_rating ~ weight * gold_rating + bias
#
# formulated this way round to match the formulation in the objective function
# of the fitted dimensions model
def fit_dimension_coef(gold_ratings, model_ratings):
    result = stats.linregress(gold_ratings, model_ratings)
    
    return (result.slope, result.intercept)


#------------
# helper functions: making pairs of seeds
def _make_seedpairs(seeds_pos, seeds_neg, paired = False):
    if paired:
        return list(zip(seeds_neg, seeds_pos))
    else:
        pairs = [ ]
    
        for negword in seeds_neg:
            for posword in seeds_pos:
                pairs.append( (negword, posword) )

        return pairs
