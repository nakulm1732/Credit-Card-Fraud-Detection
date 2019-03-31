# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 01:26:01 2019

@author: nakul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel('CreditData_RareEvent.xlsx')
df.head()
df.shape
attribute_map = {
        'age':['I', (19, 120)],
        'amount':['I', (0, 20000)],
        'checking':['N',(1,2,3,4)],
        'coapp':['N',(1,2,3)],
        'depends':['B',(1,2)],
        'duration':['I',(1,72)],
        'employed':['N',(1,2,3,4,5)],
        'existcr':['N', (1,2,3,4)],
        'foreign':['B', (1,2)],
        'history':['N',(0,1,2,3,4)],
        'housing':['N',(1,2,3)],
        'installp':['N',(1,2,3,4)],
        'job':['N',(1,2,3,4)],
        'marital':['N', (1,2,3,4)],
        'other':['N',(1,2,3)],
        'property':['N',(1,2,3,4)],
        'resident':['N',(1,2,3,4)],
        'savings':['N',(1,2,3,4,5)],
        'good_bad':['B',('good','bad')],
        'telephon':['B',(1,2)]}

df.columns
from AdvancedAnalytics import ReplaceImputeEncode, NeuralNetwork
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
drop=False, display=True)
encoded_df = rie.fit_transform(df)
Y = np.asarray(encoded_df['good_bad'])
X = np.asarray(encoded_df.drop('good_bad', axis=1))

fp_cost = np.array(df['amount'])
fn_cost = np.array(0.1*df['amount'])

c_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3, 1e+64]
best_c = 0
max_f = 0
for c in c_list:
    lgr = LogisticRegression(C=c, tol=1e-16)
    lgr_10 = cross_val_score(lgr, X, Y, scoring='f1', cv=10)
    mean = lgr_10.mean()
    if mean > max_f:
        max_f = mean
        best_c = c
        best_lgr = lgr
        
print("\nLogistic Regression Model using Entire Dataset and C = ",best_c)
from AdvancedAnalytics import logreg, calculate
from AdvancedAnalytics import DecisionTree
best_lgr.fit(X,Y)
logreg.display_binary_metrics(best_lgr, X, Y)
loss,conf_mat = calculate.binary_loss(Y,best_lgr.predict(X),\
fp_cost,fn_cost)

#Use decisionTrees now
c_list = [3,5,7,9,11,13,15,17,19]
best_c = 0
max_f = 0
for c in c_list:
    lgr = DecisionTreeClassifier(max_depth =c)
    lgr_10 = cross_val_score(lgr, X, Y, scoring='f1', cv=10)
    mean = lgr_10.mean()
    if mean > max_f:
        max_f = mean
        best_c = c
        best_lgr = lgr
        
print("\nLogistic Regression Model using Entire Dataset and C = ",best_c)
from AdvancedAnalytics import logreg, calculate
best_lgr.fit(X,Y)
DecisionTree.display_binary_metrics(best_lgr, X, Y)
loss,conf_mat = calculate.binary_loss(Y,best_lgr.predict(X),\
fp_cost,fn_cost)
        
np.random.seed(12345)
max_seed = 2**16 - 1
rand_val = np.random.randint(1, high=max_seed, size=20)
# Ratios of Majority:Minority Events
ratio = [ '50:50', '60:40', '70:30', '80:20', '90:10' ]
# Dictionaries contains number of minority and majority
# events in each ratio sample where n_majority = ratio x n_minority
rus_ratio = ({0:500, 1:500}, {0:500, 1:750}, {0:500, 1:1167}, \
{0:500, 1:2000}, {0:500, 1:4500})

# Best model is one that minimizes the loss
import math
from imblearn.under_sampling import RandomUnderSampler
c_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 1e+64]
min_loss = 1e64
best_ratio = 0
for k in range(len(rus_ratio)):
    print("\nLogistic Regression Model using " + ratio[k] + " RUS")
    best_c = 0
    min_loss_c = 1e64
    for j in range(len(c_list)):
        c = c_list[j]
        fn_loss = np.zeros(len(rand_val))
        fp_loss = np.zeros(len(rand_val))
        misc = np.zeros(len(rand_val))
        for i in range(len(rand_val)):
            rus = RandomUnderSampler(ratio=rus_ratio[k], random_state=rand_val[i], return_indices=False, replacement=False)
            X_rus, Y_rus = rus.fit_sample(X, Y)
            lgr = LogisticRegression(C=c, tol=1e-16, solver='lbfgs', max_iter=1000)
            lgr.fit(X_rus, Y_rus)
            loss, conf_mat = calculate.binary_loss(Y, lgr.predict(X), fp_cost, fn_cost, display=False)
            fn_loss[i] = loss[0]
            fp_loss[i] = loss[1]
            misc[i] = (conf_mat[1] + conf_mat[2])/Y.shape[0]
        avg_misc = np.average(misc)
        t_loss = fp_loss+fn_loss
        avg_loss = np.average(t_loss)
        if avg_loss < min_loss_c:
            min_loss_c = avg_loss
            se_loss_c = np.std(t_loss)/math.sqrt(len(rand_val))
            best_c = c
            misc_c = avg_misc
            fn_avg_loss = np.average(fn_loss)
            fp_avg_loss = np.average(fp_loss)
    if min_loss_c < min_loss:
        min_loss = min_loss_c
        se_loss = se_loss_c
        best_ratio = k
        best_reg = best_c
    print("{:.<23s}{:12.2E}".format("Best C", best_c))
    print("{:.<23s}{:12.4f}".format("Misclassification Rate",misc_c))
    print("{:.<23s} ${:10,.0f}".format("False Negative Loss",fn_avg_loss))
    print("{:.<23s} ${:10,.0f}".format("False Positive Loss",fp_avg_loss))
    print("{:.<23s} ${:10,.0f}{:5s}${:<,.0f}".format("Total Loss", \
          min_loss_c, " +/- ", se_loss_c))
print("")
print("{:.<23s}{:>12s}".format("Best RUS Ratio", ratio[best_ratio]))
print("{:.<23s}{:12.2E}".format("Best C", best_reg))
print("{:.<23s} ${:10,.0f}{:5s}${:<,.0f}".format("Lowest Loss", \
min_loss, " +/-", se_loss))
#Ensemble Modelling
n_obs = len(Y)
n_rand = 100
predicted_prob = np.zeros((n_obs,n_rand))
avg_prob = np.zeros(n_obs)
# Setup 100 random number seeds for use in creating random samples
np.random.seed(12345)
max_seed = 2**16 - 1
rand_value = np.random.randint(1, high=max_seed, size=n_rand)
# Model 100 random samples, each with a 70:30 ratio
for i in range(len(rand_value)):
    rus = RandomUnderSampler(ratio=rus_ratio[best_ratio], \
                             random_state=rand_value[i], return_indices=False, replacement=False)
    X_rus, Y_rus = rus.fit_sample(X, Y)
    lgr = LogisticRegression(C=best_c, tol=1e-16, solver='lbfgs', max_iter=1000)
    lgr.fit(X_rus, Y_rus)
    predicted_prob[0:n_obs, i] = lgr.predict_proba(X)[0:n_obs, 0]
    
for i in range(n_obs):
    avg_prob[i] = np.mean(predicted_prob[i,0:n_rand])
# Set y_pred equal to the predicted classification
y_pred = avg_prob[0:n_obs] < 0.5
y_pred.astype(np.int)
# Calculate loss from using the ensemble predictions
print("\nEnsemble Estimates based on averaging",len(rand_value), "Models")
loss, conf_mat = calculate.binary_loss(Y, y_pred, fp_cost, fn_cost)

#Start Decision Tree from here
#Use decisionTrees now
c_list = [3,5,7,9,11,13,15,17,19]
best_c = 0
max_f = 0
for c in c_list:
    lgr = DecisionTreeClassifier(max_depth =c)
    lgr_10 = cross_val_score(lgr, X, Y, scoring='f1', cv=10)
    mean = lgr_10.mean()
    if mean > max_f:
        max_f = mean
        best_c = c
        best_lgr = lgr
        
print("\nDecision Tree Model using Entire Dataset and C = ",best_c)
from AdvancedAnalytics import logreg, calculate
best_lgr.fit(X,Y)
DecisionTree.display_binary_metrics(best_lgr, X, Y)
loss,conf_mat = calculate.binary_loss(Y,best_lgr.predict(X),\
fp_cost,fn_cost)
        
np.random.seed(12345)
max_seed = 2**16 - 1
rand_val = np.random.randint(1, high=max_seed, size=20)
# Ratios of Majority:Minority Events
ratio = [ '50:50', '60:40', '70:30', '80:20', '90:10' ]
# Dictionaries contains number of minority and majority
# events in each ratio sample where n_majority = ratio x n_minority
rus_ratio = ({0:500, 1:500}, {0:500, 1:750}, {0:500, 1:1167}, \
{0:500, 1:2000}, {0:500, 1:4500})

# Best model is one that minimizes the loss
import math
from imblearn.under_sampling import RandomUnderSampler
c_list = [3,5,7,9,11,13,15,17,19]
min_loss = 1e64
best_ratio = 0
for k in range(len(rus_ratio)):
    print("\nDecision Tree Model using " + ratio[k] + " RUS")
    best_c = 0
    min_loss_c = 1e64
    for j in range(len(c_list)):
        c = c_list[j]
        fn_loss = np.zeros(len(rand_val))
        fp_loss = np.zeros(len(rand_val))
        misc = np.zeros(len(rand_val))
        for i in range(len(rand_val)):
            rus = RandomUnderSampler(ratio=rus_ratio[k], random_state=rand_val[i], return_indices=False, replacement=False)
            X_rus, Y_rus = rus.fit_sample(X, Y)
            lgr = DecisionTreeClassifier(max_depth = c)
            lgr.fit(X_rus, Y_rus)
            loss, conf_mat = calculate.binary_loss(Y, lgr.predict(X), fp_cost, fn_cost, display=False)
            fn_loss[i] = loss[0]
            fp_loss[i] = loss[1]
            misc[i] = (conf_mat[1] + conf_mat[2])/Y.shape[0]
        avg_misc = np.average(misc)
        t_loss = fp_loss+fn_loss
        avg_loss = np.average(t_loss)
        if avg_loss < min_loss_c:
            min_loss_c = avg_loss
            se_loss_c = np.std(t_loss)/math.sqrt(len(rand_val))
            best_c = c
            misc_c = avg_misc
            fn_avg_loss = np.average(fn_loss)
            fp_avg_loss = np.average(fp_loss)
    if min_loss_c < min_loss:
        min_loss = min_loss_c
        se_loss = se_loss_c
        best_ratio = k
        best_reg = best_c
    print("{:.<23s}{:12.2E}".format("Best C", best_c))
    print("{:.<23s}{:12.4f}".format("Misclassification Rate",misc_c))
    print("{:.<23s} ${:10,.0f}".format("False Negative Loss",fn_avg_loss))
    print("{:.<23s} ${:10,.0f}".format("False Positive Loss",fp_avg_loss))
    print("{:.<23s} ${:10,.0f}{:5s}${:<,.0f}".format("Total Loss", \
          min_loss_c, " +/- ", se_loss_c))
print("")
print("{:.<23s}{:>12s}".format("Best RUS Ratio", ratio[best_ratio]))
print("{:.<23s}{:12.2E}".format("Best C", best_reg))
print("{:.<23s} ${:10,.0f}{:5s}${:<,.0f}".format("Lowest Loss", \
min_loss, " +/-", se_loss))
#Ensemble Modelling
n_obs = len(Y)
n_obs
n_rand = 100
predicted_prob = np.zeros((n_obs,n_rand))
avg_prob = np.zeros(n_obs)
# Setup 100 random number seeds for use in creating random samples
np.random.seed(12345)
max_seed = 2**16 - 1
rand_value = np.random.randint(1, high=max_seed, size=n_rand)
# Model 100 random samples, each with a 70:30 ratio
for i in range(len(rand_value)):
    rus = RandomUnderSampler(ratio=rus_ratio[best_ratio], \
                             random_state=rand_value[i], return_indices=False, replacement=False)
    X_rus, Y_rus = rus.fit_sample(X, Y)
    lgr = DecisionTreeClassifier(max_depth=best_c)
    lgr.fit(X_rus, Y_rus)
    predicted_prob[0:n_obs, i] = lgr.predict_proba(X)[0:n_obs, 0]
    
for i in range(n_obs):
    avg_prob[i] = np.mean(predicted_prob[i,0:n_rand])
# Set y_pred equal to the predicted classification
y_pred = avg_prob[0:n_obs] < 0.5
y_pred.astype(np.int)
# Calculate loss from using the ensemble predictions
print("\nEnsemble Estimates based on averaging",len(rand_value), "Models")
loss, conf_mat = calculate.binary_loss(Y, y_pred, fp_cost, fn_cost)


