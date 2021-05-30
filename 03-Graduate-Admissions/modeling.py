# import python packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import statsmodels 
import statsmodels.api as sm
import statsmodels.formula.api as smf

# import sklearn linear models 

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

%matplotlib inline


# a formula to calculate adjusted r squared

def adjusted_r_suared(r_squared, num_samples, num_regressors):
    return 1 - ((1-r_squared)*(num_samples - 1) / (num_samples - num_regressors - 1))


""" This code is imported from regularization class"""
def model_expreriment(num_iter = 5, models = ['ols', 'ridge', 'lasso'], complexity = 'simple'):
    
    x_axis = np.arange(num_iter)
#     y_ols_test = []
    y_ols_train = []
#     y_lasso_test = []
    y_lasso_train = []
#     y_ridge_test = []
    y_ridge_train = []
    sample_models = {}
    for i in range(num_iter):
        
        if complexity == 'simple':
            ## split train_test 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        elif complexity == 'polynomial':
            ## test-train split
            X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size = 0.2)


        ## Standard scale mean = 0, variance = 1
        sd = StandardScaler()

        sd.fit(X_train)

        X_train = sd.transform(X_train)

#         X_test = sd.transform(X_test)

        ## Be careful about the leakage

        ## Vanilla model
        if 'ols' in models:
            lr = LinearRegression()

            lr.fit(X_train, y_train)
            
            sample_models['ols'] = lr

#             test_score = lr.score(X_test, y_test)
            train_score = lr.score(X_train, y_train)

            y_ols_train.append(train_score)

#           print('test score OLS is %.2f and train score is %.2f'%(test_score, train_score))
            print('train score is %.2f'%(train_score))

        if 'ridge' in models:
            ## Ridge in the simple setting
            ridge = Ridge(alpha = 10, max_iter= 10000)
            ridge.fit(X_train, y_train)
            sample_models['ridge'] = ridge
            y_ridge_train.append(ridge.score(X_train, y_train))
    #         print('test score Ridge is %.2f and train score is %.2f'%(ridge.score(X_test, y_test),
    #                                                             ridge.score(X_train, y_train)))
            print('train score is %.2f'%(ridge.score(X_train, y_train)))

        if 'lasso' in models:
            ## Lasso in the simple setting
            lasso = Lasso(alpha = 10, max_iter= 10000)

            lasso.fit(X_train, y_train)
            
            sample_models['lasso'] = lasso
            
            y_lasso_train.append(lasso.score(X_train, y_train))
    #         print('test score Lasso is %.2f and train score is %.2f'%(lasso.score(X_test, y_test),
    #                                                             lasso.score(X_train, y_train)))
            print('train score is %.2f'%(lasso.score(X_train, y_train)))

        i+=1
    if 'ols' in models:
        plt.plot(y_ols_train, label = 'ols')
    if 'ridge' in models:
        plt.plot(y_ridge_train, label = 'ridge')
    if 'lasso' in models:
        plt.plot(y_lasso_train, label = 'lasso')
    plt.ylim([0,1])
    plt.ylabel('R2 test score')
    plt.xlabel('number of iterations')
    plt.legend()
    return sample_models, y_ols_train, y_ridge_train, y_lasso_train
