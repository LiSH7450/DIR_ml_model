# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC 
from xgboost import XGBClassifier  
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline


df_Most = pd.read_csv('Most.txt', sep='\t')
df_Moderate = pd.read_csv('Moderate.txt',sep='\t')
df_Less = pd.read_csv('Less.txt', sep='\t')
df_Non = pd.read_csv('Non.txt',sep='\t')

positive = df_Most
positive['tag']=1
negative = df_Non
negative['tag']=0
drug_all = pd.concat([positive,negative],axis=0)
X = drug_all.iloc[:,2:779].values
y = drug_all.iloc[:,779].values

minmax = preprocessing.MinMaxScaler()
X_minmax = minmax.fit_transform(X)

sec_scr = []
for l in range(0,776):
    if X_minmax[:,l].var()<=0.001:
        sec_scr.append(l)

X = drug_all.iloc[:,2:779] 

m = list(X.columns)
x = []
for i in sec_scr:
    temp = m[i]
    x.append(temp)
x.append("D164")
X = X.drop(x, axis = 1)

def RF_permutation_test(X,y,param):
    'label_permutation'
    df_predict_RF_permu_Y = pd.DataFrame(y,columns=[0])
    df_permu_Y_RF = pd.DataFrame(y,columns=[0])
    for r in range(1000):
        y_permutation = np.random.permutation(y)
        kfold = StratifiedKFold(n_splits=5, shuffle = True,random_state= r )
        model = RandomForestClassifier(max_depth = param['max_depth'], min_samples_split = param['min_samples_split'], n_estimators = param['n_estimators'],n_jobs=-1)
        y_predict = cross_val_predict(model,X,y_permutation,cv=kfold)
        df_predict_temp = pd.DataFrame(y_predict,columns = [r])
        df_predict_RF_permu_Y = pd.concat([df_predict_RF_permu_Y,df_predict_temp],axis = 1) 
        df_permutation_temp = pd.DataFrame(y_permutation,columns = [r])    
        df_permu_Y_RF = pd.concat([df_permu_Y_RF,df_permutation_temp],axis = 1) 
           
    df_predict_RF_permu_Y.to_csv('df_predict_RF_permu_Y.csv')
    df_permu_Y_RF.to_csv('df_permu_Y_RF.csv')
       
    'feature_permutaion'
    index_ls = []
    for index,row in X.iteritems():
        index_ls.append(index)
    X_permutation = pd.DataFrame()    
    for index in index_ls:
        X_permutation[index] = np.random.permutation(X[index])  
        
    df_predict_RF_permu_X = pd.DataFrame(y,columns=[0])
    for r in range(1000):
        kfold = StratifiedKFold(n_splits=5, shuffle = True,random_state= r )
        model = RandomForestClassifier(max_depth = param['max_depth'], min_samples_split = param['min_samples_split'], n_estimators = param['n_estimators'],n_jobs=-1)
        y_predict = cross_val_predict(model,X_permutation,y,cv=kfold)
        df_predict_temp = pd.DataFrame(y_predict,columns = [r])
        df_predict_RF_permu_X = pd.concat([df_predict_RF_permu_X,df_predict_temp],axis = 1) 
        
    df_predict_RF_permu_X.to_csv('df_predict_RF_permu_X.csv')
    X_permutation.to_csv('df_permu_X_RF.csv')
    

def RF_gridsearch_result(X,y):
    'GridSearch'
    kfold = StratifiedKFold(n_splits=5, shuffle = True,random_state=7)
    param_grid = {'n_estimators':range(10,300,10),'max_depth':range(3,21,2), 'min_samples_split':range(10,210,20)}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring='roc_auc',n_jobs = -1,cv=kfold)
    grid_result = grid_search.fit(X,y)
    print("RF_Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
    
    'result'
    df_predict_RF = pd.DataFrame(y,columns=[0])
    for r in range(1000):
        kfold = StratifiedKFold(n_splits=5, shuffle = True,random_state= r )
        model = RandomForestClassifier(max_depth = grid_search.best_params_['max_depth'], min_samples_split = grid_search.best_params_['min_samples_split'], n_estimators = grid_search.best_params_['n_estimators'],n_jobs=-1)
        y_predict = cross_val_predict(model,X,y,cv=kfold)
        df_predict_temp = pd.DataFrame(y_predict,columns = [r])
        df_predict_RF = pd.concat([df_predict_RF,df_predict_temp],axis = 1) 
        
    df_predict_RF.to_csv('df_predict_RF.csv')
    return grid_search.best_params_


def XGB_gridsearch_result(X,y):
    'GridSearch'
    kfold = StratifiedKFold(n_splits=5, shuffle = True,random_state=7)
    learning_rate = [0.0001,0.001,0.01,0.1,0.2,0.3]
    gamma = [1, 0.1, 0.01, 0.001]
    eta = np.arange(0.01,0.2,0.01)
    max_depth = range(3,10,1) 
    model = XGBClassifier()  
    param_grid = dict(learning_rate = learning_rate,gamma = gamma,eta = eta,max_depth = max_depth)
    grid_search = GridSearchCV(model,param_grid,scoring = 'roc_auc',n_jobs = -1,cv = kfold)
    
    grid_result = grid_search.fit(X, y) 
    print("XGB_Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
    
    'Result'
    df_predict_XGB = pd.DataFrame(y,columns=[0])
    for r in range(1000):
        kfold = StratifiedKFold(n_splits=5, shuffle = True,random_state= r )
        model = XGBClassifier(eta=grid_search.best_params_['eta'], gamma= grid_search.best_params_['gamma'], learning_rate=grid_search.best_params_['learning_rate'], max_depth=grid_search.best_params_['max_depth'],n_jobs=-1)
        y_predict = cross_val_predict(model,X,y,cv=kfold)
        df_predict_temp = pd.DataFrame(y_predict,columns = [r])
        df_predict_XGB = pd.concat([df_predict_XGB,df_predict_temp],axis = 1) 

    df_predict_XGB.to_csv('df_predict_XGB.csv')


def SVM_grid_search_result(X,y,pre = 0):
    'GridSearch'
    kfold = StratifiedKFold(n_splits= 5, shuffle = True,random_state = 7)
    C=np.arange(0.5,10,0.5)
    gamma=[1, 0.1, 0.01, 0.001,0.0001]
    if pre == 0:
        clf = Pipeline([('ss',StandardScaler()), ('svc', SVC())])
        param_grid={'svc__gamma':gamma,'svc__C':C}
    else:
        clf = SVC()
        param_grid={'gamma':gamma,'C':C}
    grid_search = GridSearchCV(clf,param_grid,scoring = 'balanced_accuracy',n_jobs = -1,cv = kfold)     
    
    grid_result = grid_search.fit(X,y)
    print("SVM_Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
    
    'result'
    df_predict_svm = pd.DataFrame(y,columns=[0])
    for r in range(1000):
        kfold = StratifiedKFold(n_splits=5, shuffle = True,random_state= r )
        if pre == 0: 
            model = Pipeline([('ss',StandardScaler()), ('svc', SVC(kernel='rbf',gamma =  grid_search.best_params_['svc__gamma'], C =  grid_search.best_params_['svc__C']))])
        else:
            model = SVC(kernel='rbf',gamma =  grid_search.best_params_['gamma'], C =  grid_search.best_params_['C'])
        model.fit(X,y)
        y_predict = cross_val_predict(model,X,y,cv=kfold)
        df_predict_temp = pd.DataFrame(y_predict,columns = [r])
        df_predict_svm = pd.concat([df_predict_svm,df_predict_temp],axis = 1) 
       
    df_predict_svm.to_csv('df_predict_svm.csv')
    
    
def LR_grid_search_result(X,y,pre = 0):
    'GridSearch'
    kfold = StratifiedKFold(n_splits=5, shuffle = True,random_state=7)
    C = np.arange(0.5,10,0.5)
    solver = ['liblinear', 'newton-cg', 'sag', 'lbfgs']
    
    if pre == 0:
        model = Pipeline([('ss',StandardScaler()), ('LR', LR(penalty ='l2',max_iter=10000))])
        param_grid = {"LR__C": C,"LR__solver": solver}
    else:
        model = LR(penalty ='l2',max_iter=10000)
        param_grid = {"C": C,"solver": solver}
    grid_search = GridSearchCV(model,param_grid,scoring = 'roc_auc',n_jobs = -1,cv = kfold)     
    grid_result = grid_search.fit(X,y)
    print("LR_Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))

    'result'
    df_predict_LR = pd.DataFrame(y,columns=[0])
    for r in range(1000):
        kfold = StratifiedKFold(n_splits=5, shuffle = True,random_state= r )
        if pre == 0:
            model = Pipeline([('ss',StandardScaler()), ('LR_2', LR(penalty ='l2',solver = grid_search.best_params_['LR__solver'],C = grid_search.best_params_['LR__C'],max_iter=10000))])
        else:
            model = LR(penalty ='l2',solver = grid_search.best_params_['solver'],C = grid_search.best_params_['C'],max_iter=10000)
        model.fit(X,y)
        y_predict = cross_val_predict(model,X,y,cv=kfold)
        df_predict_temp = pd.DataFrame(y_predict,columns = [r])
        df_predict_LR = pd.concat([df_predict_LR,df_predict_temp],axis = 1) 
    
    df_predict_LR.to_csv('df_predict_LR.csv')


param = RF_gridsearch_result(X, y)
XGB_gridsearch_result(X, y)
SVM_grid_search_result(X, y,pre = 0)
LR_grid_search_result(X, y,pre = 0)
RF_permutation_test(X, y, param = param)