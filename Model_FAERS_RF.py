# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt


df_Most = pd.read_csv('Most.txt', sep='\t')
df_Moderate = pd.read_csv('Moderate.txt',sep='\t')
df_Less = pd.read_csv('Less.txt', sep='\t')
df_Non = pd.read_csv('Non.txt',sep='\t')
df_FAERS = pd.read_csv('FAERS.txt',sep='\t')
drugbank_150 = pd.read_csv('drugbank_150.txt',sep='\t')

X_drugbank_150 = drugbank_150.iloc[:,2:779]
X_faers = df_FAERS.iloc[:,2:779]

'feature_screening'
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
    tmp = m[i]
    x.append(tmp)

x.append("D164")
X = X.drop(x, axis = 1)
X_drugbank_150 = X_drugbank_150.drop(x,axis = 1)
X_faers = X_faers.drop(x,axis = 1)
X_out_test = pd.concat([X_faers.iloc[0:150,:],X_drugbank_150],axis=0)

'FAERS_top150&Drugbank_150:'

def model_result(Y_test,y_predict):
    print(metrics.accuracy_score(Y_test,y_predict))
    print(metrics.precision_score(Y_test,y_predict))
    print(metrics.recall_score(Y_test,y_predict))
    print(metrics.matthews_corrcoef(Y_test,y_predict))
    print(metrics.f1_score(Y_test,y_predict))
    print(metrics.balanced_accuracy_score(Y_test,y_predict))    
    print (metrics.roc_auc_score(Y_test,y_predict))
    print(metrics.average_precision_score(Y_test,y_predict))
    tn, fp, fn, tp = metrics.confusion_matrix(Y_test,y_predict, labels=[0, 1]).ravel()
    print(tp/ float(tp+ fn))
    print(tn / float(tn + fp))
    
y_true = [1]*150 + [0]*150    
    
def FAERS_Drugbank_outtest(model):
    y_predict = model.predict(X_out_test)
    model_result(y_true,y_predict)
    
 
'''RF'''

'model'
# model = RandomForestClassifier(max_depth=3, min_samples_split=10, n_estimators=20,n_jobs=-1)
# model.fit(X,y)
# joblib.dump(model, "RF.m")
model = joblib.load("RF.m")
importances=model.feature_importances_

FAERS_Drugbank_outtest(model)  


'''RF_permu_X'''

'feature_permutation'
# X_permutation = pd.DataFrame()
# for index,row in X.iteritems():
#     X_permutation[index] = np.random.permutation(X[index])
# X_permutation.to_csv('X_permutation.csv',index = 0) 

'model'
# X_permutation = pd.read_csv('X_permutation.csv')
# model_permu_X = RandomForestClassifier(max_depth=3, min_samples_split=10, n_estimators=20,n_jobs=-1)
# model_permu_X.fit(X_permutation,y)
# joblib.dump(model_permu_X, "RF_permu_X.m")
model_permu_X = joblib.load("RF_permu_X.m")

FAERS_Drugbank_outtest(model_permu_X)



''''RF_permu_Y'''

'label_permutation'
# y_permutation = np.random.permutation(y)
# y_permutation = pd.read_csv('y_permutation.csv')

'model'
# model_permu_Y = RandomForestClassifier(max_depth=3, min_samples_split=10, n_estimators=20,n_jobs=-1)
# model_permu_Y.fit(X,y_permutation)
# joblib.dump(model_permu_Y, "RF_permu_Y.m")
model_permu_Y = joblib.load("RF_permu_Y.m")

FAERS_Drugbank_outtest(model_permu_Y)


'recall count'    
y_FAERS = model.predict(X_faers)

for l in range(150,len(y_FAERS),150):
    print((y_FAERS[0:l].sum())/l)
print(y_FAERS.sum()/len(y_FAERS))    
    
FAERS_ror = pd.read_csv('FAERS_parsing.csv')
FAERS_ror['predict'] = y_FAERS
y_FAERS_scr = FAERS_ror[FAERS_ror['all']!=0]['predict']
for l in range(150,len(y_FAERS_scr),150):
    print((y_FAERS_scr[0:l].sum())/l)
print(y_FAERS_scr.sum()/len(y_FAERS_scr))


'ROC_curve_plt'
y_predict = model.predict(X_out_test)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predict)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='RF (area = {:.3f})'.format(auc))

y_predict = model_permu_X.predict(X_out_test)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predict)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='RF_permu_X (area = {:.3f})'.format(auc))

y_predict = model_permu_Y.predict(X_out_test)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predict)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='RF_permu_Y (area = {:.3f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
# plt.figure()

plt.savefig("ROC_plot.png",dpi = 600)