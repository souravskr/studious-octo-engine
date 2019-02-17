import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score



data_frame = np.asarray(pd.read_csv(sys.argv[1], sep= '\t',header= None))


# Dividing dataset into two sets based on Class (Yes(1) and No(0))
X= data_frame[:,0:347]
y = data_frame[:,-1]
thresholder = VarianceThreshold(threshold=(.5 * (1 - .5)))
X_new = thresholder.fit_transform(X)
#print(np.shape(X_new))

y = np.array(y)
Y = []
for item in y:
	Y.append([item])
Y = np.array(Y)
dataset = np.concatenate((X_new, Y), axis=1)

a = dataset[0:348, :]
l = len(a)
b = dataset[l: , :]


# Splitting  each Class (Yes(1) and No(0)) into 10 parts
yes_1, yes_2, yes_3,yes_4, yes_5, yes_6,yes_7, yes_8, yes_9,yes_10 = np.array_split(a,10,axis=0)
no_1, no_2, no_3,no_4, no_5, no_6,no_7, no_8, no_9,no_10 = np.array_split(b,10,axis=0)

# Concatenating Each spitted class

yes_no_1 = np.vstack([yes_1, no_1])
yes_no_2 = np.vstack([yes_2, no_2])
yes_no_3 = np.vstack([yes_3, no_3])
yes_no_4 = np.vstack([yes_4, no_4])
yes_no_5 = np.vstack([yes_5, no_5])
yes_no_6 = np.vstack([yes_6, no_6])
yes_no_7 = np.vstack([yes_7, no_7])
yes_no_8 = np.vstack([yes_8, no_8])
yes_no_9 = np.vstack([yes_9, no_9])
yes_no_10 = np.vstack([yes_10, no_10])
yes_no = np.array([yes_no_1,yes_no_2,yes_no_3,yes_no_4,yes_no_5,yes_no_6,yes_no_7,yes_no_8,yes_no_9,yes_no_10])

# For K-Fold CV concatenating the training sets
fold_1 = np.vstack([yes_no_2,yes_no_3,yes_no_4,yes_no_5, yes_no_6, yes_no_7,yes_no_8,yes_no_9, yes_no_10])
fold_2 = np.vstack([yes_no_1,yes_no_3,yes_no_4,yes_no_5, yes_no_6, yes_no_7,yes_no_8,yes_no_9, yes_no_10])
fold_3 = np.vstack([yes_no_1,yes_no_2,yes_no_4,yes_no_5, yes_no_6, yes_no_7,yes_no_8,yes_no_9, yes_no_10])
fold_4 = np.vstack([yes_no_1,yes_no_2,yes_no_3,yes_no_5, yes_no_6, yes_no_7,yes_no_8,yes_no_9, yes_no_10])
fold_5 = np.vstack([yes_no_1,yes_no_2,yes_no_3,yes_no_4, yes_no_6, yes_no_7,yes_no_8,yes_no_9, yes_no_10])
fold_6 = np.vstack([yes_no_1,yes_no_2,yes_no_3,yes_no_4, yes_no_5, yes_no_7,yes_no_8,yes_no_9, yes_no_10])
fold_7 = np.vstack([yes_no_1,yes_no_2,yes_no_3,yes_no_4, yes_no_5, yes_no_6,yes_no_8,yes_no_9, yes_no_10])
fold_8 = np.vstack([yes_no_1,yes_no_2,yes_no_3,yes_no_4, yes_no_5, yes_no_6,yes_no_7,yes_no_9, yes_no_10])
fold_9 = np.vstack([yes_no_1,yes_no_2,yes_no_3,yes_no_4, yes_no_5, yes_no_6,yes_no_7,yes_no_8, yes_no_10])
fold_10 = np.vstack([yes_no_1,yes_no_2,yes_no_3,yes_no_4, yes_no_5, yes_no_6,yes_no_7,yes_no_8, yes_no_9])

fold = np.array([fold_1,fold_2,fold_3,fold_4,fold_5,fold_6,fold_7,fold_8,fold_9,fold_10])

def cross_fold (training_set, test_set, k):
    x_train = training_set[:,:-1]
    y_train = training_set[:,-1]
    x_test = test_set[:,:-1]
    y_test = test_set[:,-1]
    # KNN
    classifier = neighbors.KNeighborsClassifier(n_neighbors=k, p=2)
    classifier = classifier.fit(x_train,y_train)
    y_p = classifier.predict(x_test)
    score = classifier.score(x_test, y_test)
    return metrics.accuracy_score(y_test,y_p)

# Calling the Cross-Fold Function with each of the fold and class data and finding the score for each of the K
k_value = [13]
knn_score = []
for item in k_value:
    knn_value = [cross_fold(fold_1,yes_no_1, item),cross_fold(fold_2,yes_no_2, item)
                     ,cross_fold(fold_3,yes_no_3, item),cross_fold(fold_4,yes_no_4, item)
                     ,cross_fold(fold_5,yes_no_5, item),cross_fold(fold_6,yes_no_6, item)
                     ,cross_fold(fold_7,yes_no_7, item),cross_fold(fold_8,yes_no_8, item)
                     ,cross_fold(fold_9,yes_no_9, item),cross_fold(fold_10,yes_no_10, item)]
    knn_score.append(knn_value)
np_knn_score = np.array(knn_score)

# To predict the test set results
def y_prediction(training_set, test_set, k):
    x_train = training_set[:,:-1]
    y_train = training_set[:,-1]
    x_test = test_set[:,:-1]
    classifier = KNeighborsClassifier(k, p=2)
    classifier = classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    return y_pred

# To get predicted value for each of k
predicted_y = []
for item in k_value:
    y_value = [y_prediction(fold_1,yes_no_1, item),y_prediction(fold_2,yes_no_2, item)
                     ,y_prediction(fold_3,yes_no_3, item),y_prediction(fold_4,yes_no_4, item)
                     ,y_prediction(fold_5,yes_no_5, item),y_prediction(fold_6,yes_no_6, item)
                     ,y_prediction(fold_7,yes_no_7, item),y_prediction(fold_8,yes_no_8, item)
                     ,y_prediction(fold_9,yes_no_9, item),y_prediction(fold_10,yes_no_10, item)]
    predicted_y.append(y_value)

def y_actual (dataset):
    y_dataset = dataset[:,-1]
    return y_dataset

# Getting actual value of class
actual_y = []
for elem in yes_no:
    actual_y.append(y_actual(elem))

# Building Confusion Matirx for each K
y_true = (np.concatenate((actual_y[0],actual_y[1],actual_y[2],
                          actual_y[3],actual_y[4],actual_y[5],
                          actual_y[6],actual_y[7],actual_y[8],
                          actual_y[9]),axis=0)) # Adding all actual 0 and 1 in a single list
y_true = y_true.astype(int) # Converting float to integer such as 0.0 = 0 and 1.0 = 1
y_true_pd = pd.Series(y_true,name='Actual')

y_pred = []
for row in range(len(predicted_y)): # Adding all predicted 0 and 1 in a single list
    y_pred_each_row = np.concatenate((predicted_y[row][0],predicted_y[row][1],predicted_y[row][2],predicted_y[row][3],
                             predicted_y[row][4],predicted_y[row][5],predicted_y[row][6],predicted_y[row][7],
                             predicted_y[row][8],predicted_y[row][9]))
    y_pred.append(y_pred_each_row)

y_pred_df = []
for ele in range(len(y_pred)):
    y_pred_pd = pd.Series(y_pred[ele].astype(int), name='Predicted')
    y_pred_df.append(y_pred_pd)

confusion_mat_list = []
for element in range(len(y_pred_df)):
    confusion_mat = pd.crosstab(y_pred_df[element], y_true_pd)
    confusion_mat_list.append(confusion_mat)

for value in range(len(k_value)):
    print('*********************************************\n')
    print('For Value K =', k_value[value],'\n', 'The Average Accuracy Score is: ', np.round(np_knn_score[value].mean(),2)
          ,'\n', '\n','CONFUSION METICS: ','\n',confusion_mat_list[value],'\n')
    tur = y_true_pd.tolist()
    pred = y_pred_df[value].tolist()
    con_sk = confusion_matrix(tur, pred) #Confusion matrix by Sci-kit Learn
    acc = np.round(accuracy_score(tur,pred),2)
    print('Accuracy: ', acc)
    print('PERFORMANCE METRICS: \n', classification_report(tur, pred))
    plt.show()

# For Log-Loss Function & ROC Curve

def predict_proba(training_set,test_set,k):
    x_train = training_set[:, :-1]
    y_train = training_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    # KNN
    classifier = neighbors.KNeighborsClassifier(n_neighbors=k, p=2)
    classifier = classifier.fit(x_train, y_train)
    pre_proba = classifier.predict_proba(x_test)
    pre_proba = pre_proba[:,1]
    return pre_proba


proba_1 = predict_proba(fold_1,yes_no_1, 13)
proba_2 = predict_proba(fold_2, yes_no_2, 13)
proba_3 = predict_proba(fold_3, yes_no_3, 13)
proba_4 = predict_proba(fold_4, yes_no_4, 13)
proba_5 = predict_proba(fold_5, yes_no_5, 13)
proba_6 = predict_proba(fold_6, yes_no_6, 13)
proba_7 = predict_proba(fold_7, yes_no_7, 13)
proba_8 = predict_proba(fold_8, yes_no_8, 13)
proba_9 = predict_proba(fold_9, yes_no_9, 13)
proba_10 = predict_proba(fold_10, yes_no_10,13)

proba = np.array([proba_1,proba_2,proba_3,proba_4,proba_5,proba_6,proba_7,proba_8,proba_9,proba_10])

log_loss_1 = log_loss(actual_y[0],proba_1)
log_loss_2 = log_loss(actual_y[1], proba_2)
log_loss_3 = log_loss(actual_y[2], proba_3)
log_loss_4 = log_loss(actual_y[3], proba_4)
log_loss_5 = log_loss(actual_y[4], proba_5)
log_loss_6 = log_loss(actual_y[5], proba_6)
log_loss_7 = log_loss(actual_y[6], proba_7)
log_loss_8 = log_loss(actual_y[7], proba_8)
log_loss_9 = log_loss(actual_y[8], proba_9)
log_loss_10 = log_loss(actual_y[9],proba_10)

t_logloss = np.array([log_loss_1,log_loss_2,log_loss_3,log_loss_4,log_loss_5,log_loss_6,log_loss_7,log_loss_8,
         log_loss_9,log_loss_10])

print('Log Loss: ', round(t_logloss.mean(),2))
print('*********************************************\n')

###############################################

from scipy import interp
tprs = []
base_fpr = np.linspace(0, 1, 101)
plt.figure(figsize=(5, 5))

for items in range(10):
    fpr, tpr, threshold = roc_curve(actual_y[items], proba[items])

    roc_auc = metrics.auc(fpr,tpr)
    plt.plot(fpr, tpr, 'b',label = 'AUC = %0.2f' % roc_auc)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)


tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

roc_auc = metrics.auc(base_fpr,mean_tprs)
plt.plot(base_fpr, mean_tprs, 'b',label = 'AUC = %0.2f' % roc_auc)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.title('Receiver Operating Characteristic')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
