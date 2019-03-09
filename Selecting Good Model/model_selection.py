import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold, cross_val_score,GridSearchCV
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt 
from sklearn import metrics



data = pd.read_csv(sys.argv[1],sep='\t',header=None)
real_test = pd.read_csv(sys.argv[2],sep='\t',header= None)

X = data.iloc[:,:71]
y = np.array(data.iloc[:,-1])
score_metric = ['average_precision']
#score_index = ['average_precision','balanced_accuracy','f1','recall']

#score_index = ['average_precision','balanced_accuracy', 'recall', 'f1']

strati_CV = StratifiedKFold(n_splits=10, shuffle=True)

model = RandomForestClassifier(n_estimators=100,
    criterion='entropy',
    max_depth=5,
    max_features='auto',
    min_impurity_split=None,
    n_jobs=-1,
    random_state=0,
    class_weight='balanced')

'''
selector = RFECV(model, cv=3)
selector = selector.fit(X,y)
X_new = selector.transform(X)
'''
#****************************

sfm = SelectFromModel(model, threshold=0.01)
sfm.fit(X, y)
X_new = sfm.transform(X)

#print(np.shape(X_new))

param_grid = {'n_estimators': [100, 200, 700],'max_features': ['auto', 'sqrt', 'log2']}
print('\n')
print("Random Forest: \n")
for score in score_metric:
    random_forest = GridSearchCV(model, param_grid,cv=strati_CV,scoring=score,n_jobs=-1)
    random_forest.fit(X_new,y)
    print(score,':', np.round(random_forest.best_score_,2))
print('Best Parameter: ',random_forest.best_params_)
print('Best Estimator: ',random_forest.best_estimator_)
print('Standard Deviation: ',random_forest.cv_results_['std_test_score'][random_forest.best_index_])

print('\n')

k_range = range(3,31)
weight_options = ["uniform", "distance"]
algorithms = ['auto','ball_tree', 'kd_tree', 'brute']
metric = ['minkowski', 'euclidean', 'manhattan']
param_grid = dict(n_neighbors = k_range, weights = weight_options, algorithm=algorithms, metric = metric)
knn = KNeighborsClassifier()
print("KNN with Grid-Search CV\n")
for score in score_metric:
    grid_cv_knn = GridSearchCV(knn,param_grid,cv=strati_CV,scoring=score,n_jobs=-1)
    grid_cv_knn.fit(X_new,y)
    print(score,':',np.round(grid_cv_knn.best_score_,2))

print('Best Parameter: ', grid_cv_knn.best_params_)
print('Best Estimator: ', grid_cv_knn.best_estimator_)
print('Standard Deviation: ', grid_cv_knn.cv_results_['std_test_score'][grid_cv_knn.best_index_])

print('\n')
print('Support Vector Machine:\n')

# SVC

param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf'], 'class_weight' :['balanced']}
sup_vector = SVC()
print("SVC with Grid-Search CV\n")
for score in score_metric:
    grid_cv_svc = GridSearchCV(sup_vector,param_grid,cv=strati_CV,scoring=score, n_jobs=-1)
    grid_cv_svc.fit(X_new,y)
    print(score,':',np.round(grid_cv_svc.best_score_,2))
print('\n')

print('Best Parameter: ', grid_cv_svc.best_params_,'\n')
print('Best Estimator: ',grid_cv_svc.best_estimator_)
print('Standard Deviation: ',grid_cv_svc.cv_results_['std_test_score'][grid_cv_svc.best_index_])

print('\n')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Best Model is Random Forest Classification



model_randomforest = model.fit(X,y)
prediction_rf = model.predict(real_test)
#print(np.bincount(prediction_rf))


pred_proba = model.predict_proba(real_test)



np.savetxt('ML_A3_class_1.txt', pred_proba[:,1],fmt='%s',newline='\n')


