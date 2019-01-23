import pandas as pd
import numpy as np
import math as mt
import sys
import csv
import operator
#sys.argv[1] = 'train.csv'

df_train = pd.read_csv('train.csv')
input_data_train = df_train.astype(float).values.tolist()
full_train_data = df_train.astype(float).values.tolist()
for i in full_train_data:
    del i[-1]
a = full_train_data
#a1 = a[0]
#flat_list = [item for sublist in a for item in sublist]
#print(flat_list)
#print(a1)
#print(len(a1))
#print(len(a))

df_test = pd.read_csv('test.csv')
full_test_data = df_test.astype(float).values.tolist ()
b = full_test_data
#b1 = b [0]
#print(b1)
#print(len(b1))
#print(b)


# Euclidean Distance


#################################################################
def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

d2 = []
for i in b:
    foo = [euclidean(i, j) for j in a]
    d2.append(foo)
#print('Euclidean Disntance: ', d2[0])

#print('Length: ', len(d2))
print('Each List: ', len(d2[0]))
included_data = []
for counter in range(204):
    input_data_train[counter].append(d2[0][counter])
    included_data.append(input_data_train[counter])
    #print(a[counter])
#print(included_data)
from operator import itemgetter
sorted_data = sorted(included_data, key=itemgetter(10))
print(len(included_data))
print(sorted_data)

knn_list = []
k = 3
for elements in range(k):
    knn = sorted_data[elements][9:]
    knn_list.append(knn)
print(knn_list)

from collections import Counter
data_counter_1 = 0
data_counter_2 = 0
data_counter_3 = 0
data_counter_5 = 0
data_counter_6 = 0
data_counter_7 = 0
for count in range (k):
    counted_item = knn_list[count][0]
    if counted_item == 1:
        data_counter_1 = data_counter_1 + 1
    elif counted_item == 2:
        data_counter_2 = data_counter_2 + 1
    elif counted_item == 3:
        data_counter_3 = data_counter_3 + 1
    elif counted_item == 5:
        data_counter_5 = data_counter_5 + 1
    elif counted_item == 6:
        data_counter_6 = data_counter_6 + 1
    elif counted_item == 7:
        data_counter_7 == data_counter_7 + 1
    print(counted_item)
data_counter_list = [data_counter_1,data_counter_2,data_counter_3,data_counter_5,data_counter_6,data_counter_7]
print(data_counter_list)
if data_counter_1 > 0:
    print('No. Of 1s: ', data_counter_1)
if data_counter_2 > 0:
    print('No. Of 2s: ', data_counter_2)
if data_counter_3 > 0:
    print('No. Of 3s: ', data_counter_3)
if data_counter_5 > 0:
    print('No. Of 5s: ', data_counter_5)
if data_counter_6 > 0:
    print('No. Of 6s: ', data_counter_6)
if data_counter_7 > 0:
    print('No. Of 7s: ', data_counter_7)

maximum_repeated = max(data_counter_list)
if data_counter_1 == maximum_repeated:
    print('Number of Class- 1: ', maximum_repeated)
elif data_counter_2 == maximum_repeated:
    print('Number of Class- 2: ', maximum_repeated)
elif data_counter_3 == maximum_repeated:
    print('Number of Class- 3: ', maximum_repeated)
elif data_counter_5 == maximum_repeated:
    print('Number of Class- 5: ', maximum_repeated)
elif data_counter_6 == maximum_repeated:
    print('Number of Class- 6: ', maximum_repeated)
elif data_counter_7 == maximum_repeated:
    print('Number of Class- 7: ', maximum_repeated)
print('')

accuracy_knn = (maximum_repeated*100)/k
print('Accuracy of KNN Calculation: ',accuracy_knn,'%')




 #########################################################


