import pandas as pd
import sys

# Load the training set and testing set
sys.argv[1] = 'TrainingData_A1.tsv'
df_train = pd.read_csv('TrainingData_A1.tsv', sep='\t')
input_data_train = df_train.astype(float).values.tolist()
full_train_data = df_train.astype(float).values.tolist()
for i in full_train_data:
    del i[-1]
a = full_train_data         # Remove the "Class" Column from the training set.

sys.argv[2] = 'TestData_A1.tsv'

df_test = pd.read_csv('TestData_A1.tsv', sep='\t')
full_test_data = df_test.astype(float).values.tolist ()
b = full_test_data


# Euclidean Distance

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

d2 = []
for i in b:
    foo = [euclidean(i, j) for j in a]
    d2.append(foo)

#Merge the euclidean distance into training data set
included_data = []
row = int(input("Please Enter The Row Number of the Testing Dataset: " + "(0 to 9)"))
for counter in range(204):
    input_data_train[counter].append(d2[row][counter])
    included_data.append(input_data_train[counter])
from operator import itemgetter
sorted_data = sorted(included_data, key=itemgetter(10)) # Sorted the Euclidean distance in ascending order

try:
    k = int(input('Please put the value for K: ')) # Take the value of K and set the default value of K=3
except ValueError:
    k = 3

# Make a sorted list with "Class" & Euclidean Distance
knn_list = []
for elements in range(k):
    knn = sorted_data[elements][9:]
    knn_list.append(knn)


#Calculate the "Class" based on the sorted list
m = 0
n = 0
o = 0
p = 0
q = 0
r = 0
for_test = knn_list
print('Class with Euclidean Distance for Row ', row, ' is: ', for_test)
print('And The length is ', len(knn_list))

for test in range(len(for_test)):
    z = for_test[test][0]
    if for_test[test][0] == 1:
        m += 1
        #print(m)
        #print('Class: "1"')
    elif for_test[test][0] == 2:
        n += 1
        #print(n)
        #print('Class: "2"')
    elif for_test[test][0] == 3:
        o += 1
        #print(o)
        #print('Class: "3"')
    elif for_test[test][0] == 5:
        p += 1
        #print('Class: "5"')
        #print(p)
    elif for_test[test][0] == 6:
        q +=1
        #print(q)
        #print('Class: "6"')
    elif for_test[test][0] == 7:
        #print('Class: "7"', r)
        r +=1
s = [m,n,o,p,q,r]
s_max = max(s) # Calculate the maximum iterated class
if s_max == m:
    print('Your Dataset Is in Class "1"')
elif s_max == n:
    print('Your Dataset Is in Class "2"')
elif s_max == o:
    print('Your Dataset Is in Class "3"')
elif s_max == p:
    print('Your Dataset Is in Class "5"')
elif s_max == q:
    print('Your Dataset Is in Class "6"')
elif s_max == r:
    print('Your Dataset Is in Class "7"')

#Calculate the accuracy of value of KNN
acc_knn = (s_max*100)/k
print('Estimated conditional probability of the predicted class: ',acc_knn,'%')

