import sys
import pandas as pd
from operator import itemgetter


# Load the training set and testing set
sys.argv[1] = 'TrainingData_A1.tsv'
df_train = pd.read_csv('TrainingData_A1.tsv', sep='\t')
input_data_train = df_train.astype(float).values.tolist()
full_train_data = df_train.astype(float).values.tolist()
for i in full_train_data:
    del i[-1]                   # Remove the "Class" Column from the training set.


sys.argv[2] = 'TestData_A1.tsv'

df_test = pd.read_csv('TestData_A1.tsv', sep='\t')
full_test_data = df_test.astype(float).values.tolist ()



# Euclidean Distance

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

distance_list = []
for i in full_test_data:
    distance = [euclidean(i, j) for j in full_train_data]
    distance_list.append(distance)

#Merge the euclidean distance into training data set
included_data = []
row = int(input("Please Enter The Row Number of the Testing Dataset: " + "(0 to 9)"))
for counter in range(204):
    input_data_train[counter].append(distance_list[row][counter])
    included_data.append(input_data_train[counter])
    #print(included_data)
from operator import itemgetter
sorted_data = sorted(included_data, key=itemgetter(10)) # Sorted the Euclidean distance in ascending order
print(sorted_data)

k = int(sys.argv[3]) if len(sys.argv) >= 4 else int(3)

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
print('Nearest Neighbours are in row (', row, ') is: ', knn_list)
print('For K =', k)

for test in range(len(knn_list)):
    z = knn_list[test][0]
    if knn_list[test][0] == 1:
        m += 1
    elif knn_list[test][0] == 2:
        n += 1
    elif knn_list[test][0] == 3:
        o += 1
    elif knn_list[test][0] == 5:
        p += 1
    elif knn_list[test][0] == 6:
        q +=1
    elif knn_list[test][0] == 7:
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
acc_knn = (s_max*100)/len(knn_list)
print('Estimated conditional probability of the predicted class: ',acc_knn,'%')

