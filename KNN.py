import sys
from operator import itemgetter
import pandas as pd


def load_data(train_data, test_data, k, row):
    full_training_set = pd.read_csv(train_data, sep='\t').values.tolist() #Taking Training_Data
    rd_train_data = pd.read_csv(train_data, sep='\t').values.tolist()
    rd_test_data = pd.read_csv(test_data, sep='\t').values.tolist() # Taking Testing Data
    for class_data in rd_train_data:
        del class_data[-1] # Divide the training set from class
    euc_distance = []
    for item in rd_test_data:
        distance = [euclidean_distance(item, element) for element in rd_train_data]
        euc_distance.append(distance) #Calculating the eclidean_distance and create a new list with the distance

    for column in range(204):
        full_training_set[column].append(euc_distance[row][column]) #Taking euclidean distance of each row from test set and merge into the training dataset
    sorted_full_set = sorted(full_training_set,key=itemgetter(10)) # Sort the euclidean distance based in ascending order

    new_list = []
    for value in range(k):
        a = sorted_full_set[value]
        select_class = a[-2]
        new_list.append(select_class) # based on the value of K, create a new list of class
    count_number = [new_list.count(1),new_list.count(2),new_list.count(3),new_list.count(5),new_list.count(6),new_list.count(7)] # Which class is maximum iterated
    accu_knn = round(((max(count_number)/len(new_list))*100),2) # Calculate the value of Estimated conditional probability of the predicted class
    print(int(max(new_list, key=new_list.count)), end='           ')
    print(accu_knn, "%")


def euclidean_distance(train_data, test_data): # function for euclidean distance

    return sum((p - q) ** 2 for p, q in zip(train_data, test_data)) ** .5


def main(R):
    trainingSet= sys.argv[1]
    testingSet = sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) >= 4 else int(3) # The default value of K=3, if no input of K
    #R = int(input("Please Enter The Row Number of the Testing Dataset: " + " (0 to 9)")) # seleting the row of testing dataset to find out the class of that row
    load_data(trainingSet, testingSet, K, R)
for item in range(9):
    main(item) #call the main function

