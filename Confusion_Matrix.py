import pandas as pd
import numpy as np
import sys

data_frame = np.array(pd.read_csv(sys.argv[1], sep= '\t'))
df = pd.read_csv(sys.argv[1], sep= '\t').values
class_label = data_frame[:,:1]
data = (data_frame[:,1:])
#**************************************************************
sum_data = np.sum(data)
data_range = range(len(data))
c = []
index = []
for i in data_range:
    d = data[i][i]
    c.append(d)
    index.append(i)
sum_c = np.sum(c)
ac = round(sum_c/sum_data,2)
print('Ac',ac,sep='\t')

#****************************************************************

def pre_cision(confu_mat):
    precision = []
    for column in data_range:
        data_precision = []
        for item in data_range:
            data_precision.append(confu_mat [column][item])
        sum_data_pre = np.sum(data_precision)
        precision.append(round((c[column]/sum_data_pre),2))
    return precision

def fdr(confu_mat):
    FDR = []
    for item in range(3):
        FDR.append(round(1-pre_cision(confu_mat)[item],2))
    return FDR


def re_call(confu_mat):
    recall = []
    for column in data_range:
        data_recall = []
        for item in data_range:
            data_recall.append(confu_mat [item][column])
        sum_data_recall = np.sum(data_recall)
        recall.append(round((c[column] / sum_data_recall), 2))
    return recall



#*******************************************************************

def speci_ficity(label, confu_mat):
    sum_RN = 0
    TN = 0
    for row in range(len(confu_mat)):
        for column in range(len(confu_mat)):
            if row != label and column != label:
                TN += confu_mat[row][column]

        if row != label:
            sum_RN += confu_mat[label][row]

    sum_RN = TN + sum_RN
    specificity = round((TN / sum_RN),2)

    return specificity


precision = 'P'
recall = 'R'
specificity = 'Sp'
FDR = 'FDR'
column = ' '
print("%-8s %-8s %-8s %-8s %-8s" %(column,precision,recall,specificity,FDR))

for label in range(3):
    print("%-8s %-8s %-8s %-8s %-8s" %(class_label[label][0],pre_cision(data)[label],
                                       re_call(data)[label],speci_ficity(label,data),
                                       fdr(data)[label]))
