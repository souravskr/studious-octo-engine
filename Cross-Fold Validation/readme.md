This program is for k-fold cross-validation (k-fold CV) from scratch in Python3. To implement the k-fold
CV to choose the number of neighbours and features to obtain the best KNN model for the dataset provided
in (A2_t2_dataset.tsv). this program used the KNN classifier provided by scikit learn (https://scikit-learn.org/stable/modules/neighbors.html).
The complete program takes one command-line argument: the name of the file with the data. The
program's output is the details (number of neighbours and features) of the model chosen and its
performance metrics. The given data is in a tab-delimited plain-text format with the last column indicating
the class of the instance (1 = positive, 0 = negative).
