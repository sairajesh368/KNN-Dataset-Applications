import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import math
import copy
import numpy as np
import pandas as pd

# Function to normalize data set
def normalize_data(features, loan_min, loan_max):
    for i in features:
        for j in range(len(i)-1):
            numerator=(i[j]-loan_min[j])
            denominator=(loan_max[j]-loan_min[j])
            i[j]=numerator/denominator

# Function to calculate minimum of all the features
def find_minimum_of_columns(loan_features, loan_minimum):
    for i in range(21):
        loan_minimum.append(min(row[i] for row in loan_features))

# Function to calculate maximum of all the features
def find_maximum_of_columns(flower_dimensions, loan_maximum):
    for i in range(21):
        loan_maximum.append(max(row[i] for row in flower_dimensions))

# Function to calculate euclidean distance of training data set 
def calculate_euclidean_distance_training(eucli_list, j, l):
    eucli_distance = 0
    for i in range(21):
        eucli_distance += ((eucli_list[j][i]-eucli_list[l][i])**2)
    return math.sqrt(eucli_distance)


# Function to calculate Euclidean distance between two instances of testing data set
def calculate_euclidean_distance_testing(loan_te, loan_tr, j, l):
    eucli_distance = 0
    for i in range(21):
         eucli_distance += ((loan_te[j][i]-loan_tr[l][i])**2)
    return math.sqrt(eucli_distance)

def get_label_from_prediction_dictionary(counting_labels_dict):
    label=''
    max_count=0
    for element in counting_labels_dict:
        if counting_labels_dict[element]>max_count:
            max_count=counting_labels_dict[element]
            label=element
    return label

def get_training_accuracy(loan_train, avg_training_data_list, k):

    set_accurate_predictions_count = 0
    for j in range(len(loan_train)):
        euclidean_distances = []
        for l in range(len(loan_train)):
                eucli_distance=calculate_euclidean_distance_training(loan_train,j, l)
                euclidean_distances.append(list((eucli_distance,loan_train[l][21])))
        euclidean_distances.sort()
        
        counting_labels_dict = {'Y':0,'N':0}
        for s in range(k):
            counting_labels_dict[euclidean_distances[s][1]]+=1
        label=get_label_from_prediction_dictionary(counting_labels_dict)
        if label==loan_train[j][21]:
            set_accurate_predictions_count+=1
    average_set_predictions=set_accurate_predictions_count/training_set_total_count
    avg_training_data_list.append(average_set_predictions)


def get_testing_accuracy(loan_tr, loan_te, avg_testing_data_list, k):
    set_accurate_predictions_count = 0
    tem = []
    for j in range(len(loan_te)):
        euclidean_distances = []
        for l in range(len(loan_tr)):
                eucli_distance=calculate_euclidean_distance_testing(loan_te, loan_tr, j, l)
                euclidean_distances.append(list((eucli_distance,loan_tr[l][21])))
        euclidean_distances.sort()
        counting_labels_dict = {'Y':0,'N':0}
        for s in range(k):
            counting_labels_dict[euclidean_distances[s][1]]+=1
        label=get_label_from_prediction_dictionary(counting_labels_dict)
        tem.append(label)
        if label==loan_te[j][21]:
            set_accurate_predictions_count+=1
    average_set_predictions=set_accurate_predictions_count/testing_set_total_count
    avg_testing_data_list.append(average_set_predictions)

    return tem

def calculate_precision_recall_f1score(loan_te , tem):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    actual_pos = 0
    actual_neg = 0
    for i in range(len(tem)):
        if(loan_te[i][21] == 'Y'):
            actual_pos += 1
        else:
            actual_neg += 1
        if(tem[i] == loan_te[i][21]):
            if(tem[i] == 'Y'):
                TP += 1
            else:
                TN += 1
    FN = actual_pos - TP
    FP = actual_neg - TN
    precision = TP/(TP+FP)
    recall = TP/(TP+FN) 
    f1score = (2*precision*recall)/(precision+recall)
    return f1score


if __name__ == "__main__":

    dafa_frame = pd.read_csv('loan.csv')
    loan_features = dafa_frame.iloc[:, 0:13].values
    class_labels = dafa_frame['Loan_Status'].values
    original_column_list = [0, 1, 2, 3, 4, 9, 10]
    loan_features = [list(inner_list[1:]) for inner_list in loan_features]
    unique_elements = []
    print("Calculating values")

    temp = np.array(loan_features)
    temp = temp.T
    for i in original_column_list:
        unique_elements.append(list(np.unique(temp[i])))
    temp = []
    for i in loan_features:
        idx = 0
        for j in range(len(i)-1):
            if(str(i[j]) in unique_elements[idx]):
                zero_list = [0]*(len(unique_elements[idx]))
                ind = unique_elements[idx].index(str(i[j]))
                zero_list[ind] = 1
                i[j] = zero_list
                idx += 1
        i = [item for sublist in i for item in (sublist if isinstance(sublist, list) else [sublist])]
        i = [int(item) if index < len(i) - 1 else item for index, item in enumerate(i)]
        temp.append(i)
    loan_features = temp

    k_fold = 10
    fold_indices = []
    unique_classes, class_counts = np.unique(class_labels, return_counts=True)


    for unique_class in unique_classes:
        unique_class_indices = np.where(class_labels == unique_class)[0]
        unique_class_folds = np.array_split(unique_class_indices, k_fold)
        fold_indices.extend(unique_class_folds)

    final_folds_list = []
    for fold_index in range(k_fold):
        train_index = list(fold_indices[fold_index]) + list(fold_indices[(fold_index+k_fold)])
        np.random.shuffle(train_index)
        final_folds_list.append(train_index)

    k_fold_data_features_dict={}
    k_fold = 0
    k_fold_data_labels_dict = {}
    for j in final_folds_list:
        temp_features=[]
        temp_labels=[]
        for l in j:
            temp_features.append(list(loan_features[l]))
            temp_labels.append(class_labels[l])
        k_fold_data_features_dict[k_fold]=temp_features
        k_fold_data_labels_dict[k_fold]=temp_labels
        k_fold+=1

    k_avg_testing_accuracy = []
    k_avg_testing_f1score = []


    # K range varies from 1 to 51 given K is odd
    for k in range(1,53,2):
        avg_training_data_list = []
        avg_testing_data_list = []
        avg_testing_data_list_f1score = []

        # Loop to train the algorithm 20 times with paritioning and normalizing every time
        for i in range(10):
            loan_minimum = []
            loan_maximum = []
            loan_train = []
            for il in range(10):
                if il != i:
                    loan_train+=k_fold_data_features_dict[il]
            loan_test = k_fold_data_features_dict[i]

            # Count of Training instances with 80% division
            training_set_total_count = len(loan_train)
            # Count of Testing instances with 20% division
            testing_set_total_count = len(loan_test)

            find_minimum_of_columns(loan_train, loan_minimum)
            # Find maximum of all features of training data set
            find_maximum_of_columns(loan_train, loan_maximum)

            duplicate_copy = copy.deepcopy(loan_train)

            normalize_data(duplicate_copy, loan_minimum, loan_maximum)

            loan_train = duplicate_copy
            duplicate_copy = copy.deepcopy(loan_test)

            normalize_data(duplicate_copy, loan_minimum, loan_maximum)

            loan_test = duplicate_copy
        
            get_training_accuracy(loan_train, avg_training_data_list, k)
            tem = get_testing_accuracy(loan_train, loan_test, avg_testing_data_list, k)
            f1 = calculate_precision_recall_f1score(loan_test, tem)
            avg_testing_data_list_f1score.append(f1)

        k_avg_testing_accuracy.append(avg_testing_data_list)       
        k_avg_testing_f1score.append(avg_testing_data_list_f1score)

    x_axis_list=list(range(1,53,2))

    testing_mean = [np.mean(inner_list) for inner_list in k_avg_testing_accuracy]

    testing_f1score = [np.mean(inner_list) for inner_list in k_avg_testing_f1score]

    testing_standard_deviation = [np.std(inner_list) for inner_list in k_avg_testing_accuracy]

    f1score_standard_deviation = [np.std(inner_list) for inner_list in k_avg_testing_f1score]


    # Create the first subplot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    plt.xticks(x_axis_list)
    plt.errorbar(x_axis_list, testing_mean, yerr=testing_standard_deviation, capsize=5)
    plt.xlabel('Value of K')
    plt.ylabel('Average accuracy of Testing data set')
    plt.title('K-Accuracy')

    # Create the second subplot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
    plt.xticks(x_axis_list)
    plt.errorbar(x_axis_list, testing_f1score, yerr=f1score_standard_deviation, capsize=5)
    plt.xlabel('Value of K')
    plt.ylabel('Average F1score of Testing data set')
    plt.title('K-F1score')

    plt.suptitle('Loan dataset')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()



