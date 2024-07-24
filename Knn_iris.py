import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import math
import copy
import numpy as np

# Function to normalize data set
def normalize_data(flower_dimensions, flower_dimensions_minimum, flower_dimensions_maximum):
    for i in range(len(flower_dimensions)):
        numerator_sepal_length=(flower_dimensions[i][0]-flower_dimensions_minimum[0])
        denominator_sepal_length=(flower_dimensions_maximum[0]-flower_dimensions_minimum[0])
        flower_dimensions[i][0]=numerator_sepal_length/denominator_sepal_length

        numerator_sepal_width=(flower_dimensions[i][1]-flower_dimensions_minimum[1])
        denominator_sepal_width=(flower_dimensions_maximum[1]-flower_dimensions_minimum[1])
        flower_dimensions[i][1]=numerator_sepal_width/denominator_sepal_width

        numerator_petal_length=(flower_dimensions[i][2]-flower_dimensions_minimum[2])
        denominator_petal_length=(flower_dimensions_maximum[2]-flower_dimensions_minimum[2])
        flower_dimensions[i][2]=numerator_petal_length/denominator_petal_length

        numerator_petal_width=(flower_dimensions[i][3]-flower_dimensions_minimum[3])
        denominator_petal_width=(flower_dimensions_maximum[3]-flower_dimensions_minimum[3])
        flower_dimensions[i][3]=numerator_petal_width/denominator_petal_width

# Function to calculate minimum of all the features
def find_minimum_of_columns(flower_dimensions, flower_dimensions_minimum):
    for i in range(4):
        flower_dimensions_minimum.append(min(row[i] for row in flower_dimensions))

# Function to calculate maximum of all the features
def find_maximum_of_columns(flower_dimensions, flower_dimensions_maximum):
    for i in range(4):
        flower_dimensions_maximum.append(max(row[i] for row in flower_dimensions))

# Function to calculate euclidean distance of training data set 
def calculate_euclidean_distance_training(eucli_list, j, l):
    eucli_distance = math.sqrt(((eucli_list[j][0]-eucli_list[l][0])**2)+((eucli_list[j][1]-eucli_list[l][1])**2)+((eucli_list[j][2]-eucli_list[l][2])**2)+((eucli_list[j][3]-eucli_list[l][3])**2))
    return eucli_distance


# Function to calculate Euclidean distance between two instances of testing data set
def calculate_euclidean_distance_testing(flower_dimensions_test, flower_dimensions_train, j, l):
    eucli_distance = math.sqrt(((flower_dimensions_test[j][0]-flower_dimensions_train[l][0])**2)+((flower_dimensions_test[j][1]-flower_dimensions_train[l][1])**2)+((flower_dimensions_test[j][2]-flower_dimensions_train[l][2])**2)+((flower_dimensions_test[j][3]-flower_dimensions_train[l][3])**2))
    return eucli_distance

def get_label_from_prediction_dictionary(counting_labels_dict):
    label=''
    max_count=0
    for element in counting_labels_dict:
        if counting_labels_dict[element]>max_count:
            max_count=counting_labels_dict[element]
            label=element
    return label

def get_training_accuracy(flower_dimensions_train, avg_training_data_list, k):

    set_accurate_predictions_count = 0
    for j in range(len(flower_dimensions_train)):
        euclidean_distances = []
        for l in range(len(flower_dimensions_train)):
                eucli_distance=calculate_euclidean_distance_training(flower_dimensions_train,j, l)
                euclidean_distances.append(list((eucli_distance,flower_dimensions_train[l][4])))
        euclidean_distances.sort()
        counting_labels_dict = {'Iris-setosa':0,'Iris-versicolor':0,'Iris-virginica':0}
        for s in range(k):
            counting_labels_dict[euclidean_distances[s][1]]+=1
        label=get_label_from_prediction_dictionary(counting_labels_dict)
        if label==flower_dimensions_train[j][4]:
            set_accurate_predictions_count+=1
    average_set_predictions=set_accurate_predictions_count/120
    avg_training_data_list.append(average_set_predictions)


def get_testing_accuracy(flower_dimensions_train, flower_dimensions_test, avg_testing_data_list, k):
    set_accurate_predictions_count = 0
    for j in range(len(flower_dimensions_test)):
        euclidean_distances = []
        for l in range(len(flower_dimensions_train)):
                eucli_distance=calculate_euclidean_distance_testing(flower_dimensions_test, flower_dimensions_train, j, l)
                euclidean_distances.append(list((eucli_distance,flower_dimensions_train[l][4])))
        euclidean_distances.sort()
        counting_labels_dict = {'Iris-setosa':0,'Iris-versicolor':0,'Iris-virginica':0}
        for s in range(k):
            counting_labels_dict[euclidean_distances[s][1]]+=1
        label=get_label_from_prediction_dictionary(counting_labels_dict)
        if label==flower_dimensions_test[j][4]:
            set_accurate_predictions_count+=1
    average_set_predictions=set_accurate_predictions_count/30
    avg_testing_data_list.append(average_set_predictions)


if __name__ == "__main__":

    flower_dimensions = []

    with open('iris.csv') as iris_dataset:
        iris_data = csv.reader(iris_dataset, delimiter=',')
        for instance in iris_data:
            item_in_flower_dimensions=list((float(instance[0]),float(instance[1]),float(instance[2]),float(instance[3]),instance[4]))
            flower_dimensions.append(item_in_flower_dimensions)

    # Count of Training instances with 80% division
    training_set_total_count = 120
    # Count of Testing instances with 20% division
    testing_set_total_count = 30

    k_avg_training_accuracy = []
    k_avg_testing_accuracy = []

    # K range varies from 1 to 51 given K is odd
    for k in range(1,53,2):
        avg_training_data_list = []
        avg_testing_data_list = []

        # Loop to train the algorithm 20 times with paritioning and normalizing every time
        for i in range(20):
            training_set_accurate_predictions_count = 0
            testing_set_accurate_predictions_count = 0
            flower_dimensions_minimum = []
            flower_dimensions_maximum = []
            flower_dimensions = shuffle(flower_dimensions)
            flower_dimensions_train, flower_dimensions_test = train_test_split(flower_dimensions, test_size=0.2, random_state=42)

            # Find minimum of all features of training data set
            find_minimum_of_columns(flower_dimensions_train, flower_dimensions_minimum)
            # Find maximum of all features of training data set
            find_maximum_of_columns(flower_dimensions_train, flower_dimensions_maximum)

            duplicate_copy = copy.deepcopy(flower_dimensions_train)

            normalize_data(duplicate_copy, flower_dimensions_minimum, flower_dimensions_maximum)

            flower_dimensions_train = duplicate_copy
            duplicate_copy = copy.deepcopy(flower_dimensions_test)

            normalize_data(duplicate_copy, flower_dimensions_minimum, flower_dimensions_maximum)

            flower_dimensions_test = duplicate_copy
        
            get_training_accuracy(flower_dimensions_train, avg_training_data_list, k)
            get_testing_accuracy(flower_dimensions_train, flower_dimensions_test, avg_testing_data_list, k)

        k_avg_training_accuracy.append(avg_training_data_list)
        k_avg_testing_accuracy.append(avg_testing_data_list)

    x_axis_list=list(range(1,53,2))
    training_mean=np.mean(k_avg_training_accuracy,axis=1)
    testing_mean=np.mean(k_avg_testing_accuracy,axis=1)
    training_standard_deviation=np.std(k_avg_training_accuracy,axis=1)
    testing_standard_deviation=np.std(k_avg_testing_accuracy,axis=1)


    # Plotting Graphs
    plt.figure(1)
    plt.xticks(x_axis_list)
    plt.title('Training set graph')
    plt.xlabel('Value of K')
    plt.ylabel('Average accuracy of Training data set')
    plt.errorbar(x_axis_list, training_mean, yerr=training_standard_deviation, capsize=5)

    plt.figure(2)
    plt.xticks(x_axis_list)
    plt.title('Testing set graph')
    plt.xlabel('Value of K')
    plt.ylabel('Average accuracy of Testing data set')
    plt.errorbar(x_axis_list, testing_mean, yerr=testing_standard_deviation, capsize=5)

    plt.show()


