"""
testing_multibiomarker_predictive_analytics_model.py

Script for running the model created by the previous step on the test data.

Input:
        testSet (String): the test set with only the features selected by the modeller included
        testSet_labels (String): the test set labels
        maximums_filename (String): the file with the maximum values of the testset features
        minimums_filename (String): the file with the minimum values of the testset features
        averages_filename (String): the file with the average values from the preprocessing
        features_filename (String): the selected features filename taken from the training step
        missing_imputation_method (Integer): the missing imputation method done in the preprocessing step of the
        training data (1, 2)
        normalization_method (Integer): the normalization method done in the preprocessing step of the training data
        (1, 2)
        model (String): the model created by the training step
        selection_flag (Integer): 0 for multi-class problem, 1 for regression, 2 for two-class problem
        data_been_preprocessed_flag (Integer): 1 if it has been preprocessed, 0 if it hasn't
        variables_for_normalization_string: (String) the selected variables for normalization as a string with comma or
        newline separated strings, eg. "ABC,DEF,GHI"
        filetype (Integer): 7 if it is a gene expressions file, other number if it's a biomarkers file
        has_features_header (Integer): 1 if the testSet has a features header, 0 if it doesn't
        has_samples_header (Integer): 1 if the testSet has a samples header, 0 if it doesn't
        training_labels_filename (String): the filename of the training labels
        length_of_features_filename (String): the filename with the length of the features of the training set, taken
        from step 02.
        length_of_features_from_training_filename: (String) the filename with the length of features of the training set
        (from step 04)
        output_folder (String): the output folder
        selected_comorbidities_string: (String) the selected comorbidities in the step 04 that have been deleted from
        the original dataset
Output:
        result_labels.txt: the predicted labels by the model

Example run:

    For multi-class problem:
        python3.6 testing_multibiomarker_predictive_analytics_model_backend.py Input/multi/test_dataset.txt
        Input/multi/test_labels.txt Input/multi/maximums.txt Input/multi/minimums.txt
        Input/multi/averages_for_missing_values_imputation.txt Input/multi/features_list.txt 1 1
        Input/multi/model_2018_10_09 0 0 "" 8 0 0 Input/multi/training_labels.txt Input/multi/length_of_features.txt
        Input/multi/length_of_features_from_training.txt Output_Folder/ "Feature_0,Feature_1"

    For regression:
        python3.6 testing_multibiomarker_predictive_analytics_model_backend.py Input/regr/test_dataset.txt
        Input/regr/test_labels.txt Input/regr/maximums.txt Input/regr/minimums.txt
        Input/regr/averages_for_missing_values_imputation.txt Input/regr/features_list.txt 1 1
        Input/regr/model_2018_10_09 1 0 "" 8 0 0 Input/regr/training_labels.txt Input/regr/length_of_features.txt
        Input/regr/length_of_features_from_training.txt Output_folder/  "Feature_0,Feature_1"

    For two class:
        python3.6 testing_multibiomarker_predictive_analytics_model_backend.py Input/twoclass/new/new2/test_dataset.txt
        Input/twoclass/new/new2/test_labels.txt Input/twoclass/new/new2/maximums_tr.txt
        Input/twoclass/new/new2/minimums_tr.txt Input/twoclass/new/new2/averages_for_missing_values_imputation_tr.txt
        Input/twoclass/new/new2/features_list.txt 1 1 Input/twoclass/new/new2/model.txt 2 0 "" 8 0 0
        Input/twoclass/new/new2/training_labels.txt Input/twoclass/new/new2/length_of_features.txt
        Input/twoclass/new/new2/length_of_features_from_training.txt Output_folder/  "Feature_0,Feature_1"
"""
import sys
import os
import time
import configparser
import datetime
import logging
import numpy as np
import csv
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
import shutil
import math

import dataset_testing_preprocessing as preprocess
import twoclass_predictor as twoclass
import regression_predictor as regression
import multiclass_predictor as multiclass
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

def initLogging():
    """
    Purpose: sets the logging configurations and initiates logging
    """
    config = configparser.ConfigParser()
    scriptPath = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
    scriptParentPath = os.path.abspath(os.path.join(scriptPath, os.pardir))
    configParentPath = os.path.abspath(os.path.join(scriptParentPath, os.pardir))
    config.read(configParentPath + '/insybio.ini')

    todaystr = datetime.date.today().strftime("%Y%m%d")
    logging.basicConfig(filename="/media/data0/_backendlogs/biomarkers_reports.log",
                        level=logging.DEBUG, format='%(asctime)s\t %(levelname)s\t%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')


def select_predictor(dataset_initial, features, models_zip, selection_flag, training_labels_filename, output_folder,
                     thread_num, testset_labels_filename, user='unknown', jobid=0, pid=0):
    """
    Selects which predictor to run according to the values of the selection_flag.
    :param dataset_initial: (2d array): array with the dataset and its feature values
    :param features: (list): datasets features names list
    :param models_zip: (String) path to zip file with model files and supplementary files
    :param selection_flag: (integer) 0 for multi-class, 1 for regression, 2 for two-class
    :param training_labels_filename: (file) original dataset labels used for training these models
    :param output_folder: path for runtime and produced files
    :param thread_num: (integer) number of threads used for parallel calculations
    :param testset_labels_filename: optional file with testset known labels, in odrder to calculate metrics
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    """
    result = []

    result, modelfiles = parse_models_folder(models_zip, output_folder + 'models/', user, jobid, pid)
    if not result:
        return [0, modelfiles]
    if selection_flag == 0:
        result = multiclass.predictor(dataset_initial, features, modelfiles[0], modelfiles[2], modelfiles[1],
                                      thread_num, user, jobid, pid)
    elif selection_flag == 1:
        result = regression.predictor(dataset_initial, features, modelfiles[0], modelfiles[2], modelfiles[1],
                                      thread_num, user, jobid, pid)
    elif selection_flag == 2:
        result = twoclass.predictor(dataset_initial, features, modelfiles[0], modelfiles[2], modelfiles[1], thread_num,
                                    user, jobid, pid)
    if result[0]:
        if testset_labels_filename:
            error = calculate_metrics(result[1], testset_labels_filename, training_labels_filename, selection_flag,
                                      output_folder, user, jobid, pid)
            if error[0]:
                return [0, error[1]]
        create_labels_file(result[1], training_labels_filename, selection_flag, output_folder, user, jobid, pid)
        return [1, "Successful completion!"]
    return [0, result[1]]


def parse_models_folder(zip_folder, out_folder, user='unknown', jobid=0, pid=0):
    """
    Decompress the model folder, and create the appropriate files and variables
    :param zip_folder:
    :param out_folder:
    :param user:
    :param jobid:
    :param pid:
    :return:
    """
    model_files = [[], '', '']
    try:
        shutil.unpack_archive(zip_folder, out_folder, 'zip')
        models = os.listdir(out_folder)
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError, could not unpack archived model files.".format(pid, jobid,
                                                                                                          user))
        return 0, "Error, could not unpack archived model files."
    for file in models:
        if file.endswith(".pkl") or file.endswith(".pkl.z") or file.endswith(".hdf5"):
            model_files[0].append(out_folder + file)
        elif file.endswith("classifierChain.csv"):
            model_files[1] = out_folder + file
        else:
            model_files[2] = out_folder + file
    model_files[0] = sorted(model_files[0])
    return True, model_files


def calculate_metrics(predictions, known_labels_file, training_labels_file, selection_flag, outfolder,
                      user='unknown', jobid=0, pid=0):
    """
    If Testing labels are provided calculate metrics and write them in metrics.txt file
    :param predictions: list with predicted labels created from the predictors (twoclass and multiclass labels are
    transformed to integers)
    :param known_labels_file: Test set known labels, to calculate metrics
    :param training_labels_file: Training labels, in order to create an association list with twoclass/multiclass labels
    and their integer values
    :param selection_flag: type of classifier/predictor used, 0 multiclass, 1 regression and 2 twoclass
    :param outfolder: folder path for the produced files
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's PID
    :return:
    """

    if selection_flag == 0:
        # predictions = read_csv_to_string(predicted_labels_file)
        test_labels = parse_labels(known_labels_file, len(predictions), selection_flag,
                                   parse_training_labels(training_labels_file), user, jobid, pid)
        multiclass_metrics(predictions, test_labels, outfolder, user, jobid, pid)
    elif selection_flag == 1:
        # predictions = read_csv_to_float(predicted_labels_file)
        test_labels = parse_labels(known_labels_file, len(predictions), selection_flag, [], user, jobid, pid)
        regression_metrics(predictions, test_labels, outfolder, user, jobid, pid)
    elif selection_flag == 2:
        # predictions = read_csv_to_string(predicted_labels_file)
        test_labels = parse_labels(known_labels_file, len(predictions), selection_flag,
                                   parse_training_labels(training_labels_file), user, jobid, pid)
        twoclass_metrics(predictions, test_labels, outfolder, user, jobid, pid)
    return False, ''


def regression_metrics(predicted, expected, outfolder, user, jobid, pid):
    """
    calculate regression metrics, Mean squared error, Relative Absolute error and Root Relative Squared error
    :param predicted: predicted labels
    :param expected: expected labels from test labels file
    :param outfolder:  output folder for the metrics.txt file
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:
    """

    mse_accuracy = math.sqrt(mean_squared_error(expected, predicted))
    rae = (sum(abs(np.array(predicted) - np.array(expected))) /
           sum(abs(np.array(expected) - np.array(expected).mean())))
    rrse = math.sqrt(sum((np.array(expected) - np.array(predicted)) ** 2) /
                     sum((np.array(expected) - np.array(expected).mean()) ** 2))

    # Storage of Metrics
    try:
        with open(outfolder + os.sep + "metrics.txt", 'w') as metrics_file:

            metrics_file.write('Root Mean Square Error: {:.4f}\n'.format(mse_accuracy))
            metrics_file.write('Relative Absolute Error: {:.2f} %\n'.format(rae * 100))
            metrics_file.write('Root Relative Squared Error: {:.2f} %\n'.format(rrse * 100))
    except IOError:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in producing metrics file.".format(pid, jobid, user))


def multiclass_metrics(predicted, expected, outfolder, user, jobid, pid):
    """
    Calculate multiclass metrics, Accuracy
    :param predicted: predicted labels
    :param expected: expected labels from test labels file
    :param outfolder:  output folder for the metrics.txt file
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:
    """

    accuracy = accuracy_score(expected, predicted)
    f1 = f1_score(expected, predicted, average='weighted')
    precision = precision_score(expected, predicted, average='weighted')
    recall = recall_score(expected, predicted, average='weighted')
    f2 = (5 * precision * recall) / (4 * precision + recall)

    err = 0.0
    for err_index in range(len(predicted)):
        if predicted[err_index] != expected[err_index]:
            err = err + 1.0
    manhattan_distance = 1.0 - float(err / len(predicted))

    # Storage of Support vectors or trees generated for each model in the Front
    try:
        with open(outfolder + os.sep + "metrics.txt", 'w') as metrics_file:
            metrics_file.write('Testing accuracy: {:.2f} %\n'.format(accuracy * 100))
            metrics_file.write('Testing F1 score: {:.2f} %\n'.format(f1 * 100))
            metrics_file.write('Testing Precision: {:.2f} %\n'.format(precision * 100))
            metrics_file.write('Testing Recall: {:.2f} %\n'.format(recall * 100))
            metrics_file.write('Testing F2 score: {:.2f} %\n'.format(f2 * 100))
            # metrics_file.write('Training ROC AUC: {:.2f} \n'.format(training_roc))
            metrics_file.write('Testing Manhattan Distance: {:.2f} \n'.format(manhattan_distance))

    except IOError:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in producing metrics file.".format(pid, jobid, user))


def twoclass_metrics(predicted, expected, outfolder, user, jobid, pid):
    """
    Calculate twoclass metrics, Accuracy, Specificity and Sencitivity
    :param predicted: predicted labels
    :param expected: expected labels from test labels file
    :param outfolder:  output folder for the metrics.txt file
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:
    """
    print(predicted)
    print(expected)
    # predicted_new = []
    # for i in predicted:
    #     if i[1] >= i[0]:
    #         predicted_x = 1
    #         predicted_new.append(predicted_x)
    #     else:
    #         predicted_x = 0
    #         predicted_new.append(predicted_x)
    TP, FP, TN, FN = ROC_measures(expected, predicted)
    accuracy = float((TP + TN) / (TP + TN + FP + FN))
    f1 = float(TP / (TP + 0.5 * (FP + FN)))
    f2 = float(5 * TP / (5 * TP + 4 * FN + FP))
    roc = roc_auc_score(expected, predicted)

    fpr, tpr, _ = roc_curve(expected, predicted)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Roc curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate or (1- Specificity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(outfolder + os.sep + "roccurve.png")
    plt.clf()

    try:
        specificity = float(TN / (TN + FP))
    except ZeroDivisionError:
        specificity = 0
    try:
        sensitivity = float(TP / (TP + FN))
    except ZeroDivisionError:
        sensitivity = 0
    # Storage of Support vectors or trees generated for each model in the Front
    try:
        with open(outfolder + os.sep + "metrics.txt", 'w') as metrics_file:

            metrics_file.write('Testing accuracy: {:.2f} %\n'.format(accuracy * 100))
            metrics_file.write('Testing specificity: {:.2f} %\n'.format(specificity * 100))
            metrics_file.write('Testing sensitivity: {:.2f} %\n'.format(sensitivity * 100))

            metrics_file.write('Testing F1 score: {:.2f} %\n'.format(f1 * 100))
            metrics_file.write('Testing F2 score: {:.2f} %\n'.format(f2 * 100))
            metrics_file.write('Testing ROC AUC: {:.2f} \n'.format(roc))
    except IOError:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in producing metrics file.".format(pid, jobid, user))


def ROC_measures(y_actual, y_predict):  # function to compute tp, tn, fp, fn for ROC
    """
    Calculate Roc measures, True positives, true negatives, false positives and false negatives
    :param y_actual: Actual labels
    :param y_predict: Predicted labels
    :return: the values of TP, FP, TN and FN
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_predict)):
        if y_actual[i] == y_predict[i] == 1:
            TP += 1
        if y_predict[i] == 1 and y_actual[i] != y_predict[i]:
            FP += 1
        if y_actual[i] == y_predict[i] == 0:
            TN += 1
        if y_predict[i] == 0 and y_actual[i] != y_predict[i]:
            FN += 1
    return TP, FP, TN, FN


def create_labels_file(predictions, training_labels_file, selection_flag, output_folder, user, jobid, pid):
    """
    Write the predicted labels in output file, if they were originally alphanumeric transform them
    :param predictions: list of predictions
    :param training_labels_file: training labels file
    :param selection_flag: selection flag between multiclass, regression and twoclass
    :param output_folder: outputfolder for the created labels file
    :param user: thos job's user
    :param jobid: this job's ID
    :param pid:this job's PID
    :return:
    """

    if selection_flag == 0:
        # predictions = read_csv_to_string(predicted_labels_file)
        predictions = transform_labels_to_alpha_multiclass(predictions, parse_training_labels(training_labels_file))
    elif selection_flag == 2:
        # predictions = read_csv_to_string(predicted_labels_file)
        predictions = transform_labels_to_alpha_twoclass(predictions, parse_training_labels(training_labels_file))
    try:
        with open(output_folder + "result_labels.txt", "w") as result_labels_fid:
            for label in predictions:
                result_labels_fid.write(str(label) + "\t")
    except IOError:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in producing predicted labels file.".format(pid, jobid, user))


def parse_training_labels(labels_filename):
    """
    Find original training Unique Labels Set
    :param labels_filename: training labels file
    :return: set of unique labels
    """

    delim_labels = preprocess.find_delimiter(labels_filename)

    # Parse input files
    labels = list()
    with open(labels_filename, "r") as labels_fid:
        for line in csv.reader(labels_fid, delimiter=delim_labels):
            for word in line:
                labels.append(word.strip())
    unique_labels = sorted(list(set(labels)))

    return unique_labels


def parse_labels(labels_filename, data_length, selection_flag, unique_labels, user, jobid, pid):
    """
    Parse labels file according to each estimator selected
    :param labels_filename: filename of the labels file
    :param data_length: the length of the input data
    :param selection_flag: which estimator will be used, in order to know the type of labels
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's PID
    :return: labels list
    """

    delim_labels = preprocess.find_delimiter(labels_filename)

    # Parse input files
    labels = list()
    with open(labels_filename, "r") as labels_fid:
        for line in csv.reader(labels_fid, delimiter=delim_labels):
            for word in line:
                if selection_flag == 1:
                    labels.append(float(word.strip()))
                else:
                    labels.append(word.strip())
    label_length_diff = len(labels) - data_length

    if label_length_diff > 1:
        logging.error("PID:{}\tJOB:{}\tUSER:{}\tLabels file has {} more values than samples in dataset.".format(
            pid, jobid, user, label_length_diff))
        return [None, "Labels file has more values than samples in dataset"]
    elif label_length_diff < 0:
        logging.error("PID:{}\tJOB:{}\tUSER:{}\tLabels file has {} less values than samples in dataset.".format(
            pid, jobid, user, abs(label_length_diff)))
        return [None, "Labels file has less values than samples in dataset"]
    elif label_length_diff == 1:
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tLabels file has a header".format(pid, jobid, user))
        labels = labels[1:]

    logging.info("PID:{}\tJOB:{}\tUSER:{}\tLabels file was successfully parsed!".format(pid, jobid, user))

    # unique_labels = sorted(list(set(labels)))
    if selection_flag == 0 or selection_flag == 2:
        try:
            float(unique_labels[0])
            labels = [int(lbl) for lbl in labels]
        except ValueError:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tFound  non numeric labels.".format(pid, jobid, user))
            if selection_flag == 0:
                labels = transform_labels_to_numeric_multiclass(labels, unique_labels)
            else:
                labels = transform_labels_to_numeric_twoclass(labels, unique_labels)
        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tUnexpected Error.".format(pid, jobid, user))
            return [None, "Unexpected Error in reading labels file."]
    return labels


def transform_labels_to_numeric_twoclass(labels, unique_labels):
    """
    Transforms alphanumeric labels to numeric ones in the case of two classes.
    :param labels: the input labels
    :param unique_labels: the list with the two unique labels
    :return: new_labels: the encoded 0-1 new labels
    """

    labels_dict = dict()
    labels_dict[unique_labels[0]] = 0
    labels_dict[unique_labels[1]] = 1
    # logging.info(str(labels_dict))
    new_labels = [labels_dict[x] for x in labels]
    # logging.info("Labels transformed to numeric successfully.")
    return new_labels


def transform_labels_to_alpha_twoclass(labels, unique_labels):
    """
    Transforms numeric labels back to alphanumeric according to given unique_labels one to one mapping.
    :param labels: the input labels
    :param unique_labels: the list with the two unique labels
    :return: new_labels: the decoded labels with alphanumeric names
    """

    new_labels = []
    for x in labels:
        if x == 0:
            new_labels.append(unique_labels[0])
        elif x == 1:
            new_labels.append(unique_labels[1])
    # logging.info("Labels transformed to alphanumeric successfully.")
    return new_labels


def transform_labels_to_numeric_multiclass(labels, unique_labels):
    """
    Transforms alphanumeric labels to numeric ones in the case of multiple classes.
    :param labels: the input labels
    :param unique_labels: the list with the two unique labels
    :return: new_labels: the encoded 0-1 new labels
    """

    labels_dict = dict()
    for i, label in enumerate(unique_labels):
        labels_dict[label] = i

    # logging.info(str(labels_dict))
    new_labels = [int(labels_dict[x]) for x in labels]
    # logging.info("Labels transformed to numeric successfully.")
    return new_labels


def transform_labels_to_alpha_multiclass(labels, unique_labels):
    """
    Transforms numeric labels back to alphanumeric according to given unique_labels one to one mapping.
    :param labels: the input labels
    :param unique_labels: the list with the two unique labels
    :return: new_labels: the decoded labels with alphanumeric names
    """

    labels_dict = dict()
    for i, label in enumerate(unique_labels):
        labels_dict[str(i)] = label

    # logging.info(str(labels_dict))
    new_labels = [labels_dict[str(x)] for x in labels]

    # logging.info("Labels transformed to alphanumeric successfully.")
    return new_labels


def print_data_all(data, markers, labels, folder_name, filename, user='unknown', jobid=0, pid=0):
    """
    Writes data and labels to a file.
    :param data: input data (list of lists)
    :param markers: input biomarkers (list)
    :param labels: input labels (list)
    :param folder_name: output folder
    :param filename: output filename
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:
    """
    try:
        with open(folder_name + filename, 'w') as file:
            message = ''
            for i in range(len(data[0])):
                message = message + '\t' + labels[i]
            message += '\n'
            for i in range(len(data)):
                message += markers[i] + '\t'
                message += '\t'.join(map(str, data[i]))
                # for j in range(len(data[0])):
                #     message += '\t' + str(data[i][j])
                message += '\n'
            file.write(message)
            logging.info("PID:{}\tJOB:{}\tUSER:{}\tData have been written to file successfully.".format(pid, jobid,
                                                                                                        user))
            return True
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has been raised during parsing the selected"
                          " features.".format(pid, jobid, user))
        return False


def parse_selected_features_string(astring):
    """
    Parses a string and strips it from commas or newline characters.
    :param astring: the input string with comma separated or newline separated values
    :return: A list with the substrings of the original string.
    """

    # print(astring)
    if "," in astring:
        return astring.split(",")
    elif "\\n" in astring:
        return astring.split("\\n")
    else:
        # raise ValueError("The string doesn't contain comma separated values or newline separated values !")
        return astring


def write_one_line_tab_delimited_file(proteins, filename, user='unknown', jobid=0, pid=0):
    """
    Writes an one dimensional list to an one line tab delimited file.
    :param proteins: (list): the one dimensional list
    :param filename: (string): the filename where the data are being written
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:
    """
    try:
        with open(filename, 'w') as handle:
            handle.write("\t".join(proteins))
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has been raised during I/O in a file.".format(
            pid, jobid, user))


def parse_data_generic(data_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses dataset and splits it into Features, sample_name and data lists, expecting both feature and sample headers
    :param data_filename: dataset filename
    :param delimiter: the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: a list of three lists, [features, data, samples].
    """

    num_of_lines = 0
    features = list()
    data = list()
    samples = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            if num_of_lines == 0:
                for j, value in enumerate(line):
                    if j > 0:
                        samples.append(value.strip())
            else:
                data.append([])
                for j, value in enumerate(line):
                    if j == 0:
                        features.append(value.strip())
                    else:
                        if value != '' and value != "#VALUE!":
                            data[num_of_lines - 1].append(float(value))
                        else:
                            data[num_of_lines - 1].append('')
            num_of_lines += 1
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return [features, data, samples]


def new_parse_data(data_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses dataset and splits it into Features, sample_name and data lists, expecting only feature headers
    :param data_filename: dataset filename
    :param delimiter: the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: a list of two lists, [features, data].
    """

    num_of_lines = 0
    features = list()
    data = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            data.append([])
            for j, value in enumerate(line):
                if j == 0:
                    features.append(value)
                else:
                    if value != '' and value != "#VALUE!":
                        data[num_of_lines].append(float(value))
                    else:
                        data[num_of_lines].append('')
            num_of_lines += 1
    # print('Data were successfully parsed!')
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return [features, data]


def parse_data_with_only_samples(data_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses dataset and splits it into Features, sample_name and data lists, expecting only sample headers
    :param data_filename: dataset filename
    :param delimiter: the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: a list of two lists, [data, samples].
    """

    num_of_lines = 0
    data = list()
    samples = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            for value in line:
                if num_of_lines == 0:
                    samples.append(value.strip())
                else:
                    data.append([])
                    if value != '' and value != "#VALUE!":
                        data[num_of_lines - 1].append(float(line[j]))
                    else:
                        data[num_of_lines - 1].append('')
                num_of_lines += 1
    # print('Data were successfully parsed!')
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return [data, samples]


def parse_only_dataset(dataset_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses dataset and splits it into Features, sample_name and data lists, expecting no headers
    :param dataset_filename: dataset filename
    :param delimiter: the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: a list data, data.
    """

    data = list()
    num_of_lines = 0
    with open(dataset_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            data.append([])
            for value in line:
                if value != '' and value != '#VALUE!':
                    data[num_of_lines].append(float(value))
                else:
                    data[num_of_lines].append('')
            num_of_lines += 1
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return data


# function to read csv cells as float, used for maximums and minimums files
def read_csv_to_float(file_name, user='unknown', jobid=0, pid=0):
    """
    Convert csv file to list of floats
    :param file_name: csv file
    :param user: this job's user
    :param jobid: this job's D
    :param pid:  this job's PID
    :return: list of floats
    """
    data = []
    with open(file_name, 'r') as f:
        read = csv.reader(f)
        for row in read:  # convert yo float
            for cell in row:
                try:
                    data.append(float(cell))
                except ValueError:
                    logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in csv reader {} not convertible to "
                                      "float.".format(pid, jobid, user, cell))
    return data


# function to read csv cells as float, used for labels files
def read_csv_to_string(file_name, user='unknown', jobid=0, pid=0):
    """
    onvert csv file to list of strings
    :param file_name: csv file
    :param user: this job's user
    :param jobid: this job's D
    :param pid:  this job's PID
    :return: list of strings
    """
    data = []
    with open(file_name, 'r') as f:
        read = csv.reader(f)
        for row in read:  # convert yo float
            for cell in row:
                try:
                    data.append(str(cell))
                except ValueError:
                    logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in csv reader {} not convertible to "
                                      "string.".format(pid, jobid, user, cell))
    return data


def preprocess_testing_dataset(test_filename, maximums_filename, minimums_filename, averages_filename,
                               training_features_file, missing_imputation_method, normalization_method,
                               has_feature_headers, has_sample_headers, variables_for_normalization_string,
                               output_folder, data_been_preprocessed_flag, user='unknown', jobid=0, pid=0):
    """
    Perform the same preprocessing steps as the training dataset, or based to the selected specifications to the
     testing dataset
    :param test_filename: Test set file
    :param maximums_filename: maximums filename, used for arithmetic normalization
    :param minimums_filename:  minimums filename, used for arithmetic normalization
    :param averages_filename: averages filename, used for Averages imputation
    :param training_features_file: selected training features file
    :param missing_imputation_method:  Missing imputation to use, 1 for Average 2 for KNN
    :param normalization_method: Normalization method to use 1 for Arithmetic and 2 for logarithmic
    :param has_feature_headers: if the input dataset has features headers
    :param has_sample_headers: if the input dataset has sample headers
    :param variables_for_normalization_string: list of features to normalize seperately
    :param output_folder: output folder for created files
    :param data_been_preprocessed_flag: If the testing dataset needs preprocessing
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's PID
    :return: True, preprocessed dataset and feature list if successful, ot False and the error message
    """

    # ######## use min/ max saved during training and write new function
    # read max-min csvs as list, this is full list of omics before duplicate averaging
    if maximums_filename and minimums_filename and normalization_method == 1 and data_been_preprocessed_flag == 0:
        maximums = read_csv_to_float(maximums_filename, user, jobid, pid)
        minimums = read_csv_to_float(minimums_filename, user, jobid, pid)
    else:
        maximums = []
        minimums = []
    print('missing 2',  missing_imputation_method)
    end_code, preprocessed_dataset, testing_features = preprocess.preprocess_data(
        test_filename, maximums, minimums, averages_filename, training_features_file, missing_imputation_method,
        normalization_method, output_folder, data_been_preprocessed_flag, variables_for_normalization_string,
        has_feature_headers, has_sample_headers, user=user, jobid=jobid, pid=pid)
    if end_code:
        # return the preprocessed dataset and features
        return True, preprocessed_dataset, testing_features
    else:
        # return that an error occurred and a message
        return False, preprocessed_dataset, ''


def run_all(testset_filename, testset_labels_filename, maximums_filename, minimums_filename, averages_filename,
            features_filename, missing_imputation_method, normalization_method, model_filename, selection_flag,
            data_been_preprocessed_flag, variables_for_normalization_string, filetype, has_features_header,
            has_samples_header, training_labels_filename, length_of_features_filename,
            length_of_features_from_training_filename, output_folder, selected_comorbidities_string, thread_num=2,
            user='unknown', jobid=0, pid=0):
    """
    Selects which predictor to run according to the values of the selection_flag.

    Args:
        testset_filename (string): the testset filename
        testset_labels_filename (string): the testset labels filename
        maximums_filename: (String) the filename with the maximum values of each feature
        minimums_filename: (String) the filename with the minimum values of each feature
        averages_filename: (String) the filename with the average values for each sample
        features_filename: (String) the filename with the indexes of the selected features extracted from the training
         step
        normalization_method: (Integer) 1 for arithmetic, 2 for logarithmic
        missing_imputation_method: (Integer) 1 for average imputation, 2 for KNN imputation
        model_filename: (String) the model filename
        selection_flag (integer): 0 for multi-class, 1 for regression, 2 for two-class
        data_been_preprocessed_flag: (integer) 0 if data haven't been preprocessed and 1 if they have
        variables_for_normalization_string: (String) the selected variables for normalization as a string with comma or
         newline separated strings, eg. "ABC,DEF,GHI"
        filetype: 7 if it is a file from bionets with only features, not 7 if it's  not a file from bionets
        has_features_header: 1 if it has features header, 0 if it doesn't have
        has_samples_header: 1 if it has samples header, 0 if it doesn't have
        training_labels_filename (String): the filename of the training labels
        length_of_features_filename (String): the filename with the length of the features of the training set, taken
        from step 02.
        length_of_features_from_training_filename: (String) the filename with the length of features of the training set
         (from step 04)
        output_folder (String): the output folder
        selected_comorbidities_string: (String) the selected comorbidities in the step 04 that have been deleted from
        the original dataset
        thread_num: number of available  dor parallel processes
        user (String): this job's username
        jobid (Integer): this job's ID in biomarkers_jobs table
        pid (Integer): this job's PID
    """

    # initLogging()

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print('MISSING', missing_imputation_method)
    error, testset, features = preprocess_testing_dataset(
        testset_filename, maximums_filename, minimums_filename, averages_filename, features_filename,
        missing_imputation_method, normalization_method, has_features_header, has_samples_header,
        variables_for_normalization_string, output_folder, data_been_preprocessed_flag, user, jobid, pid)

    if error:
        try:
            result = select_predictor(testset, features, model_filename, selection_flag, training_labels_filename,
                                      output_folder, thread_num, testset_labels_filename, user, jobid, pid)
        except ValueError:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException while predicting.".format(pid, jobid, user))
            return [0, "Exception while predicting, please check if the dataset and labels are correct for this model"]
    else:
        logging.error("PID:{}\tJOB:{}\tUSER:{}\tException while running preprocessing.".format(pid, jobid, user))
        return [0, "Exception while running parsing and preprocessing: {}".format(testset), '']

    return result


if __name__ == "__main__":
    testset_filename1 = sys.argv[1]
    testset_labels_filename1 = sys.argv[2]
    maximums_filename1 = sys.argv[3]
    minimums_filename1 = sys.argv[4]
    averages_filename1 = sys.argv[5]    
    maximums_filename1 = ""
    minimums_filename1 = ""
    averages_filename1 = ""
    features_filename1 =  sys.argv[6]
    missing_imputation_method1 = int(sys.argv[7])
    print(missing_imputation_method1)
    normalization_method1 = int(sys.argv[8])
    model_filename1 = sys.argv[9]
    selection_flag1 = int(sys.argv[10])
    print(selection_flag1)
    data_been_preprocessed_flag1 = int(sys.argv[11])
    selected_commorbidities_string1 = sys.argv[12]
    selected_commorbidities_string1 = ""
    filetype1 = int(sys.argv[13])
    has_features_header1 = int(sys.argv[14])
    has_samples_header1 = int(sys.argv[15])
    training_labels_filename1 = sys.argv[16]
    length_of_features_filename1 = sys.argv[17]
    length_of_features_filename1 = ""
    length_of_features_from_training_filename1 = sys.argv[18]
    output_folder1 = sys.argv[19]
    selected_comorbidities_string1 = sys.argv[20]
    selected_comorbidities_string1 = ""
    initLogging()
    ret = run_all(testset_filename1, testset_labels_filename1, maximums_filename1, minimums_filename1,
                  averages_filename1, features_filename1, missing_imputation_method1, normalization_method1,
                  model_filename1, selection_flag1, data_been_preprocessed_flag1, selected_commorbidities_string1,
                  filetype1, has_features_header1, has_samples_header1, training_labels_filename1,
                  length_of_features_filename1, length_of_features_from_training_filename1, output_folder1,
                  selected_comorbidities_string1)
    print(ret)
