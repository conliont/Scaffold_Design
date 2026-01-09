'''
run the script for optimization:

python3 /PATH/bioreactor.py 
--output_folder /PATH/folder/ 
--model_dir /PATH 
--scaling_dir /PATH
--desired_outputs  12 70 8 0.05 0.4 --population 10 --generations 20


run the script for prediction:

python3 /home/insybio/Downloads/bioreactor.py 
--sphere_diameter 0.4 --sphere_distance 0.4 --delaunay_mesh 0.21 
--delaunay_spacing 0.1 --output_folder /PATH/folder/ 
--model_dir /PATH/bioreactor 
--scaling_dir /PATH/bioreactor

'''

from __future__ import print_function
import os
import random
import statistics
import csv
import numpy as np
import scipy.stats as st
import time
import math
import sys
import logging
import copy
from collections import Counter

from copy import deepcopy
import argparse
import matplotlib.pyplot as plt

from itertools import chain
import rpy2.robjects as robjects
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri, conversion, r, FloatVector
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Activate pandas conversion for R data frames
pandas2ri.activate()

from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import Counter  # for counting classes from labels list
from sklearn.metrics import fbeta_score, make_scorer, r2_score, explained_variance_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, zero_one_loss
from scipy.spatial import distance
from sklearn.base import clone

from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from sklearn import svm

import joblib
from joblib import Parallel, delayed
from joblib.parallel import cpu_count

from mlxtend.frequent_patterns import apriori  # association rule
from mlxtend.preprocessing import TransactionEncoder

import itertools  # for permutation of multi-labels
import json  # for classifier chain dictionary writing for final front1 solutions

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# DB save
import configparser
import psycopg2
import datetime


random.seed()  # Initialize random generator seed with the current local time
indices = []  # hold the indices of features who match to one feature

# Import R packages
Matrix = importr("Matrix")
glmnet = importr("glmnet")


class Config:
    MIN_VALUES = [4.5, 39.1, 0.3, 0.02, 0.09] # area/volume , bone_porosity, connectivity_density, trabecular spacing, trabecular thickness
    MAX_VALUES = [28.2, 100, 22.9, 0.2, 0.93]
    PATHS = {
        "model_dir": None,
        "scaling_dir": None
    }

    @staticmethod
    def update_paths(model_dir, scaling_dir):
        Config.PATHS.update({
            "model_dir": model_dir,
            "scaling_dir": scaling_dir
        })


class ModelLoader:
    @staticmethod
    def load_r_models(model_dir, model_names):
        """Load multiple R models from the specified directory."""
        return [read_r_model(os.path.join(model_dir, f"cvmodel_{name}.rds")) for name in model_names]

    @staticmethod
    def load_scaling_factors(scaling_dir, keys):
        """Load scaling factors for multiple features from the specified directory."""
        return {
            key: parse_scaling_csv(os.path.join(scaling_dir, f"{key}_scaling_parameters.csv"))
            for key in keys
        }

# Function to load an R model from an `.rds` file
def read_r_model(file_name):
    """Read an R model from the specified `.rds` file."""
    readRDS = robjects.r['readRDS']
    model = readRDS(file_name)
    return model

# Function to parse scaling factors from a CSV file
def parse_scaling_csv(file_path):
    """Parse scaling factors (mean, std deviation) from a CSV file."""
    scaling_df = pd.read_csv(file_path)
    means = scaling_df.set_index('Feature')['Mean'].to_dict()
    std_devs = scaling_df.set_index('Feature')['StdDev'].to_dict()
    return {'means': means, 'std_devs': std_devs}

def scale_input_data(input_data, means, std_devs):
    """ Scales the input data using the provided mean and std deviation."""
    # Perform element-wise scaling
    return (input_data - means) / std_devs

def bound_output(predicted_values):
    """ Ensure that each predicted value is within the allowed min and max limits. """
    bounded_values = []
    for i, value in enumerate(predicted_values):
        min_val = Config.MIN_VALUES[i]
        max_val = Config.MAX_VALUES[i]
        if value < min_val:
            bounded_values.append(min_val)
        elif value > max_val:
            bounded_values.append(max_val)
        else:
            bounded_values.append(value)
    return bounded_values

def rescale_output_data(scaled_output, mean, std_dev):
    """Rescales the output data to its original scale."""
    return (scaled_output * std_dev) + mean

    

class BioreactorPrediction:

    def __init__(self):
        self.models = ModelLoader.load_r_models(Config.PATHS["model_dir"], ["area_volume", "bone_porosity", "connectivity_density", "trabecular_thickness", "trabecular_spacing"])
        self.scaling_params = ModelLoader.load_scaling_factors(Config.PATHS["scaling_dir"], ["area", "porosity", "density", "thickness", "spacing"])

        
    def predict(self, sphere_diameter, sphere_distance, delaunay_mesh, delaunay_spacing):
        """
        Uses the models to make predictions based on the provided input parameters.
        """
        r_predict = r['predict']
        predictions = []

        # Prepare the raw input data as a numpy array
        raw_input_data = np.array([sphere_diameter, sphere_distance, delaunay_mesh, delaunay_spacing])

        for i, (model, model_name) in enumerate(zip(self.models, self.scaling_params.keys())):
            try:
                # Load and print scaling parameters for the model
                scaling_info = self.scaling_params[model_name]
                means = np.array(list(scaling_info['means'].values())[:-1])  # exclude output mean
                std_devs = np.array(list(scaling_info['std_devs'].values())[:-1])  # exclude output std dev
                
                #print("Scaling parameters for model:", model_name, scaling_info)

                # Scale the data using the saved parameters for the specific model
                scaled_data = scale_input_data(raw_input_data, means, std_devs)
                print(f"Scaled input in prediction: {scaled_data}")
                print(f"Prediction scaling parameters for model {model_name}: Means={means}, StdDevs={std_devs}")

                # Convert the scaled data to an R matrix
                input_r_matrix = r['as.matrix'](numpy2ri.py2rpy(scaled_data.reshape(1, -1)))

                # Use the lambda.min value for prediction if available
                if 'cv.glmnet' in r['class'](model):
                    lambda_min = model.rx2('lambda.min')[0]
                    prediction = r_predict(model, newx=input_r_matrix, s=lambda_min, type="response")
                else:
                    prediction = r_predict(model, newx=input_r_matrix)

                predictions.append(float(prediction[0]))  # Append the prediction
              
            except Exception as e:
                print(f"Error predicting with model {model_name}: {e}")
                predictions.append(None)
        '''
        # Rescale each prediction to the original scale using saved output scaling parameters
        for i, prediction in enumerate(predictions):
            output_name = list(self.scaling_params.keys())[i]
            output_mean = self.scaling_params[output_name]['means']['output']
            output_std_dev = self.scaling_params[output_name]['std_devs']['output']
            
            # Rescale and update prediction
            if prediction is not None:
                predictions[i] = rescale_output_data(prediction, output_mean, output_std_dev)
        '''
        # Use the bound_output function to ensure predictions are within the allowed bounds
        bounded_predictions = bound_output(predictions)
        
        # Debugging: Print the predictions before and after bounding
        print(f"Predictions before bounding: {predictions}")
        print(f"Predictions after bounding: {bounded_predictions}")

        predictions = bounded_predictions

        return predictions


class OptimizationProcess:

    def __init__(self, min_values, max_values, population, generations, output_folder,
                 goal_values, mutation_prob=0.2, arith_crossover_prob=1.00, 
                 two_point_crossover=0.00, convergeThreshold=0.001):

        self.pathFeature = output_folder

        logging.info("Initializing OptimizationProcess with population size: %d, generations: %d", population, generations)

        # Load models and scaling parameters
        self.models = ModelLoader.load_r_models(Config.PATHS["model_dir"], ["area_volume", "bone_porosity", "connectivity_density", "trabecular_thickness", "trabecular_spacing"])
        self.scaling_params = ModelLoader.load_scaling_factors(Config.PATHS["scaling_dir"], ["area", "porosity", "density", "thickness", "spacing"])
        
        # Optimization parameters
        self.min_values = min_values  # Min values for the input parameters
        self.max_values = max_values  # Max values for the input parameters
        self.population = int(population)  # Population size for optimization
        self.generations = int(generations)  # Number of generations for optimization
        self.mutation_prob = [0.001, float(mutation_prob)]  # Mutation probability
        self.arithCross_prob = float(arith_crossover_prob)  # Arithmetic crossover probability
        self.twoCross_prob = float(two_point_crossover)  # Two-point crossover probability
        
        # Goal values and goal significances based on fitness functions
        self.goal_values = goal_values  # Goal values for optimization
        self.goal_significance = [1] * 5 + [0]  # 5 fitness functions + 1 for the average
        self.predictions_storage = [[] for _ in range(generations)]
        # Set the convergence threshold
        self.convergeThreshold = convergeThreshold  # Add this attribute to control convergence
        self.goal_header = "Fitness_Model_1, Fitness_Model_2, Fitness_Model_3, Fitness_Model_4, Fitness_Model_5, Average_Fitness"

        self.avgSimThr = 0.90  # average similarity between the members of the population and its best member
        self.gaussMut_varPro = [0.1, 0.5]  # gaussian variance proportion, min=0.1 and max = 0.5
        # model variables for goal values (post pareto front goal tuning) for visualisation
        self.max_eval_per_generation = [0] * self.generations  # best values per goal per generation
        self.average_eval_per_generation = [0] * self.generations  # weighted avg value per generation across all goals
        self.sum_ranked_eval_per_generation = [0] * self.generations  # Sum of weighted average goal per generation
        self.average_ranked_eval_per_generation = [0] * self.generations  # Avg of weighted average goal per generation
        self.max_sol_per_generation = [0] * self.generations  # best sol per generation
  
        self.Pareto_Wheel_Output = ''  # output message from Pareto front and roulette wheel

        # Output message for tracking progress
        self.output_message = ''  # Output messages during optimization


    # Accessor functions for parameters
    def setPopulation(self, value):
        self.population = value

    def getPopulation(self):
        return self.population

    def setGenerations(self, value):
        self.generations = value

    def getGenerations(self):
        return self.generations

    def setFolds(self, value):
        self.folds = value

    def getFolds(self):
        return self.folds
    
    # Simplified setter methods for each fitness function
    def setGoalSignificanceFeature1(self, value):
        self.goal_significance[0] = float(value)

    def setGoalSignificanceFeature2(self, value):
        self.goal_significance[1] = float(value)

    def setGoalSignificanceFeature3(self, value):
        self.goal_significance[2] = float(value)

    def setGoalSignificanceFeature4(self, value):
        self.goal_significance[3] = float(value)

    def setGoalSignificanceFeature5(self, value):
        self.goal_significance[4] = float(value)

    def getGoalSignificances(self):
        return self.goal_significance
    '''
    def setGoalSignificancesByUser(self, feature_significance, mse_significance, model_complexity_significance):
        self.setGoalSignificanceFeatures(feature_significance)
        self.setGoalSignificanceMSE(mse_significance)
        self.setGoalSignificanceModelComplexity(model_complexity_significance)
    '''
    def setGoalSignificancesByUserList(self, goal_significances):
        self.setGoalSignificanceFeature1(goal_significances[0])
        self.setGoalSignificanceFeature2(goal_significances[1])
        self.setGoalSignificanceFeature3(goal_significances[2])
        self.setGoalSignificanceFeature4(goal_significances[3])
        self.setGoalSignificanceFeature5(goal_significances[4])
    
    def setPaths(self, featurepath):
        self.pathFeature = featurepath

    def getPathFeature(self):
        return self.pathFeature

    # This function initializes a population of solutions (float vectors) for optimisation parameters
         
    def initialize(self):
        population = self.getPopulation()  # Get the population size
        #logging.info("Initializing population with size: %d", population)
        min_values = self.min_values  # Now using the passed min_values argument
        max_values = self.max_values  # Now using the passed max_values argument

        # Initialize the population: a list of lists (each list is a chromosome of 4 float values)
        individuals = [[0 for _ in range(4)] for _ in range(population)]  # 4 inputs for each chromosome

        # Generate random float values for each chromosome in the population
        for i in range(len(individuals)):  # For each individual in the population
            for j in range(4):  # For each of the 4 inputs
                # Generate random float values between the corresponding min and max
                individuals[i][j] = random.uniform(min_values[j], max_values[j])
        
        #logging.debug("Sample individual from initialized population: %s", individuals[0])

        self.output_message += "Population initialised for %s individuals \n" % len(individuals)
        print("individuals initialized as:", individuals)
        return individuals    

    
    # This function is used by non_Dominated_Sol() to compare
    # if a sol i.e. goal_value dominates another sol i.e. goal_value
    def dominate(self, solution1, solution2):
        check = 2  # initialise solution comparison check to 2 i.e. equal
        ffs = len(solution1)
        dominate1 = 1
        equal1 = 1
        f = 0
        while f < ffs and dominate1 == 1:
            if solution1[f] > solution2[f]:
                equal1 = 0
            elif solution1[f] == solution2[f]:
                equal1 = 1
            else:
                dominate1 = 0
            f = f + 1

        if dominate1 == 1 and equal1 == 0:
            check = 1
        elif dominate1 == 1 and equal1 == 1:
            check = 2
        else:
            dominate2 = 1
            equal2 = 1
            f = 0
            while f < ffs and dominate2 == 1:
                if solution2[f] > solution1[f]:
                    equal2 = 0
                elif solution2[f] == solution1[f]:
                    do_nothing = 1
                else:
                    dominate2 = 0
                f = f + 1

            if dominate2 == 1 and equal2 == 0:
                check = 3
        
        return check

    def getParetoFront(self, population, goal_values, stopIndx):
        assigned = 0
        fronts = [0] * population  # initialise pareto front for population size
        front = 1
        #  create copy of goal_values so that original goal are not changed. Exclude weighted average
        eval_temp = copy.deepcopy(np.array(goal_values)[:, :stopIndx:1])  # change asarray to array to create copy
        eval_temp = eval_temp.T  # take transpose so that goals are in rows and observations in columns
        # print("Pareto front")
        # eval_goal = copy.deepcopy(goal_values) # take deep copy
        # eval_temp = list(map(list, zip(*eval_goal))) # create transpose of list of lists i.e. goals
        # print("Number of fitness values: %d" % len(eval_temp))
        # print("Population size: %d" % len(eval_temp[0]))

        self.Pareto_Wheel_Output += "Goal length in pareto front {}: \n".format(len(eval_temp))

        number_of_solutions = copy.deepcopy(population)  # deep copy of length of population
        self.Pareto_Wheel_Output += "Solution length in pareto front {}: \n".format(number_of_solutions)

        # ref: Mishra, K. K., & Harit, S. (2010). A fast algorithm for finding the non-dominated set in multi
        # objective optimization. International Journal of Computer Applications, 1(25), 35-39.
        while assigned < population:  # iterate until non-dominated solutions assigned is less than population size
            non_dominated_solutions = [0] * number_of_solutions
            index = [i for i in range(len(eval_temp[0]))]  # generate list of index based on count of goals
            self.Pareto_Wheel_Output += "Index for Ordered List {}: \n".format(index)

            eval_temp_index = list(zip(eval_temp[0], index))  # create list of tuples with goal val of sol. & index
            # sort the goal values in decrease order i.e. highest comes 1st
            self.Pareto_Wheel_Output += "Tuple of goal and list index {}: \n".format(eval_temp_index)
            ordered_list = sorted(range(len(eval_temp_index)), key=lambda k: eval_temp_index[k], reverse=True)
            self.Pareto_Wheel_Output += "Ordered list for Pareto Front {}: \n".format(ordered_list)

            non_dominated_solutions[0] = ordered_list[0]  # assign the 1st goal values as 1st non-dominated sol.
            number_of_non_dominated_solutions = 1  # initialise the non-dominated sol count
            self.Pareto_Wheel_Output += "Non-dominated solution for Pareto Front {}: \n".format(
                non_dominated_solutions)

            for i in range(1, number_of_solutions):  # iterate from 1 to pop size as 1st i.e. 0 index is already
                # selected
                n = 0
                condition = 0
                condition2 = 1
                # print ("in loop")
                while n < number_of_non_dominated_solutions and condition == 0:  # compare ordered list of solutions
                    # pairs
                    solution1 = [0] * (len(eval_temp))  # excludes weighted average goal & classification model
                    solution2 = [0] * (len(eval_temp))  # excludes weighted average goal & classification model

                    for j in range(len(eval_temp)):  # iterate on no of goals; compare solutions by goals
                        solution1[j] = eval_temp[j][ordered_list[i]]
                        solution2[j] = eval_temp[j][ordered_list[n]]

                    check = self.dominate(solution1, solution2)  # compare the goal values in descending order
                    if check == 3:
                        condition = 1
                        condition2 = 0
                    elif check == 1:
                        if number_of_non_dominated_solutions == 1:
                            condition = 1
                            non_dominated_solutions[0] = ordered_list[i]
                        else:
                            number_of_non_dominated_solutions = number_of_non_dominated_solutions - 1
                            del non_dominated_solutions[n]

                    n = n + 1

                if condition2 == 1:
                    non_dominated_solutions[number_of_non_dominated_solutions] = ordered_list[i]
                    number_of_non_dominated_solutions = number_of_non_dominated_solutions + 1

            sorted_non_dominated_solutions = sorted(non_dominated_solutions,
                                                    reverse=True)  # index sorting of non-dominated sol
            self.Pareto_Wheel_Output += "Sorted non-dominated solution index {} \n".format(
                sorted_non_dominated_solutions)
            self.Pareto_Wheel_Output += "Non-dominated solutions: {} \n".format(non_dominated_solutions)
            for i in range(number_of_non_dominated_solutions):
                assigned = assigned + 1
                # if fronts[sorted_non_dominated_solutions[i]] == 0:
                fronts[sorted_non_dominated_solutions[i]] = front
                for j in range(len(eval_temp)):  # sets the chosen goals to zero
                    # self.Pareto_Wheel_Output += "Setting to zero goals for %s "%(eval_temp[j]
                    # [sorted_non_dominated_solutions[i]]	)  + "\n"
                    eval_temp[j][sorted_non_dominated_solutions[i]] = -1000
            front = front + 1

        self.Pareto_Wheel_Output += "Calculated Pareto Frontiers {} \n\n".format(fronts)
        # print("Pareto Frontiers Calculation Completed")

        # ## check if front = 0 for error capture
        if min(fronts) == 0:
            # f = open('FrontERROR.txt','w') # redirect all print statements to file output
            # print ('Pareto front zero detected',file=f)
            # assign front 0 to be max(front)+1 so that it is worst sol
            newFront = max(fronts) + 1
            for i, f in enumerate(fronts):
                if f == 0:
                    # fronts[i]=max(fronts)+1
                    fronts[i] = newFront
        # self.Pareto_Wheel_Output += "After Zero Front Reassignment: Calculated Pareto Frontiers %s "%(fronts)
        # + "\n\n"
        # np.savetxt(path + os.sep +'ZEROFRONTOutput.txt', [self.Pareto_Wheel_Output], fmt='%s', newline='\n')
        # print Pareto and Roulette Wheenl message

        return fronts  # returns list of pareto fronts for the observations i.e. population


    # Tune fitness values by locating and using solution niches
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    # uses distance for solutions within a given pareto front to see if they are similar or diff
    # and assigns a new fitness score
    
    def tuneFitnessValue(self, fronts, sigma_share, evaluation_values, rep, population, individuals):
        # ###### evaluation values has last values as weighted average
        min_values = self.min_values
        max_values = self.max_values

        # print("Starting Fitness Tuning As Per Pareto Frontiers")

        for i in range(1, max(
                fronts) + 1):  # start from 1st pareto front; index of fronts is starting with 1, increment by 1
            self.Pareto_Wheel_Output += "Fitness Tuning As Per Pareto Front: {} \n".format(i)
            ind = [y for y, x in enumerate(fronts) if x == i]  # get individuals in pareto front
            self.Pareto_Wheel_Output += "Individuals in pareto front {} \n".format(ind)

            # Calculate max values per goal per pareto frontier
            max_significances = [-1000] * (len(evaluation_values) - 1)  # exclude weighted average
            for j in range(len(ind)):
                for goal in range(len(evaluation_values) - 1):
                    if evaluation_values[goal][ind[j]] >= max_significances[goal]:
                        max_significances[goal] = evaluation_values[goal][ind[j]]

            self.Pareto_Wheel_Output += "Max. Significance {} \n".format(max_significances)

            # compute distance based similarity for the solutions within a given front
            for j in range(len(ind)):  # iterate for individuals within a given front
                m = 0
                for k in range(len(ind)):
                    d = 0
                    # gene_count=0
                    for gene in range(len(individuals[0])):
                        d = d + ((individuals[ind[j]][gene] - individuals[ind[k]][gene]) / float(
                            max_values[gene] - min_values[gene])) ** 2
                    # gene_count= gene_count + 1
                    d = math.sqrt(d / (len(individuals[0])))

                    if d <= sigma_share:
                        m = m + (1 - ((d / float(sigma_share)) ** 2))
                    if m == 0:
                        m = 1
                self.Pareto_Wheel_Output += "Value of m  {} \n".format(m)
                for goal in range(len(evaluation_values) - 1):
                    evaluation_values[goal][ind[j]] = float(max_significances[goal]) / m

        # # recalculated weighted sum of goals using goal_significance
        # goals are in rows and samples in columns
        for i in range(len(evaluation_values[0])):  # compute weighted sum for each sample
            evaluation_values[-1][i] = 0  # reset weighted average i.e. last row to zero
            for j in range(len(evaluation_values) - 1):  # exclude last row i.e. weighted sum in iteration
                evaluation_values[-1][i] = evaluation_values[-1][i] + evaluation_values[j][i] * self.goal_significance[
                    j]
            evaluation_values[-1][i] = evaluation_values[-1][i] / float(len(evaluation_values) - 1)

        # # Update Sum of weighted avg goal and avg of weighhted avg goal for the generation
        # to be used for visualisation
        sum_ranked = 0
        for i in range(population):
            sum_ranked = sum_ranked + evaluation_values[-1][i]

        self.sum_ranked_eval_per_generation[rep] = sum_ranked
        self.average_ranked_eval_per_generation[rep] = sum_ranked / population

        # #### write tuned goals to csv for testing for 3 generations only
        # if rep <4:
        #   df = pd.DataFrame(evaluation_values)
        #   df.to_csv(path_or_buf = pathFeature + os.sep+ str(rep)+"GoalsTunedPostFront.csv", index=False, header=False)

        # print("Fitness Tuning As Per Pareto Frontiers Completed")

        return evaluation_values


    # roulette wheel based selection of ranked individuals
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    # evaluation values are of shape goals X samples
    
    def roulette_Wheel(self, evaluation_values_tuned, population, individuals, rep, bestSolIndex):

        min_values = self.min_values
        selected_individuals = [[0 for x in range(len(min_values))] for x in range(population)]
        indv_wheel = copy.deepcopy(
            individuals)  # included this to avoid any link between individuals and selected indv.
        # calculate the cumulative proportions i.e. selection probability based on weighted avg goal
        sum_prop = [0] * (population + 1)
        for i in range(1, population + 1):
            sum_prop[i] = sum_prop[i - 1] + evaluation_values_tuned[-1][i - 1] / float(
                self.sum_ranked_eval_per_generation[rep])

        self.Pareto_Wheel_Output += "Cumulative proportions based on weighted avg post tuning of Fitness " \
                                    "Value: {}\n".format(sum_prop)

        for i in range(1, population):  # get proportions from index 1 through pop size as 0 is for best ind.
            random_number = random.uniform(0, 1)  # generate probability randomly
            for j in range(0, population):  # select those indv. for which random no is within cumulative prop
                if sum_prop[j] <= random_number < sum_prop[j + 1]:
                    selected_individuals[i] = copy.deepcopy(indv_wheel[j])  # assign via deepcopy and not original

        # # assign best sol based on max of weighted sum of goals to index 0
        selected_individuals[0] = copy.deepcopy(indv_wheel[bestSolIndex])  # assign via deepcopy to avoid index editing

        # print("Roulette Wheel Selection Completed")

        return selected_individuals


    # function to apply two-point cross over
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos  
       
    def GACross_Over(self, population, cross_indv):

        # print("Doing Two Point Crossover")
        self.output_message += "Crossover population {}\n".format(population)

        temp_individuals = [[0 for x in range(len(self.min_values))] for x in range(population)]

        self.output_message += "Temp individual length {}\n".format(len(temp_individuals))

        cross_indv_copy = copy.deepcopy(cross_indv)  # take deep copy to avoid index 0 editing
        features = copy.deepcopy(len(cross_indv_copy[0]))  # take deep copy to avoid index 0 editing

        best_sol_before = sum(cross_indv[0])  # index 0 holds best sol
        self.output_message += "best sol sum before {}\n".format(sum(cross_indv[0]))

        for i in range(1, population - 1, 2):  # 1st is best sol so preserve it
            random_number = random.uniform(0, 1)  # generate prob randomly
            if random_number < self.twoCross_prob:
                self.output_message += "Doing 2-point crossover for sol: {}, {}\n".format(i, i + 1)
                cross_point1 = 0
                cross_point2 = 0
                while cross_point1 == cross_point2:
                    cross_point1 = math.ceil((features - 4) * random.uniform(0, 1))  # minus 4 to leave last positions
                    if cross_point1 < math.floor((2 * features - 1) / 3):
                        width = math.ceil(random.uniform(0, 1) * (math.floor(features - 1) / 3 - 2))
                        cross_point2 = cross_point1 + width
                    else:
                        width = math.ceil(random.uniform(0, 1)) * (
                                math.floor(features / 3 - 1) - 2 - (cross_point1 - math.floor(2 * features / 3)))
                        cross_point2 = cross_point1 + width
                if cross_point1 > cross_point2:
                    temp_cross_point = cross_point1
                    cross_point1 = cross_point2
                    cross_point2 = temp_cross_point
                width = int(width)
                cross_point1 = int(cross_point1)
                cross_point2 = int(cross_point2)
                for j in range(cross_point1, cross_point2 + 1):
                    temp_individuals[i][j] = copy.deepcopy(cross_indv_copy[i + 1][j])
                for j in range(cross_point1, cross_point2 + 1):
                    cross_indv_copy[i + 1][j] = copy.deepcopy(cross_indv_copy[i][j])
                for j in range(cross_point1, cross_point2 + 1):
                    cross_indv_copy[i][j] = copy.deepcopy(temp_individuals[i][j])

            elif self.twoCross_prob <= random_number < (self.twoCross_prob + self.arithCross_prob):
                # arithmetic cross over
                self.output_message += "Doing arithmetic crossover for sol: {}, {}\n".format(i, i + 1)
                alpha = random.uniform(0, 1)
                for j in range(0, features):
                    temp_individuals[i][j] = copy.deepcopy(
                        alpha * cross_indv_copy[i][j] + (1 - alpha) * cross_indv_copy[i + 1][j])
                    temp_individuals[i + 1][j] = copy.deepcopy(
                        (1 - alpha) * cross_indv_copy[i][j] + (alpha) * cross_indv_copy[i + 1][j])
                for k in range(0, features):
                    cross_indv_copy[i][k] = copy.deepcopy(temp_individuals[i][k])
                    cross_indv_copy[i + 1][k] = copy.deepcopy(temp_individuals[i + 1][k])

        best_sol_after = sum(cross_indv_copy[0])  # index 0 holds best sol; take after sum from copy version
        self.output_message += "best sol sum after {}\n".format(sum(cross_indv_copy[0]))

        # ## check if best solution is preserved
        if (best_sol_before - best_sol_after) != 0.0:
            with open('CrossOverERROR.txt', 'w') as fcross:  # redirect all print statements to file output
                self.output_message += "Cross over of best sol detected \n"
                # print('Cross over for best solution detected. Sol sum delta %s' % (best_sol_before - best_sol_after),
                #       file=fcross)

        # print("Crossover completed")
        return copy.deepcopy(cross_indv_copy)


    # function to apply adaptive mutation; Use deep copy for assignment to avoid index 0 editing
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    # script edited for adaptive mutation rate

    def adaptiveMutation(self, population, mutate_indv, rep):

        min_values = self.min_values
        max_values = self.max_values
        self.output_message += "Doing adaptive mutation \n"
        self.output_message += "population size {}\n".format(population)

        best_sol_before = sum(mutate_indv[0])  # index 0 holds best sol
        mutate_indv_copy = copy.deepcopy(mutate_indv)  # take deep copy so that index 0 is not chnaged

        # get the mutation probability and gaussian variance proportion based on average similarity
        # ref: Rapakoulia, T., Theofilatos, K., Kleftogiannis, D., Likothanasis, S., Tsakalidis, A., & Mavroudi, S.
        # (2014).
        # EnsembleGASVR: a novel ensemble method for classifying missense single nucleotide polymorphisms.
        # Bioinformatics, 30(16), 2324-2333.

        # compute average similarity between the members of the population and its best member i.e. at index 0
        a = np.array(mutate_indv_copy[:0:-1])  # take all elements from bottom to 1, exclude 0
        b = np.array([mutate_indv_copy[0]])  # best solution
        CDist = distance.cdist(a, b, 'chebyshev')
        AvgSimilarity = np.mean(CDist) / (np.max(CDist) - np.min(CDist))
        self.Pareto_Wheel_Output += "Avg. Similarity {} \n".format(AvgSimilarity)

        numerator = float(self.mutation_prob[1]) - (1 / self.population)  # 0.2-(1/Pop size)
        delta = float(rep) * numerator / float(self.generations)

        logging.debug(f"Average similarity in mutation: {AvgSimilarity}")

        if round(AvgSimilarity, 2) >= round(self.avgSimThr,
                                            2):  # high similarity, so increase the diversity with higher mutation
            mutationRate = float(self.mutation_prob[1]) + delta  # 0.2+ delta
            gaussVarProp = float(self.gaussMut_varPro[1]) + delta  # 0.5+ delta
        else:
            mutationRate = float(self.mutation_prob[1]) - delta  # 0.2 - delta
            gaussVarProp = float(self.gaussMut_varPro[1]) - delta  # 0.5 - delta
            if mutationRate < float(self.mutation_prob[0]):  # reset if lesser than lower bound
                mutationRate = float(self.mutation_prob[0])
            if gaussVarProp < float(self.gaussMut_varPro[0]):  # reset if lesser than lower bound
                gaussVarProp = float(self.gaussMut_varPro[0])

        self.Pareto_Wheel_Output += "Mutation Rate chosen {} \n".format(mutationRate)
        self.Pareto_Wheel_Output += "Gaussian Variance Proportion chosen {} \n".format(gaussVarProp)

        feature = copy.deepcopy(len(mutate_indv_copy[0]))
        # apply mutation operator
        for i in range(1, population):  # preserve the 1st sol as its best
            logging.debug(f"Mutating individual {i}")
            for j in range(0, feature):
                random_number = random.uniform(0, 1)
                if random_number < mutationRate:
                    # Gaussian distribution. mean zero and standard deviation = (max-min)*variance proportion
                    mutate_indv_copy[i][j] = copy.deepcopy(
                        mutate_indv_copy[i][j] + random.gauss(0, gaussVarProp * (max_values[j] - min_values[j])))
            # self.output_message += "sol mutated: %s"%i + "\n"

            # Correct values out of boundaries
            for j in range(0, len(min_values)):
                if mutate_indv_copy[i][j] < min_values[j]:
                    mutate_indv_copy[i][j] = copy.deepcopy(min_values[j])
                if mutate_indv_copy[i][j] > max_values[j]:
                    mutate_indv_copy[i][j] = copy.deepcopy(max_values[j])

        best_sol_after = sum(
            mutate_indv_copy[0])  # index 0 holds best sol; take sum of index 0 from copy to check if any change

        # print(best_sol_before - best_sol_after)
        # ## check if best solution is preserved
        if (best_sol_before - best_sol_after) != 0.0:
            logging.error("Mutation of the best solution occurred!")
            with open('MutationERROR.txt', 'w') as fmutate:  # redirect all print statements to file output
                self.output_message += "Mutation of best sol in Gen {} \n".format(rep)
                # print('Mutation of best solution detected. Sol sum delta %s' % (best_sol_before - best_sol_after),
                #       file=fmutate)

        return copy.deepcopy(mutate_indv_copy)

    
    def evaluate_individuals(self, individual, goal_values, desired_outputs, post_evaluate, generation_idx):
        
        # Check predictions_storage before adding anything
        #print(f"[Debug] Before appending: predictions_storage has {len(self.predictions_storage)} generations.")
        #print(f"[Debug] Current generation index: {generation_idx}. Content of predictions_storage[generation_idx]: {self.predictions_storage[generation_idx]}")

        from rpy2.robjects import numpy2ri
        from sklearn.metrics import mean_squared_error
        numpy2ri.activate()

        mse_values = []
        predictions = []
        robjects.r('library(glmnet)')
        r_predict = r['predict']

        for i, model in enumerate(self.models):
            model_name = list(self.scaling_params.keys())[i]  # Get model-specific key name
            try:
                scaling_info = self.scaling_params[model_name]
                means = np.array(list(scaling_info['means'].values())[:-1])
                std_devs = np.array(list(scaling_info['std_devs'].values())[:-1])

                # Scale individual input using specific model scaling parameters
                scaled_individual = scale_input_data(individual, means, std_devs)
                
                # Debug: Log scaling and input data
                print(f"[evaluate_individuals] Model: {model_name}, Original input: {individual}")
                print(f"[evaluate_individuals] Model: {model_name}, Scaled input: {scaled_individual}")

                # Convert scaled individual to R matrix
                individual_r = robjects.r['as.matrix'](robjects.DataFrame({
                    'param1': [scaled_individual[0]],
                    'param2': [scaled_individual[1]],
                    'param3': [scaled_individual[2]],
                    'param4': [scaled_individual[3]]
                }))

                # Prediction
                if 'cv.glmnet' in robjects.r['class'](model):
                    lambda_min = model.rx2('lambda.min')[0]
                    y_pred_r = r_predict(model, newx=individual_r, s=lambda_min, type="response")
                else:
                    y_pred_r = r_predict(model, individual_r)
  
                # Convert prediction to float
                y_pred = float(y_pred_r[0])
                predictions.append(y_pred)

                # Debug
                print(f"[evaluate_individuals] Model: {model_name}, Raw prediction: {y_pred}")

            except Exception as e:
                print(f"Error evaluating individual with model {model_name}: {e}")
                mse_values.append(float('inf'))  # High error if evaluation fails

        bounded_predictions = bound_output(predictions)
                
        for i, pred in enumerate(bounded_predictions):
            # Calculate MSE
            desired_value = desired_outputs[i]
            mse = mean_squared_error([desired_value], [pred])
            print(f"MSE for index {i}: {mse}")
            #mse = mean_squared_error([desired_value], [y_pred])
            mse_values.append(mse)

        # Store predictions in predictions_storage for later use
        self.predictions_storage[generation_idx].append(bounded_predictions)

        # Debug to verify predictions_storage contents
        #print(f"[Debug] After appending: predictions_storage[{generation_idx}] now has {len(self.predictions_storage[generation_idx])} items.")
        #print(f"[Debug] Content of predictions_storage[{generation_idx}]: {self.predictions_storage[generation_idx]}")

        # Calculate fitness values
        fitness_values = [1 / (1 + mse) for mse in mse_values]

        # Set goal_values based on fitness
        goal_values = [0.0001] * 6
        for idx in range(min(5, len(fitness_values))):
            goal_values[idx] = fitness_values[idx]

        # Final goal value with weighted significance
        goal_values[5] = sum(goal_values[j] * self.goal_significance[j] for j in range(5)) / sum(self.goal_significance[:5])

        print(f"Debug: Final goal values for individual: {goal_values}")
        
        return goal_values
    
    def bioreactor_optimization(self, desired_outputs):
        #logging.info("Starting bioreactor optimization")
        # Start timing the optimization process
        start = time.perf_counter()
        self.output_message += "Parameters selected Gen: {} , Pop: {} \n".format(self.generations, self.population)

        # Initialize population of individual solutions with 4 float values (inputs to the models)
        individuals = self.initialize()  # Generate initial population
        self.output_message += "Initial population formulated. Sample individuals length: {}, {}\n".format(
            len(individuals[0]), len(individuals[4]))
        self.output_message += "Sample individual: {}\n".format(individuals[4])

        # Define generation and population variables
        generations = self.getGenerations()
        population = self.getPopulation()

        # Initialize variables for storing results
        sigma_share = 0.5 / (float(len(individuals[0])) ** 0.1)  # Sigma for distance threshold
        premature_termination = 0
        front1_sol_perGen = []
        front1_feat_perGen = []
        final_selected_best = []
        best_index = 0
        front1_index = 0

        # Modify solHeader to include both input parameters and predicted outputs
        solHeader = ['Sphere diameter', 'Sphere distance', 'Delaunay mesh dimension', 'Delaunay point spacing']

        # Initialize the attribute to store predicted outputs per generation
        self.predicted_outputs_per_generation = []  # This will store the outputs (predictions) for each individual

        # Evolutionary process (GA optimization loop)
        for rep in range(generations):  # Iterate through generations
            goal_values = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

            start_gen_time = time.perf_counter()
            self.output_message += "\nGeneration : {}\n".format(rep)
            logging.debug("Generation : {}".format(rep))
            post_evaluate = False

            # Evaluate individuals in parallel
            goal_values_old = copy.deepcopy(goal_values)
            predicted_outputs_for_generation = []
            goal_values = Parallel(n_jobs=2)(
                delayed(self.evaluate_individuals)(ind, copy.deepcopy(goal_values),
                                                copy.deepcopy(desired_outputs), copy.deepcopy(post_evaluate), rep)
                for ind in individuals
            )            
            
            # Add this debug print right after evaluating individuals
            # print(f"Debug: Fitness values for Generation {rep}: {goal_values}")
            
            #goal_values = [flatten_list(gv) for gv in goal_values]
            for individual in individuals:
                predicted_output = self.evaluate_individuals(individual, goal_values, desired_outputs, post_evaluate=True, generation_idx=rep)         
                predicted_outputs_for_generation.append(predicted_output[:5])  # First 5 values are the predictions
                print(f"[Optimization - Raw Predictions] Model predictions before storage: {predicted_output[:5]}")
            print(f"[Optimization] Predicted outputs for generation {rep}: {predicted_outputs_for_generation}")

            self.predicted_outputs_per_generation.append(predicted_outputs_for_generation)
                        
            #predicted_outputs_for_generation = self.predicted_outputs_per_generation[rep]  # Use already stored predictions

            
            # Add detailed logging of the time taken per generation
            end_gen_time = time.perf_counter()
            #logging.debug("Generation %d completed in %.2f seconds", rep, end_gen_time - start_gen_time)

            average_performance = 0  # average performance for convergence. initialise to zero
            convergence = 0.0
            # [i for i in goal_values if i] # this will create a new list which does not have any empty item,
            # else zip returns null if any value is null
            # avgAll = [float(sum(col))/len(col) for col in zip(*goal_values)] # avg of all goals
            avgAll = [float(sum(col)) / len(col) for col in zip(*[i for i in goal_values if i])]  # avg of all goals
            # print("Average of all goals %s" % avgAll)
            # print("Number of all goals: %d" % len(avgAll))

            average_performance = copy.deepcopy(avgAll[5])  # index 9 has weighted avg
            self.average_eval_per_generation[rep] = copy.deepcopy(average_performance)
            ## best_performance = list(
            ##    map(max, zip(*[i for i in goal_values if i]))) 
            best_performance = list(map(max, zip(*[i for i in goal_values if isinstance(i, list)]))) # includes max of classification model as well
            # print()
            best_avg_perf = copy.deepcopy(best_performance[5])  # max weighted average; index 9 has weighted avg
            bestSolIndex = 0
            for idx in range(len(goal_values)):
                # if round(goal_values[idx][9],4) == round(best_avg_perf,4): # float comparison, so using round
                if goal_values[idx][5] < -1:
                    print(goal_values[idx])
                if goal_values[idx][5] == best_avg_perf:  # float comparison
                    bestSolIndex = idx
                    self.output_message += "best sol index found\n"
                    break
            
            self.output_message += "best sol index {}\nbest avg performance {}\n avg of avg goal {}\n".format(
                bestSolIndex, best_avg_perf, average_performance)

            convergence = math.fabs(best_avg_perf - average_performance)
            # print("\nConvergence value %s" % convergence)
            # Convergence criterion is checked in order to stop the evolution if the population is deemd us converged
            if convergence < self.convergeThreshold * average_performance:
                premature_termination = 1

            # Estimate pareto fronts for the solutions of this generation based on all goals except weighted average
            stopIndx = (len(goal_values[0]) - 2)  # goals to slice for pareto front computation, excludes weighted
                      
            # average
            # print("\nPareto stopIndx is: %s" % stopIndx)
            self.Pareto_Wheel_Output += "Generation : {}\n".format(rep + 1)
            fronts = self.getParetoFront(len(copy.deepcopy(individuals)), goal_values, stopIndx)

            # #### save front1 solutions, features are intersect of GA individuals and filter method
            post_evaluate = True  # when true this flag avoid classifier fitting within evaluate
            for_front1 = copy.deepcopy(individuals)
            for i, ind in enumerate(for_front1[:]):  # iterate on solutions to pic sol with front = 1
                if fronts[i] == 1:
                    # print("%d" % i)
                    front1_sol_perGen.append([])
                    front1_sol_perGen[front1_index].append(rep)  # add generation number
                    front1_sol_perGen[front1_index].extend(ind)
                    front1_feat_perGen.append([])
                    front1_feat_perGen[front1_index].append(rep)  # add generation number
                    front1_feat_perGen[front1_index].extend(
                        self.evaluate_individuals(ind, copy.deepcopy(goal_values[i]),
                                                copy.deepcopy(desired_outputs), post_evaluate, generation_idx=rep)
                    )
                    front1_index = front1_index + 1

            # get max per goal; and best sol
            self.max_eval_per_generation[rep] = list(map(max, zip(*[i for i in goal_values if i])))[5]
            self.output_message += "best performance :" + "\n"
            self.output_message += str(self.max_eval_per_generation[rep])
            self.output_message += "\n"

            self.max_sol_per_generation[rep] = copy.deepcopy(individuals[bestSolIndex])  # use index of bestSolIndex,
            # there can be more than one index with max weighted average
            self.output_message += "best solution :\n"
            self.output_message += str(self.max_sol_per_generation[rep])
            self.output_message += "\n"
            self.output_message += str(individuals[bestSolIndex])
            self.output_message += "\n"

            # get final features for best sol per gen, features are intersect of GA individuals and filter method
            post_evaluate = True  # when true this flag avoid classifier fitting within evaluate
            final_selected_best.append([])
            # print("\nFinal features for best solution per generation")
            final_selected_best[best_index].extend(
                self.evaluate_individuals(copy.deepcopy(individuals[bestSolIndex]), 
                                        copy.deepcopy(goal_values[bestSolIndex]),
                                        copy.deepcopy(desired_outputs), 
                                        post_evaluate, generation_idx=rep)  # Ensure `generation_idx=rep` is inside this call
            )
            best_index = best_index + 1
            # Check if its last generation, do not apply pareto, roulette wheel, cross over and mutation
            if rep == (generations - 1) or premature_termination == 1:

                if premature_termination == 0:
                    # print("This was last generation without convergence")
                    self.output_message += "No Convergence\n"
                else:
                    # print("Solution converged")
                    self.output_message += "Convergence at Generation : {}\n".format(rep + 1)

                np.savetxt(self.pathFeature + os.sep + "goals_Final.csv", goal_values, delimiter=",",
                           header=self.goal_header)

                # get final features for final sol, features are intersect of GA individuals and filter method
                post_evaluate = True
                post_evaluate2 = False  # when true this flag avoid classifier fitting within evaluate
                finalFeat_selected = []
                for_final = copy.deepcopy(individuals)
                # ##KOSTAS
                # if len(MULTI_LABELS) == 1:
                # print("multi_label_test")
                for i, ind in enumerate(for_final[:]):
                    finalFeat_selected.append([])
                    finalFeat_selected[i].extend(
                        self.evaluate_individuals(ind, copy.deepcopy(goal_values[i]),
                              copy.deepcopy(desired_outputs), post_evaluate, generation_idx=rep))


                # write pareto front1 of the final solution; Sol sets in Pareto front 1 to be used in prediction
                # model
                front1FinalIndv = []
                front1FinalGoal = []
                front1FinalFeature = []
                index = 0
                front1FinalPredictions = []
                # for i in range(population): # iterate on solutions to pic sol with front = 1
                for_final_front1 = copy.deepcopy(individuals)
                for i, ind in enumerate(for_final_front1[:]):
                    if fronts[i] == 1 and max(finalFeat_selected[i]) > 0:
                        front1FinalIndv.append([])
                        front1FinalIndv[index].extend(ind)
                        front1FinalGoal.append([])
                        front1FinalGoal[index].extend(goal_values[i])
                        front1FinalFeature.append([])
                        front1FinalFeature[index].extend(finalFeat_selected[i])
                        index = index + 1

                try:
                    df = pd.DataFrame(front1FinalIndv)
                    df.to_csv(path_or_buf=self.pathFeature + os.sep + "FinalSolFront1.csv", index=False,
                              header=solHeader)
                except ValueError as e:
                    df = pd.DataFrame(front1FinalIndv)
                    df.to_csv(path_or_buf=self.pathFeature + os.sep + "FinalSolFront1.csv", index=False, header=False)

                np.savetxt(self.pathFeature + os.sep + "goals_FinalFront1.csv", front1FinalGoal, delimiter=",",
                           header=self.goal_header)
        
                 # write original sol

            # compute fitness value guided by grouping of sol in each pareto fronts and recompute weighted average
            # so retain weighted average goal
            stopIndxAvg = (len(goal_values[0]) - 1)  # last column is classification model so exclude that
            # print("stopIndxAvg %s" % stopIndxAvg)

            evaluation_values = copy.deepcopy(np.asarray(goal_values)[:, :stopIndxAvg:1])
            evaluation_values = evaluation_values.T
            evaluation_values_tuned = self.tuneFitnessValue(fronts, sigma_share, evaluation_values, rep,
                                                            len(individuals), individuals)

            # roulett wheel selection on tuned evaluation values by pareto front
            wheel_indv = copy.deepcopy(individuals)  # take a deep copy so that best sol is not altered
            selected_individuals = self.roulette_Wheel(evaluation_values_tuned, len(individuals), wheel_indv, rep,
                                                       bestSolIndex)

            # do cross-over
            cross_indv = copy.deepcopy(selected_individuals)  # take deep copy so that best ind is not altered; index 0 solution
            cross_individuals = self.GACross_Over(len(individuals), cross_indv)

            # do mutation
            mutate_indv = copy.deepcopy(cross_individuals)  # take deep copy so that best ind is not altered; index 0 solution
            mutated_individuals = self.adaptiveMutation(len(individuals), mutate_indv, rep)
         
            individuals = copy.deepcopy(mutated_individuals)

        # last solution is the final solution
        try:
            dfFinalSol = pd.DataFrame(individuals)
            dfFinalSol.to_csv(path_or_buf=self.pathFeature + os.sep + "FinalSolutionsAll.csv", index=False,
                              header=solHeader)
        except ValueError as e:
            dfFinalSol = pd.DataFrame(individuals)
            dfFinalSol.to_csv(path_or_buf=self.pathFeature + os.sep + "FinalSolutionsAll.csv", index=False,
                              header=False)

        # solutions provided in FinalSolFront1.csv
        self.process_final_solutions(desired_outputs)
        #self.process_optimal_solutions(desired_outputs)

        self.output_message += "\nTime (in minutes) taken for Optimizing Bioreactors Design: %s" % (
                (time.perf_counter() - start) / 60.00) + "\n"

        # ### write avg and best performance goal values before pareto and avg after pareto
        dfAvg = pd.DataFrame(self.average_eval_per_generation)
        dfAvg.to_csv(path_or_buf=self.pathFeature + os.sep + "AvgOfAvgGoalsPerGen.csv", index=False, header=False)

        dfAvgFront = pd.DataFrame(self.average_ranked_eval_per_generation)
        dfAvgFront.to_csv(path_or_buf=self.pathFeature + os.sep + "AvgOfAvgGoalPerGenPostFront.csv", index=False,
                          header=False)
     
        try:
            df = pd.DataFrame(front1_sol_perGen)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "Front1SolPerGen.csv", index=False,
                      header=solHeader)
        except ValueError as e:
            df = pd.DataFrame(front1_sol_perGen)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "Front1SolPerGen.csv", index=False, header=False)

        try:
            dfMax = pd.DataFrame(self.max_eval_per_generation)
            dfMax.to_csv(path_or_buf=self.pathFeature + os.sep + "BestPerformancePerGen.csv", index=False,
                         header=False)
        except TypeError as e:
            logging.exception("typeError: object of type int has no len() occurred for best sol generation :")
            max_eval = [i for i in self.max_eval_per_generation if i]
            if len(max_eval) > 0:
                logging.info("Writing best goals by dropping null")
                dfMax = pd.DataFrame(max_eval)
                dfMax.to_csv(path_or_buf=self.pathFeature + os.sep + "BestPerformancePerGen.csv", index=False,
                             header=False)
        self.process_optimization_results()
        self.generate_plots()
        return front1FinalIndv, front1FinalFeature, final_selected_best[-1]

    
    def calculate_predictions(self, individual):
        predicted_values = []
        r_predict = r['predict']

        for i, (model, model_name) in enumerate(zip(self.models, self.scaling_params.keys())):
            scaling_info = self.scaling_params[model_name]
            means = np.array(list(scaling_info['means'].values())[:-1])
            std_devs = np.array(list(scaling_info['std_devs'].values())[:-1])

            scaled_individual = scale_input_data(individual, means, std_devs)
            
            try:
                individual_r = robjects.r['as.matrix'](robjects.DataFrame({
                    'param1': [scaled_individual[0]],
                    'param2': [scaled_individual[1]],
                    'param3': [scaled_individual[2]],
                    'param4': [scaled_individual[3]]
                }))
                
                if 'cv.glmnet' in robjects.r['class'](model):
                    lambda_min = model.rx2('lambda.min')[0]
                    y_pred_r = r_predict(model, newx=individual_r, s=lambda_min, type="response")
                else:
                    y_pred_r = r_predict(model, individual_r)

                # Convert prediction to float and add to predictions list
                y_pred = float(y_pred_r[0])
                predicted_values.append(y_pred)

                #predicted_values.append(y_pred)

                # Debug: Log predictions
                print(f"[calculate_predictions] Model: {model_name}, Raw prediction: {y_pred}")

            except Exception as e:
                print(f"Error calculating prediction for model {model_name}: {e}")
                predicted_values.append(None)  # Append None if an error occurs

        #  apply bounding
        bounded_predictions = bound_output(predicted_values)
        predicted_values=bounded_predictions

        return predicted_values  
    

    def process_final_solutions(self, desired_outputs):
        final_solutions_path = os.path.join(self.pathFeature, "FinalSolFront1.csv")
        final_solutions = pd.read_csv(final_solutions_path)

        results = []
        headers = [
            "Prediction", "Desired Value", "Absolute Error",
            "Prediction", "Desired Value", "Absolute Error",
            "Prediction", "Desired Value", "Absolute Error",
            "Prediction", "Desired Value", "Absolute Error",
            "Prediction", "Desired Value", "Absolute Error"
        ]

        for _, solution in final_solutions.iterrows():
            individual = solution.values.tolist()
            predictions = self.calculate_predictions(individual)
            errors = [abs(d - float(p)) for d, p in zip(desired_outputs, predictions)]

            row = []
            for j in range(5):  
                row.append(f"{float(predictions[j]):.4f}")   
                row.append(f"{desired_outputs[j]}")     
                row.append(f"{errors[j]:.4f}")              

            results.append(row)

        final_results_df = pd.DataFrame(results, columns=headers)
        output_path = self.pathFeature + os.sep + "final_predictions_vs_desired.csv"
        final_results_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONE)
    
    '''
    def process_optimal_solutions(self, desired_outputs):
        # Load FinalSolFront1.csv containing the final solutions
        final_solutions_path = os.path.join(self.pathFeature, "FinalSolFront1.csv")
        final_solutions = pd.read_csv(final_solutions_path)

        # Prepare lists to hold each model's predictions, desired values, and errors
        results = []
        
        # Define headers for the CSV
        headers = [
            "Solution Parameters",  # Will hold the actual solution values
            "Area/Volume Prediction", "Desired Area/Volume", "Area/Volume Absolute Error",
            "Bone Porosity Prediction", "Desired Bone Porosity", "Bone Porosity Absolute Error",
            "Connectivity Density Prediction", "Desired Connectivity Density", "Connectivity Density Absolute Error",
            "Trabecular Thickness Prediction", "Desired Trabecular Thickness", "Trabecular Thickness Absolute Error",
            "Trabecular Spacing Prediction", "Desired Trabecular Spacing", "Trabecular Spacing Absolute Error"
        ]

        # Use predictions from the last generation in predicted_outputs_per_generation
        final_generation_predictions = self.predictions_storage[-1]

        # Loop over each final solution and compute predictions and errors
        for index, solution in final_solutions.iterrows():
            # Extract the individual solution parameters as a list and convert them to a string
            individual = solution.values.tolist()
            solution_str = ', '.join(map(str, individual))  # Convert the solution to a string format

            # Fetch predictions from the stored optimization results
            # Assuming the predictions are stored generation by generation
            # We assume the final generation predictions are what we want
            # You may need to adjust the index to match the correct generation
            #final_generation_predictions = self.predicted_outputs_per_generation[-1]
            predictions = final_generation_predictions[index]

            # Debug: Verify predictions
            print(f"[process_optimal_solutions] Raw predictions for solution {index}: {predictions}")

            #bounded_predictions = bound_output(predictions)
            #print(f"[process_optimal_solutions] Bounded predictions for solution {index}: {bounded_predictions}")
            
#####
            # Inside the loop for each solution
            recalculated_predictions = self.calculate_predictions(individual)
            print(f"[Debug] Saved: {predictions}, Recalculated: {recalculated_predictions}")
#####

            # Calculate the error (absolute difference) between predictions and desired values
            #errors = [abs(d - p) for d, p in zip(desired_outputs, bounded_predictions)]
            errors = [abs(d - p) for d, p in zip(desired_outputs, predictions)]
            # Prepare a row with the solution string and each model's prediction, desired value, and error
            row = [solution_str]  # The actual solution values as a string
            for j in range(5):  # Assuming 5 models (Area/Volume, Bone Porosity, etc.)
                #row.append(bounded_predictions[j])
                row.append(predictions[j])   # Bounded prediction
                row.append(desired_outputs[j])      # Desired value
                row.append(errors[j])               # Error
            results.append(row) 

        # Create a DataFrame to store the results
        final_results_df = pd.DataFrame(results, columns=headers)

        # Save the final results to a CSV
        output_path = self.pathFeature + os.sep + "optimal_final_predictions_vs_desired.csv"
        final_results_df.to_csv(output_path, index=False)

        print(f"Final predictions and errors saved to {output_path}")
    '''

    def process_optimization_results(self):
        """ Process the produced files to keep only the unique solutions. """

        # Load necessary CSV files
        sol_front_path = os.path.join(self.pathFeature, "FinalSolFront1.csv")
        goals_front_path = os.path.join(self.pathFeature, "goals_FinalFront1.csv")
        pred_desired_path = os.path.join(self.pathFeature, "final_predictions_vs_desired.csv")

        # Paths for output files
        sol_front_output = os.path.join(self.pathFeature, "unique_solutions.csv")
        goals_output = os.path.join(self.pathFeature, "unique_goals.csv")
        pred_desired_output = os.path.join(self.pathFeature, "predictions_vs_desired.csv")

        # Process the 'FinalSolFront1.csv' file to ensure uniqueness
        df_sol = pd.read_csv(sol_front_path)
        df_pred = pd.read_csv(pred_desired_path)

        unique_solutions = df_sol.drop_duplicates()
        unique_pred = df_pred.drop_duplicates()

        unique_solutions.to_csv(sol_front_output, index=False)
        unique_pred.to_csv(pred_desired_output, index=False)

        # Process 'goals_FinalFront1.csv' and apply transformation
        df_goals = pd.read_csv(goals_front_path).drop_duplicates()
        df_transformed_goals = (1 / df_goals) - 1

        # Ensure headers are set correctly
        new_headers = ['Area/Volume MSE', 'Bone Porosity MSE', 'Connectivity density MSE', 'Trabecular thickness MSE', 'Trabecular spacing MSE', 'Average']
        df_transformed_goals.columns = new_headers
        df_transformed_goals.to_csv(goals_output, index=False)

        # Now, add the 'Average' column from the transformed goals to predictions_vs_desired.csv
        if 'Average' in df_transformed_goals.columns:
            average_absolute_error = df_transformed_goals['Average'].apply(abs).reset_index(drop=True)
            
            # Add the "Average Absolute Error" column to predictions without suffixes
            combined_df = pd.concat([unique_pred.reset_index(drop=True), average_absolute_error], axis=1)
            
            # Set columns explicitly to prevent suffixes
            combined_headers = [
                "Prediction", "Desired Value", "Absolute Error",
                "Prediction", "Desired Value", "Absolute Error",
                "Prediction", "Desired Value", "Absolute Error",
                "Prediction", "Desired Value", "Absolute Error",
                "Prediction", "Desired Value", "Absolute Error",
                "Average Absolute Error"
            ]
            
            combined_df.columns = combined_headers

            # Save the final DataFrame to the output path
            combined_df.to_csv(pred_desired_output, index=False, quoting=csv.QUOTE_NONE)
        else:
            print("Error: 'Average' column not found in transformed goals.")


    def generate_plots(self):
        """ Generate the plots for the average and the best performance. """

        # Define paths to the files to be plotted
        best_performance_path = os.path.join(self.pathFeature, "BestPerformancePerGen.csv")
        avg_goals_path = os.path.join(self.pathFeature, "AvgOfAvgGoalsPerGen.csv")

        # Define paths for the output images
        best_performance_plot_path = os.path.join(self.pathFeature, "BestPerformancePerGen_plot.png")
        avg_goals_plot_path = os.path.join(self.pathFeature, "AvgOfAvgGoalsPerGen_plot.png")

        # Load data and generate plots
        try:
            # Plot for BestPerformancePerGen.csv
            best_performance_data = pd.read_csv(best_performance_path, header=None)
            plt.figure()
            plt.plot(best_performance_data, marker='o', linestyle='-', color='b')
            plt.title("Best Performance per Generation")
            plt.xlabel("Generation")
            plt.ylabel("Performance")
            plt.grid(True)
            plt.savefig(best_performance_plot_path)
            plt.close()
            print("BestPerformancePerGen plot saved to:", best_performance_plot_path)

            # Plot for AvgOfAvgGoalsPerGen.csv
            avg_goals_data = pd.read_csv(avg_goals_path, header=None)
            plt.figure()
            plt.plot(avg_goals_data, marker='o', linestyle='-', color='g')
            plt.title("Average of Goals per Generation")
            plt.xlabel("Generation")
            plt.ylabel("Average Goal Value")
            plt.grid(True)
            plt.savefig(avg_goals_plot_path)
            plt.close()
            print("AvgOfAvgGoalsPerGen plot saved to:", avg_goals_plot_path)

        except Exception as e:
            print(f"Error generating plots: {e}")

## Functions used for validating what user gives as inputs every time

def parse_float(value):
    try:
        return float(value.replace(',', '.'))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'")

def validate_desired_outputs(desired_outputs):
    """ Validate desired outputs to ensure they are within allowed ranges. """
    min_values = [4.5, 39.1, 0.3, 0.02, 0.09]
    max_values = [28.2, 100, 22.9, 0.2, 0.93]
    names = ["Area/Volume", "Bone Porosity", "Connectivity Density", "Trabecular Thickness", "Trabecular Spacing"]

    for i, value in enumerate(desired_outputs):
        if not (min_values[i] <= value <= max_values[i]):
            raise argparse.ArgumentTypeError(
                f"Invalid input for {names[i]}: {value}. Must be between {min_values[i]} and {max_values[i]}."
            )

def validate_input_params(input_params):
    """ Validate sphere-related values to ensure they are within allowed ranges. """
    min_values = [0.1, 0.1, 0.12, 0.1]
    max_values = [1.0, 1.0, 0.9, 0.9]
    names = ["Sphere Diameter", "Sphere Distance", "Delaunay Mesh Dimension", "Delaunay Spacing"]

    for i, value in enumerate(input_params):
        if not (min_values[i] <= value <= max_values[i]):
            raise argparse.ArgumentTypeError(
                f"Invalid input for {names[i]}: {value}. Must be between {min_values[i]} and {max_values[i]}."
            )

# inputs

def readArguments(argv):
    parser = argparse.ArgumentParser()

    # Arguments common to both modes (such as output_folder if needed)
    parser.add_argument('--output_folder', required=True, help='Folder to save the output files.')
    parser.add_argument('--model_dir', required=True, help='Directory containing the pre-trained R models.')
    parser.add_argument('--scaling_dir', required=True, help='Directory containing scaling parameter files.')
 
    # Arguments specific to optimization
    parser.add_argument('--min_values', nargs=4, type=float, metavar=('min_diameter', 'min_distance', 'min_mesh', 'min_spacing'),
                        default=[0.1, 0.1, 0.12, 0.1],
                        help='Min values for the 4 inputs, default is [0.1, 0.1, 0.12, 0.1]')

    parser.add_argument('--max_values', nargs=4, type=float, metavar=('max_diameter', 'max_distance', 'max_mesh', 'max_spacing'),
                        default=[1.0, 1.0, 0.9, 0.9],
                        help='Max values for the 4 inputs, default is [1.0, 1.0, 0.9, 0.9]')

    parser.add_argument('-pp', '--population', nargs='?', type=int, default=50, help='Population size, default is 50')
    parser.add_argument('-g', '--generations', nargs='?', type=int, default=300, help='Count of max generations, default is 200')
    parser.add_argument('-f', '--folds', nargs='?', type=int, default=5, help='Count of folds for cross-validation, default is 5')
    parser.add_argument('-fv', '--goal_values', nargs=6, type=float, default=[0.0001]*6,
                        help='Fitness values for each solution after evaluation. Default values are 0.0001 for 11 goal values')
    
    #parser.add_argument('--desired_outputs', nargs=5, type=float, help='Desired output values to optimize towards (e.g., area_volume, bone_porosity, etc.).')
    parser.add_argument('--desired_outputs', nargs=5, type=parse_float, help='Desired output values to optimize towards (e.g., area_volume, bone_porosity, etc.).')
    parser.add_argument('-gs', '--goal_significance', nargs=5, type=parse_float, default=[1.0] * 5, help='Goal significance values for the 5 fitness functions (default is [1.0, 1.0, 1.0, 1.0, 1.0])')

    # Arguments specific to prediction
    parser.add_argument('--sphere_diameter', type=parse_float, help="Sphere diameter input")
    parser.add_argument('--sphere_distance', type=parse_float, help="Sphere distance input")
    parser.add_argument('--delaunay_mesh', type=parse_float, help="Delaunay mesh dimension input")
    parser.add_argument('--delaunay_spacing', type=parse_float, help="Delaunay point spacing input")

    parser.add_argument('--jobid', type=int, help='Job ID passed from backend')

    args = parser.parse_args()

    # Determine whether it's prediction or optimization based on arguments provided

    if args.sphere_diameter and args.sphere_distance and args.delaunay_mesh and args.delaunay_spacing:
        args.mode = 'predict' 
        validate_input_params([args.sphere_diameter, args.sphere_distance, args.delaunay_mesh, args.delaunay_spacing])

    elif args.desired_outputs and args.output_folder:
        args.mode = 'optimize' 
        validate_desired_outputs(args.desired_outputs)
    else:
        raise ValueError("Invalid arguments: either provide prediction inputs or optimization settings.")

    Config.update_paths(args.model_dir, args.scaling_dir)

    return args


def main(varDict, user='unknown', jobid=0, pid=0):
    """
    Main function that handles both optimization and prediction.
    """

    if varDict['mode'] == 'optimize':
        # Run optimization process
        min_values = varDict['min_values']
        max_values = varDict['max_values']
        goal_values = [0.0001 for _ in range(6)]  # Initialize default goal values
        
        # Get the output folder path from varDict
        path = varDict['output_folder']  # get cwd and append the file path to it
        output_folder = os.path.join(path, 'Output_Optimization')
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        log_file_path = os.path.join(output_folder, "osteonet_LOG.log")
        logging.basicConfig(
            filename=log_file_path,
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        logging.info("Logging configured. Log file created at %s", log_file_path)  

        # Get user-defined desired outputs and significance values
        desired_outputs = varDict['desired_outputs']
        goal_significance = varDict['goal_significance']

        # Initialize optimization process
        optimization = OptimizationProcess(
            min_values=min_values,
            max_values=max_values,
            population=varDict['population'],
            generations=varDict['generations'],
            goal_values=goal_values,
            output_folder=output_folder
        )
        
        optimization.setGoalSignificancesByUserList(varDict['goal_significance'])
        optimization.bioreactor_optimization(varDict['desired_outputs'])

        # Run optimization and handle exceptions
        try:
            front1FinalIndv, front1FinalFeature, best_solution = optimization.bioreactor_optimization(desired_outputs)
            # Init configuration parser
            conn, config = config_connect_db()
            message = {"output": path}
            # Everything successful
            set_job_completed(jobid, conn, message, user, pid)
           
            sys.exit("Optimization process has finished successfully")
        except Exception as e:
            # Init configuration parser
            conn, config = config_connect_db()
            set_job_error(jobid, conn, user, result={'error': result[1]}, pid=0)           
            sys.exit("Optimization process has finished unsuccessfully")

            print(f"Error during optimization: {e}")
            return False

        return True

    elif varDict['mode'] == 'predict':
        # Run the prediction process
        sphere_diameter = varDict['sphere_diameter']
        sphere_distance = varDict['sphere_distance']
        delaunay_mesh = varDict['delaunay_mesh']
        delaunay_spacing = varDict['delaunay_spacing']

        # Initialize prediction class
        prediction_model = BioreactorPrediction()

        # Get predictions
        predictions = prediction_model.predict(
            sphere_diameter=sphere_diameter,
            sphere_distance=sphere_distance,
            delaunay_mesh=delaunay_mesh,
            delaunay_spacing=delaunay_spacing
        )

        # Prepare to save the predictions in a CSV file
        output_folder = args.output_folder

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            os.chmod(output_folder, 0o777)
        
        # Define the CSV file path
        predictions_path = os.path.join(output_folder, "bioreactor_predictions.csv")

        # Prepare the data for saving
        model_names = ['Area/Volume', 'Bone porosity', 'Connectivity density', 'Trabecular thickness', 'Trabecular spacing']
        data = {'Model': model_names, 'Prediction': predictions}
        df = pd.DataFrame(data)

        # Save the DataFrame to CSV
        df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
        return True

    else:
        print("Error: Invalid mode detected. Ensure correct arguments are provided.")
        return False

# Find diagnostics.ini file to config the program and connect to Database
def config_connect_db():
    """
    Get configurations from ini file, and connect to the database.
    :return: conn: connection object of db, config: configuration dictionary
    """
    config = configparser.ConfigParser()

    # Check if the file is loaded successfully
    config_path = '/media/scripts_osteonet/diagnostics.ini'
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit("Configuration file not found.")
    
    # Try to read the config file
    config.read(config_path)

    # Check if the section exists in the config
    if 'diagnostics.db' not in config:
        print("Section 'diagnostics.db' not found in the config file.")
        sys.exit("Invalid config file format.")

    # Connect to the database
    try:
        conn = psycopg2.connect(
            user=config['diagnostics.db']['dbusername'],
            password=config['diagnostics.db']['dbpassword'],
            host=config['diagnostics.db']['dbhost'],
            database=config['diagnostics.db']['dbname']
        )
        conn.autocommit = True
    except psycopg2.Error as e:
        sys.exit(f"No connection to the database: {e}")

    return conn, config

def set_job_completed(job_id, conn, result_json, job_user='unknown', pid=0):
    """
    updates db that the job processing completed succesfully, set status = 3,
    and endruntimestamp to current timestamp
    :param job_id: job's id
    :param conn: db connection object
    :param result_json: dictionary of output file and messages
    :param job_user: job's user
    :param pid: job's pid
    :return: True
    """
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tDB update that the job processing completed".format(pid, job_id, job_user))
    timestamp = int(time.time())
    result = json.dumps(result_json)
    cur = conn.cursor()
    query = "UPDATE osteonet_optimization_jobs SET status = 3, endtimestamp = %s, result = %s WHERE id = %s"
    cur.execute(query, (timestamp, result, job_id))
    return True


# Set current job to completed with error

def set_job_error(job_id, conn, job_user='unknown', pid=0, result={"error": "Unknown error."}):
    """
    updates db that the job processing completed unsuccessfully, set status = 4,
    and startruntimestamp and endtimestamp to current timestamp
    :param job_id: job's id
    :param conn: db connection object
    :param job_user: job's user
    :param pid: job's pid
    :param result: the error message in json that should be updated
    :return: True
    """
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tDB update that the job encountered an error".format(pid, job_id, job_user))
    timestamp = int(time.time())
    # result = "{\"error\": \"Either miRNA or Targets are not existing.\"}"
    cur = conn.cursor()
    query = "UPDATE optimization_osteonet_jobs SET status = 4, endtimestamp = %s, result = %s WHERE id = %s"
    cur.execute(query, (timestamp, json.dumps(result), job_id))
    return True


if __name__ == '__main__':
    args = readArguments(sys.argv[1:])  # Get parameters based on input
    varDict = vars(args)  # gives dictionary of variable name and value; NOT Namespace
    #main(varDict)  # pass the dictionary of arguments to main
    main(varDict, jobid=args.jobid) 