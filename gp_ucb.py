import csv
import random
import time
import math
import copy
import random 
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt
from math import log
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
#from info_process import get_job_id, get_job_throughput,get_job_cpu, get_operator_cpu, get_operator_throughput

scalar_x = 16.0
scalar = 8.0 # The scalar of the index
max_parallelism = 60
max_time = 700000
# cpu_threshold = 0.7
current_time = 1
search_num = 200
gradient_descent_iteration = 10000
sigma = 0.1
l = 20
learning_rate = 0.05   # The step-size of gradient descent


def gaussian_process_bandit(parameter, performance, candidate_configuration):
    # # normalize the performance and target
    # performance = performance + [target]
    # performance = preprocessing.scale(performance)
    # target = performance[-1]
    # performance = performance[:-1]
    #
    # # generate the candidate cloud configuration
    dimension = len(parameter[0])
    # # print(candidate_configuration)
    # # compute the K_t, k_t(x), k_t(x,x')
    num_candidate = len(candidate_configuration)
    count = len(performance)
    large_tmpk = np.exp(-squareform(pdist(np.array(parameter),'euclidean'))**2 / (2 * l**2))
    inverse_tmpk = np.linalg.inv(large_tmpk + sigma**2 * np.eye(count))
    tmp_mu = inverse_tmpk.dot(np.array(performance))

    score = []
    for j in range(num_candidate):
        small_tmpk = np.exp(-np.sum(np.abs((candidate_configuration[j] + np.zeros((count, dimension))) -
                                           np.array(parameter))**2, axis = -1) / (2* l **2))
        mu = small_tmpk.dot(tmp_mu)
        var = sqrt(1-small_tmpk.dot(inverse_tmpk.dot(small_tmpk)))
        beta_t = sqrt(2*log((count*math.pi)**2/current_time/0.05))
        score.append(beta_t * var + mu)
    index = (np.argmax(score))
    print(candidate_configuration[index])
        # ratting[j,:] = [mu,var,  beta_t * var  - abs(mu - target)]
    return candidate_configuration[index]
    # flag = np.argmax(ratting[:,2])

    #print(ratting)
    #print(beta_t)
    # return candidate_configuration[flag]
        

def pick_nex_configuration(searched_configuration, performance):
    count = len(performance)
    dimension = len(searched_configuration[0])
    large_tmpk = np.exp(-squareform(pdist(np.array(searched_configuration), 'euclidean'))**2 / (2 * l**2))
    # print(large_tmpk)
    inverse_tmpk = np.linalg.inv(large_tmpk + sigma**2 * np.eye(count))
    tmp_mu = inverse_tmpk.dot(np.array(performance))
    candidate_configuration = [random.uniform(0.0, scalar_x), random.uniform(0.0, scalar)]
    searched_configuration = np.array(searched_configuration)

    # perform gradient descent to find the configuration that maximizes the optimizatin objective
    for j in range(gradient_descent_iteration):
        small_tmpk = np.exp(-np.sum(np.abs((candidate_configuration + np.zeros((count, dimension)))
                                           - np.array(searched_configuration))**2, axis=-1) / (2 * l ** 2))
        # print(small_tmpk)
        small_tmpk_gradient_x_mu = small_tmpk*(np.array(searched_configuration[:, 0]) - candidate_configuration[0]) / l**2
        small_tmpk_gradient_y_mu = small_tmpk*(np.array(searched_configuration[:, 1]) - candidate_configuration[1]) / l**2
        # mu = small_tmpk.dot(tmp_mu)
        var = sqrt(1-small_tmpk.dot(inverse_tmpk.dot(small_tmpk)))
        beta_t = sqrt(2*log((count*math.pi)**2/current_time/0.05))
        gradient_x = small_tmpk_gradient_x_mu.dot(tmp_mu) + beta_t * 1/2 * 1/var * \
                     (-small_tmpk_gradient_x_mu.dot(inverse_tmpk.dot(small_tmpk)) -
                      small_tmpk.dot(inverse_tmpk.dot(small_tmpk_gradient_x_mu)))
        # print(gradient_x)
        gradient_y = small_tmpk_gradient_y_mu.dot(tmp_mu) + beta_t * 1/2 * 1/var * \
                     (-small_tmpk_gradient_y_mu.dot(inverse_tmpk.dot(small_tmpk)) -
                      small_tmpk.dot(inverse_tmpk.dot(small_tmpk_gradient_y_mu)))
        candidate_configuration = [candidate_configuration[0] + learning_rate*gradient_x,
                                   candidate_configuration[1] + learning_rate*gradient_y]
        # print(candidate_configuration)
        gradient_projection = []
        for k, value in enumerate(candidate_configuration):
            if value < 0.0:
                value = 0.0
            if k == 0 and value > scalar_x:
                value = scalar_x
            if k == 1 and value > scalar:
                value = scalar
            gradient_projection.append(value)
        candidate_configuration = gradient_projection
    # print(candidate_configuration)
    return candidate_configuration
        # performance_metric = beta_t * var + mu


if __name__ == '__main__':
    # print(gaussian_process_bandit(configuration_info,performance_info,60000))
    path = './statistics_1.csv'
    data = pd.read_csv(path, header=0)
    # print(max(data))
    data = np.array(data)
    # print(np.where(data == np.max(data)))
    # print(data[20-1][200-1])
    num_row = data.shape[0]
    num_column = data.shape[1]
    print(data.max(1))
    initial_x = 40
    initial_y = 100
    # initial_x = random.randint(0, num_row)
    # initial_y = random.randint(0, num_column)
    configuration_info = [[initial_x/num_row*scalar_x, initial_y/num_column*scalar]]
    # print(configuration_info)
    performance_info = [data[initial_x-1][initial_y-1]]
    print(performance_info)

    # tmp = []
    # candidate = []
    # for i in range(data.shape[0]):
    #     candidate.append([(i+1)/num_row*scalar])
    # for i in candidate:
    #     for j in range(data.shape[1]/100):
    #         tmp.append(i+[(1+j)*100/num_column*scalar])
    # candidate = tmp

    for i in range(search_num):
        # new_configuration = gaussian_process_bandit(configuration_info, performance_info, candidate)
        new_configuration = pick_nex_configuration(configuration_info, performance_info)
        # To take integers which are most close to the fractional value
        new_configuration = [int(new_configuration[0]*num_row/scalar_x + 0.5),
                             int(new_configuration[1]*num_column/scalar + 0.5)]
        # if new_configuration[0] >= num_row:
        #     new_configuration[0] = num_row - 1
        # if new_configuration[1] >= num_column:
        #     new_configuration[1] = num_column - 1
        new_performance = data[new_configuration[0]-1][new_configuration[1]-1]
        print(new_configuration)
        print(new_performance)
        performance_info.append(new_performance)
        configuration_info.append([new_configuration[0]/num_row*scalar_x, new_configuration[1]/num_column*scalar])
        current_time = current_time + 1
    print(np.max(performance_info))


