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
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt
from math import log
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
import scipy.spatial
# import seaborn as sns
from matplotlib.ticker import ScalarFormatter


current_time = 1
# sigma = 0.005
sigma = 0.02
# sigma = 0.01
# l = 1.8
l = 8


def gaussian_process_bandit(parameter, performance, candidate_configuration):
    global current_time
    dimension = len(parameter[0])
    num_candidate = len(candidate_configuration)
    count = len(performance)
    # large_tmpk = np.exp(-squareform(pdist(np.array(parameter), 'euclidean')) ** 2 / (2 * l ** 2))
    large_tmpk = np.exp(-squareform(pdist(np.array(parameter), 'cityblock')) / (2 * l))
    # large_tmpk = 1 - squareform(pdist(np.array(parameter), 'cosine'))
    inverse_tmpk = np.linalg.inv(large_tmpk + sigma ** 2 * np.eye(count))
    tmp_mu = inverse_tmpk.dot(np.array(performance))

    score = []
    for j in range(num_candidate):
        # small_tmpk = np.exp(-np.sum(np.abs((candidate_configuration[j] + np.zeros((count, dimension))) -
        #                                   np.array(parameter)) ** 2, axis=-1) / (2 * l ** 2))
        small_tmpk = np.exp(-np.sum(np.abs((candidate_configuration[j] + np.zeros((count, dimension))) -
                                           np.array(parameter)), axis=-1) / (2 * l))
        #small_tmpk = 1 - scipy.spatial.distance.cdist([candidate_configuration[j]], np.array(parameter), "cosine")[0]
        mu = small_tmpk.dot(tmp_mu)
        var = sqrt(1 - small_tmpk.dot(inverse_tmpk.dot(small_tmpk)))
        # beta_t = sqrt(2 * log((count * math.pi) ** 2 / math.sqrt(current_time) / 0.02))
        beta_t = sqrt(2 * log((count * math.pi) ** 2 / current_time / 0.02))
        score.append(beta_t * var + mu)
    current_time += 1
    index = (np.argmax(score))
    return index


