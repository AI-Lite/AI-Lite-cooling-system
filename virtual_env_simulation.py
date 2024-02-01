#coding:utf-8
#import time
#import math
#import copy
#import random
import os
import numpy as np
#from sklearn.externals 
import joblib
import logging

TARGET_TEMP = 23


class EnvTemp(object):
    def __init__(self):
        generator_0 = [[29.1, 83], [29.1, 83], [29.1, 83], [29.1, 83], [29.1, 83], [29.1, 83],
                       [29.1, 83], [29.1, 83], [29.1, 83], [29.1, 83], [29.1, 83], [29.1, 83],
                       [29.0, 84], [29.0, 84], [29.0, 84], [29.0, 84], [29.0, 84], [29.0, 84],
                       [29.0, 84], [29.0, 84], [29.0, 84], [29.0, 84], [29.0, 84], [29.0, 84],
                       [28.9, 83], [28.9, 83], [28.9, 83], [28.9, 83], [28.9, 83], [28.9, 83],
                       [28.9, 83], [28.9, 83], [28.9, 83], [28.9, 83], [28.9, 83], [28.9, 83],
                       [28.9, 82], [28.9, 82], [28.9, 82], [28.9, 82], [28.9, 82], [28.9, 82],
                       [28.9, 82], [28.9, 82], [28.9, 82], [28.9, 82], [28.9, 82], [28.9, 82],
                       [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83],
                       [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83],
                       [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83],
                       [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83],
                       [28.6, 84], [28.6, 84], [28.6, 84], [28.6, 84], [28.6, 84], [28.6, 84],
                       [28.6, 84], [28.6, 84], [28.6, 84], [28.6, 84], [28.6, 84], [28.6, 84],
                       [28.9, 84], [28.9, 84], [28.9, 84], [28.9, 84], [28.9, 84], [28.9, 84],
                       [28.9, 84], [28.9, 84], [28.9, 84], [28.9, 84], [28.9, 84], [28.9, 84],
                       [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81],
                       [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81],
                       [29.8, 81], [29.8, 81], [29.8, 81], [29.8, 81], [29.8, 81], [29.8, 81],
                       [29.8, 81], [29.8, 81], [29.8, 81], [29.8, 81], [29.8, 81], [29.8, 81],
                       [30.1, 79], [30.1, 79], [30.1, 79], [30.1, 79], [30.1, 79], [30.1, 79],
                       [30.1, 79], [30.1, 79], [30.1, 79], [30.1, 79], [30.1, 79], [30.1, 79],
                       [30.6, 74], [30.6, 74], [30.6, 74], [30.6, 74], [30.6, 74], [30.6, 74],
                       [30.6, 74], [30.6, 74], [30.6, 74], [30.6, 74], [30.6, 74], [30.6, 74],
                       [31.6, 71], [31.6, 71], [31.6, 71], [31.6, 71], [31.6, 71], [31.6, 71],
                       [31.6, 71], [31.6, 71], [31.6, 71], [31.6, 71], [31.6, 71], [31.6, 71],
                       [31.9, 67], [31.9, 67], [31.9, 67], [31.9, 67], [31.9, 67], [31.9, 67],
                       [31.9, 67], [31.9, 67], [31.9, 67], [31.9, 67], [31.9, 67], [31.9, 67],
                       [31.5, 67], [31.5, 67], [31.5, 67], [31.5, 67], [31.5, 67], [31.5, 67],
                       [31.5, 67], [31.5, 67], [31.5, 67], [31.5, 67], [31.5, 67], [31.5, 67],
                       [31.3, 68], [31.3, 68], [31.3, 68], [31.3, 68], [31.3, 68], [31.3, 68],
                       [31.3, 68], [31.3, 68], [31.3, 68], [31.3, 68], [31.3, 68], [31.3, 68],
                       [31.0, 70], [31.0, 70], [31.0, 70], [31.0, 70], [31.0, 70], [31.0, 70],
                       [31.0, 70], [31.0, 70], [31.0, 70], [31.0, 70], [31.0, 70], [31.0, 70],
                       [30.5, 74], [30.5, 74], [30.5, 74], [30.5, 74], [30.5, 74], [30.5, 74],
                       [30.5, 74], [30.5, 74], [30.5, 74], [30.5, 74], [30.5, 74], [30.5, 74],
                       [30.2, 75], [30.2, 75], [30.2, 75], [30.2, 75], [30.2, 75], [30.2, 75],
                       [30.2, 75], [30.2, 75], [30.2, 75], [30.2, 75], [30.2, 75], [30.2, 75],
                       [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74],
                       [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74],
                       [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79],
                       [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79],
                       [29.5, 79], [29.5, 79], [29.5, 79], [29.5, 79], [29.5, 79], [29.5, 79],
                       [29.5, 79], [29.5, 79], [29.5, 79], [29.5, 79], [29.5, 79], [29.5, 79],
                       [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79],
                       [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79],
                       [29.3, 79], [29.3, 79], [29.3, 79], [29.3, 79], [29.3, 79], [29.3, 79],
                       [29.3, 79], [29.3, 79], [29.3, 79], [29.3, 79], [29.3, 79], [29.3, 79]]
        generator_1 = [[29.2, 80], [29.2, 80], [29.2, 80], [29.2, 80], [29.2, 80], [29.2, 80],
                       [29.2, 80], [29.2, 80], [29.2, 80], [29.2, 80], [29.2, 80], [29.2, 80],
                       [29.1, 81], [29.1, 81], [29.1, 81], [29.1, 81], [29.1, 81], [29.1, 81],
                       [29.1, 81], [29.1, 81], [29.1, 81], [29.1, 81], [29.1, 81], [29.1, 81],
                       [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82],
                       [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82],
                       [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82],
                       [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82], [29.0, 82],
                       [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83],
                       [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83],
                       [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83],
                       [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83],
                       [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83],
                       [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83], [28.7, 83],
                       [28.9, 81], [28.9, 81], [28.9, 81], [28.9, 81], [28.9, 81], [28.9, 81],
                       [28.9, 81], [28.9, 81], [28.9, 81], [28.9, 81], [28.9, 81], [28.9, 81],
                       [29.2, 78], [29.2, 78], [29.2, 78], [29.2, 78], [29.2, 78], [29.2, 78],
                       [29.2, 78], [29.2, 78], [29.2, 78], [29.2, 78], [29.2, 78], [29.2, 78],
                       [30.2, 73], [30.2, 73], [30.2, 73], [30.2, 73], [30.2, 73], [30.2, 73],
                       [30.2, 73], [30.2, 73], [30.2, 73], [30.2, 73], [30.2, 73], [30.2, 73],
                       [30.8, 68], [30.8, 68], [30.8, 68], [30.8, 68], [30.8, 68], [30.8, 68],
                       [30.8, 68], [30.8, 68], [30.8, 68], [30.8, 68], [30.8, 68], [30.8, 68],
                       [31.0, 67], [31.0, 67], [31.0, 67], [31.0, 67], [31.0, 67], [31.0, 67],
                       [31.0, 67], [31.0, 67], [31.0, 67], [31.0, 67], [31.0, 67], [31.0, 67],
                       [31.7, 63], [31.7, 63], [31.7, 63], [31.7, 63], [31.7, 63], [31.7, 63],
                       [31.7, 63], [31.7, 63], [31.7, 63], [31.7, 63], [31.7, 63], [31.7, 63],
                       [31.7, 62], [31.7, 62], [31.7, 62], [31.7, 62], [31.7, 62], [31.7, 62],
                       [31.7, 62], [31.7, 62], [31.7, 62], [31.7, 62], [31.7, 62], [31.7, 62],
                       [31.2, 67], [31.2, 67], [31.2, 67], [31.2, 67], [31.2, 67], [31.2, 67],
                       [31.2, 67], [31.2, 67], [31.2, 67], [31.2, 67], [31.2, 67], [31.2, 67],
                       [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64],
                       [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64],
                       [30.9, 68], [30.9, 68], [30.9, 68], [30.9, 68], [30.9, 68], [30.9, 68],
                       [30.9, 68], [30.9, 68], [30.9, 68], [30.9, 68], [30.9, 68], [30.9, 68],
                       [31.1, 67], [31.1, 67], [31.1, 67], [31.1, 67], [31.1, 67], [31.1, 67],
                       [31.1, 67], [31.1, 67], [31.1, 67], [31.1, 67], [31.1, 67], [31.1, 67],
                       [30.5, 72], [30.5, 72], [30.5, 72], [30.5, 72], [30.5, 72], [30.5, 72],
                       [30.5, 72], [30.5, 72], [30.5, 72], [30.5, 72], [30.5, 72], [30.5, 72],
                       [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74],
                       [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74], [30.1, 74],
                       [29.8, 77], [29.8, 77], [29.8, 77], [29.8, 77], [29.8, 77], [29.8, 77],
                       [29.8, 77], [29.8, 77], [29.8, 77], [29.8, 77], [29.8, 77], [29.8, 77],
                       [29.9, 75], [29.9, 75], [29.9, 75], [29.9, 75], [29.9, 75], [29.9, 75],
                       [29.9, 75], [29.9, 75], [29.9, 75], [29.9, 75], [29.9, 75], [29.9, 75],
                       [29.9, 76], [29.9, 76], [29.9, 76], [29.9, 76], [29.9, 76], [29.9, 76],
                       [29.9, 76], [29.9, 76], [29.9, 76], [29.9, 76], [29.9, 76], [29.9, 76],
                       [29.7, 78], [29.7, 78], [29.7, 78], [29.7, 78], [29.7, 78], [29.7, 78],
                       [29.7, 78], [29.7, 78], [29.7, 78], [29.7, 78], [29.7, 78], [29.7, 78]]
        generator_2 = [[29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81],
                       [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81], [29.4, 81],
                       [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79],
                       [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79],
                       [29.6, 76], [29.6, 76], [29.6, 76], [29.6, 76], [29.6, 76], [29.6, 76],
                       [29.6, 76], [29.6, 76], [29.6, 76], [29.6, 76], [29.6, 76], [29.6, 76],
                       [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79],
                       [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79],
                       [28.9, 80], [28.9, 80], [28.9, 80], [28.9, 80], [28.9, 80], [28.9, 80],
                       [28.9, 80], [28.9, 80], [28.9, 80], [28.9, 80], [28.9, 80], [28.9, 80],
                       [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78],
                       [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78],
                       [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79],
                       [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79], [29.2, 79],
                       [29.4, 78], [29.4, 78], [29.4, 78], [29.4, 78], [29.4, 78], [29.4, 78],
                       [29.4, 78], [29.4, 78], [29.4, 78], [29.4, 78], [29.4, 78], [29.4, 78],
                       [29.5, 80], [29.5, 80], [29.5, 80], [29.5, 80], [29.5, 80], [29.5, 80],
                       [29.5, 80], [29.5, 80], [29.5, 80], [29.5, 80], [29.5, 80], [29.5, 80],
                       [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79],
                       [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79], [29.6, 79],
                       [30.4, 76], [30.4, 76], [30.4, 76], [30.4, 76], [30.4, 76], [30.4, 76],
                       [30.4, 76], [30.4, 76], [30.4, 76], [30.4, 76], [30.4, 76], [30.4, 76],
                       [31.2, 73], [31.2, 73], [31.2, 73], [31.2, 73], [31.2, 73], [31.2, 73],
                       [31.2, 73], [31.2, 73], [31.2, 73], [31.2, 73], [31.2, 73], [31.2, 73],
                       [31.7, 66], [31.7, 66], [31.7, 66], [31.7, 66], [31.7, 66], [31.7, 66],
                       [31.7, 66], [31.7, 66], [31.7, 66], [31.7, 66], [31.7, 66], [31.7, 66],
                       [31.6, 68], [31.6, 68], [31.6, 68], [31.6, 68], [31.6, 68], [31.6, 68],
                       [31.6, 68], [31.6, 68], [31.6, 68], [31.6, 68], [31.6, 68], [31.6, 68],
                       [32.1, 65], [32.1, 65], [32.1, 65], [32.1, 65], [32.1, 65], [32.1, 65],
                       [32.1, 65], [32.1, 65], [32.1, 65], [32.1, 65], [32.1, 65], [32.1, 65],
                       [32.0, 65], [32.0, 65], [32.0, 65], [32.0, 65], [32.0, 65], [32.0, 65],
                       [32.0, 65], [32.0, 65], [32.0, 65], [32.0, 65], [32.0, 65], [32.0, 65],
                       [31.4, 68], [31.4, 68], [31.4, 68], [31.4, 68], [31.4, 68], [31.4, 68],
                       [31.4, 68], [31.4, 68], [31.4, 68], [31.4, 68], [31.4, 68], [31.4, 68],
                       [31.6, 66], [31.6, 66], [31.6, 66], [31.6, 66], [31.6, 66], [31.6, 66],
                       [31.6, 66], [31.6, 66], [31.6, 66], [31.6, 66], [31.6, 66], [31.6, 66],
                       [31.3, 69], [31.3, 69], [31.3, 69], [31.3, 69], [31.3, 69], [31.3, 69],
                       [31.3, 69], [31.3, 69], [31.3, 69], [31.3, 69], [31.3, 69], [31.3, 69],
                       [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70],
                       [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70],
                       [29.8, 76], [29.8, 76], [29.8, 76], [29.8, 76], [29.8, 76], [29.8, 76],
                       [29.8, 76], [29.8, 76], [29.8, 76], [29.8, 76], [29.8, 76], [29.8, 76],
                       [30.0, 74], [30.0, 74], [30.0, 74], [30.0, 74], [30.0, 74], [30.0, 74],
                       [30.0, 74], [30.0, 74], [30.0, 74], [30.0, 74], [30.0, 74], [30.0, 74],
                       [29.5, 78], [29.5, 78], [29.5, 78], [29.5, 78], [29.5, 78], [29.5, 78],
                       [29.5, 78], [29.5, 78], [29.5, 78], [29.5, 78], [29.5, 78], [29.5, 78],
                       [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79],
                       [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79], [29.4, 79]]
        generator_3 = [[29.1, 80], [29.1, 80], [29.1, 80], [29.1, 80], [29.1, 80], [29.1, 80],
                       [29.1, 80], [29.1, 80], [29.1, 80], [29.1, 80], [29.1, 80], [29.1, 80],
                       [28.9, 79], [28.9, 79], [28.9, 79], [28.9, 79], [28.9, 79], [28.9, 79],
                       [28.9, 79], [28.9, 79], [28.9, 79], [28.9, 79], [28.9, 79], [28.9, 79],
                       [28.7, 80], [28.7, 80], [28.7, 80], [28.7, 80], [28.7, 80], [28.7, 80],
                       [28.7, 80], [28.7, 80], [28.7, 80], [28.7, 80], [28.7, 80], [28.7, 80],
                       [28.9, 78], [28.9, 78], [28.9, 78], [28.9, 78], [28.9, 78], [28.9, 78],
                       [28.9, 78], [28.9, 78], [28.9, 78], [28.9, 78], [28.9, 78], [28.9, 78],
                       [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83],
                       [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83], [28.6, 83],
                       [28.7, 78], [28.7, 78], [28.7, 78], [28.7, 78], [28.7, 78], [28.7, 78],
                       [28.7, 78], [28.7, 78], [28.7, 78], [28.7, 78], [28.7, 78], [28.7, 78],
                       [28.5, 80], [28.5, 80], [28.5, 80], [28.5, 80], [28.5, 80], [28.5, 80],
                       [28.5, 80], [28.5, 80], [28.5, 80], [28.5, 80], [28.5, 80], [28.5, 80],
                       [29.0, 80], [29.0, 80], [29.0, 80], [29.0, 80], [29.0, 80], [29.0, 80],
                       [29.0, 80], [29.0, 80], [29.0, 80], [29.0, 80], [29.0, 80], [29.0, 80],
                       [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78],
                       [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78],
                       [30.4, 74], [30.4, 74], [30.4, 74], [30.4, 74], [30.4, 74], [30.4, 74],
                       [30.4, 74], [30.4, 74], [30.4, 74], [30.4, 74], [30.4, 74], [30.4, 74],
                       [31.1, 61], [31.1, 61], [31.1, 61], [31.1, 61], [31.1, 61], [31.1, 61],
                       [31.1, 61], [31.1, 61], [31.1, 61], [31.1, 61], [31.1, 61], [31.1, 61],
                       [32.8, 60], [32.8, 60], [32.8, 60], [32.8, 60], [32.8, 60], [32.8, 60],
                       [32.8, 60], [32.8, 60], [32.8, 60], [32.8, 60], [32.8, 60], [32.8, 60],
                       [32.8, 55], [32.8, 55], [32.8, 55], [32.8, 55], [32.8, 55], [32.8, 55],
                       [32.8, 55], [32.8, 55], [32.8, 55], [32.8, 55], [32.8, 55], [32.8, 55],
                       [32.9, 57], [32.9, 57], [32.9, 57], [32.9, 57], [32.9, 57], [32.9, 57],
                       [32.9, 57], [32.9, 57], [32.9, 57], [32.9, 57], [32.9, 57], [32.9, 57],
                       [33.6, 52], [33.6, 52], [33.6, 52], [33.6, 52], [33.6, 52], [33.6, 52],
                       [33.6, 52], [33.6, 52], [33.6, 52], [33.6, 52], [33.6, 52], [33.6, 52],
                       [32.8, 57], [32.8, 57], [32.8, 57], [32.8, 57], [32.8, 57], [32.8, 57],
                       [32.8, 57], [32.8, 57], [32.8, 57], [32.8, 57], [32.8, 57], [32.8, 57],
                       [32.4, 57], [32.4, 57], [32.4, 57], [32.4, 57], [32.4, 57], [32.4, 57],
                       [32.4, 57], [32.4, 57], [32.4, 57], [32.4, 57], [32.4, 57], [32.4, 57],
                       [32.5, 60], [32.5, 60], [32.5, 60], [32.5, 60], [32.5, 60], [32.5, 60],
                       [32.5, 60], [32.5, 60], [32.5, 60], [32.5, 60], [32.5, 60], [32.5, 60],
                       [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64],
                       [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64], [31.6, 64],
                       [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70],
                       [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70], [30.5, 70],
                       [29.8, 73], [29.8, 73], [29.8, 73], [29.8, 73], [29.8, 73], [29.8, 73],
                       [29.8, 73], [29.8, 73], [29.8, 73], [29.8, 73], [29.8, 73], [29.8, 73],
                       [29.9, 72], [29.9, 72], [29.9, 72], [29.9, 72], [29.9, 72], [29.9, 72],
                       [29.9, 72], [29.9, 72], [29.9, 72], [29.9, 72], [29.9, 72], [29.9, 72],
                       [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78],
                       [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78], [29.3, 78],
                       [29.0, 81], [29.0, 81], [29.0, 81], [29.0, 81], [29.0, 81], [29.0, 81],
                       [29.0, 81], [29.0, 81], [29.0, 81], [29.0, 81], [29.0, 81], [29.0, 81]]
        # generator_load = [88]#,114,139]
        it_load = [[26, 53], [53, 88, 114, 139, 114, 88], [26, 53, 88, 114, 139, 114, 88, 53], [26,26]]
        generator_load = it_load[1]
        self.__temp_generatores = [generator_0, generator_1, generator_2, generator_3]
        self.__load_generatores = generator_load
        self.__generator_idx = 0
        self.__cur_idx = 0
        self.__load_idx = 0
        self.num = 0

    def get(self):
        temp = self.__temp_generatores[self.__generator_idx][self.__cur_idx][0]
        hum = self.__temp_generatores[self.__generator_idx][self.__cur_idx][1]
        load = self.__load_generatores[self.__load_idx]*1000
        # load = self.__load_generatores[self.__load_idx]
        # load = self.__load_generatores[self.__load_idx]
        time = 0
        if self.__cur_idx != 288 - 1:
            time += 1
            self.__cur_idx += 1
            #self.__load_idx = np.random.choice(list(range(len(self.__load_generatores))))
            # if time == 18:
            self.__load_idx = (self.__load_idx + 1) % 6
                # time = 0
        else:
            time += 1
            self.__cur_idx = 0
            self.num = self.num + 1
            #self.__load_idx = np.random.choice(list(range(len(self.__load_generatores))))
            # if self.num <= 12:
            #      self.__load_idx = 0
            # else:
            #      self.__load_idx = 1
            self.__generator_idx = np.random.choice(list(range(len(self.__temp_generatores))))
        return temp, hum, load


_env_temp_ins = EnvTemp()


def _get_env_temp():
    return _env_temp_ins.get()  # 获得一个温度,湿度，和负载


_SettingTempArray = [21, 21, 21, 21, 21]
_OnoffFlagArray = [1, 1, 1, 1, 1]

_EnvState = {}
Model=joblib.load(os.path.join(os.path.dirname(__file__), 'model/model.pkl'))  # 加载训练好的xgboost模型
Scaler_x=joblib.load(os.path.join(os.path.dirname(__file__), 'model/scaler_x.pkl'))  # 加载x归一化模型
# Scaler_y = joblib.load('./model/scaler_y.pkl')  # 加载y归一化模型
Temp_Pos = ['AcDeliveryTemp_24','AcDeliveryTemp_25','AcDeliveryTemp_27','AcDeliveryTemp_28','AcDeliveryTemp_43',
            'AcReturnTemp_24','AcReturnTemp_25','AcReturnTemp_27','AcReturnTemp_28','AcReturnTemp_43','RackColdTemp_12',
            'RackColdTemp_13','RackColdTemp_14','RackColdTemp_15','RackColdTemp_22','RackColdTemp_23','RackColdTemp_24',
            'RackColdTemp_25','RackColdTemp_27','RackColdTemp_29','RackColdTemp_3','RackColdTemp_30','RackColdTemp_31',
            'RackColdTemp_34','RackColdTemp_35','RackColdTemp_4','RackColdTemp_7','RackColdTemp_8','RackColdTemp_9',
            'RackHotTemp_12','RackHotTemp_13','RackHotTemp_14','RackHotTemp_15','RackHotTemp_22','RackHotTemp_23',
            'RackHotTemp_24','RackHotTemp_25','RackHotTemp_27','RackHotTemp_29','RackHotTemp_3','RackHotTemp_30',
            'RackHotTemp_31','RackHotTemp_34','RackHotTemp_35','RackHotTemp_4','RackHotTemp_7','RackHotTemp_8',
            'RackHotTemp_9','InstantPue']


def get_cold_temp_mean__(_EnvState):
    fRackColdTempes = []
    for label in _EnvState:
        if label.find("RackColdTemp_") == 0:
            fRackColdTempes.append(_EnvState[label])
    return round(np.mean(fRackColdTempes), 1)

def get_env_state__():
    env_temp, hum, load = _get_env_temp()
    # print(env_temp)
    _EnvState['EnvTemp'] = env_temp
    _EnvState["ItElectricPower"] =load
    _EnvState['Humidity'] = hum
    set_ac_open = np.array(_OnoffFlagArray)
    set_ac_temp = np.array(_SettingTempArray)
    env_state = np.array([env_temp, hum, load])
    data_input = np.concatenate((env_state, set_ac_temp, set_ac_open))
    data_input = np.reshape(data_input, (-1, 13))
    data_input[:, :8] = Scaler_x.transform(data_input[:, :8])
    pred = Model.predict(data_input)
    #pred = Scaler_y.inverse_transform(pred)
    pred = np.reshape(pred, (-1))
    pred_arr = pred.tolist()
    # print(pred_arr)

    for i in range(len(pred_arr)):
        _EnvState[Temp_Pos[i]] = round(pred_arr[i], 4)
    _EnvState["TotalElectricPower"] = load * _EnvState["InstantPue"]
    
    _EnvState["AcWorkTime_24"] = 1 if _OnoffFlagArray[0] else -1
    _EnvState["AcWorkTime_25"] = 1 if _OnoffFlagArray[1] else -1
    _EnvState["AcWorkTime_27"] = 1 if _OnoffFlagArray[2] else -1
    _EnvState["AcWorkTime_28"] = 1 if _OnoffFlagArray[3] else -1
    _EnvState["AcWorkTime_43"] = 1 if _OnoffFlagArray[4] else -1
    
    _EnvState["AcSetTemp_24"] = _SettingTempArray[0]
    _EnvState["AcSetTemp_25"] = _SettingTempArray[1]
    _EnvState["AcSetTemp_27"] = _SettingTempArray[2]
    _EnvState["AcSetTemp_28"] = _SettingTempArray[3]
    _EnvState["AcSetTemp_43"] = _SettingTempArray[4]
    
    _EnvState["AcOnoffState_24"] = _OnoffFlagArray[0]
    _EnvState["AcOnoffState_25"] = _OnoffFlagArray[1]
    _EnvState["AcOnoffState_27"] = _OnoffFlagArray[2]
    _EnvState["AcOnoffState_28"] = _OnoffFlagArray[3]
    _EnvState["AcOnoffState_43"] = _OnoffFlagArray[4]
    _EnvState["ColdTempMean"] = get_cold_temp_mean__(_EnvState)
    _EnvState["AcXPos_24"] = 1092
    _EnvState["AcXPos_25"] = 752
    _EnvState["AcXPos_27"] = 752
    _EnvState["AcXPos_28"] = 412
    _EnvState["AcXPos_43"] = 1160
    _EnvState["AcYPos_24"] = 350
    _EnvState["AcYPos_25"] = 350
    _EnvState["AcYPos_27"] = 100
    _EnvState["AcYPos_28"] = 100
    _EnvState["AcYPos_43"] = 100
    _EnvState["RackXPos_12"] = 820
    _EnvState["RackXPos_13"] = 888
    _EnvState["RackXPos_14"] = 956
    _EnvState["RackXPos_15"] = 1024
    _EnvState["RackXPos_2"] = 140
    _EnvState["RackXPos_22"] = 344
    _EnvState["RackXPos_23"] = 412
    _EnvState["RackXPos_24"] = 480
    _EnvState["RackXPos_25"] = 548
    _EnvState["RackXPos_27"] = 684
    _EnvState["RackXPos_29"] = 820
    _EnvState["RackXPos_3"] = 208
    _EnvState["RackXPos_30"] = 888
    _EnvState["RackXPos_31"] = 956
    _EnvState["RackXPos_34"] = 1160
    _EnvState["RackXPos_35"] = 1228
    _EnvState["RackXPos_4"] = 276
    _EnvState["RackXPos_7"] = 480
    _EnvState["RackXPos_8"] = 548
    _EnvState["RackXPos_9"] = 616
    _EnvState["RackYPos_12"] = 100
    _EnvState["RackYPos_13"] = 100
    _EnvState["RackYPos_14"] = 100
    _EnvState["RackYPos_15"] = 100
    _EnvState["RackYPos_2"] = 100
    _EnvState["RackYPos_22"] = 350
    _EnvState["RackYPos_23"] = 350
    _EnvState["RackYPos_24"] = 350
    _EnvState["RackYPos_25"] = 350
    _EnvState["RackYPos_27"] = 350
    _EnvState["RackYPos_29"] = 350
    _EnvState["RackYPos_3"] = 100
    _EnvState["RackYPos_30"] = 350
    _EnvState["RackYPos_31"] = 350
    _EnvState["RackYPos_34"] = 350
    _EnvState["RackYPos_35"] = 350
    _EnvState["RackYPos_4"] = 100
    _EnvState["RackYPos_7"] = 100
    _EnvState["RackYPos_8"] = 100
    _EnvState["RackYPos_9"] = 100
    return _EnvState


def get_env_state():
    jEnvState = get_env_state__()
    
    jEnvState2 = {}
    aParamLabeles = []
    aParamDataes = []
    for sLabel in jEnvState:
        aParamLabeles.append(sLabel)
        aParamDataes.append(jEnvState[sLabel])
    jEnvState2["ParamLabel"] = aParamLabeles
    jEnvState2["ParamData"] = aParamDataes
    
    return jEnvState, jEnvState2

def set_setting_temp(temp_array):
    for i in range(5):
        if temp_array[i] < 12 or temp_array[i] > 34:
            print("The setting temp invalid")
            break
    for i in range(5):
        _SettingTempArray[i] = temp_array[i]


def set_onoff(onoff_array):
    logging.info(onoff_array)
    for i in range(5):
        _OnoffFlagArray[i] = onoff_array[i]
    

__all__ = ["get_env_state",
        "set_setting_temp",
        "set_onoff"]


def main1():
    import matplotlib.pyplot as plt

    def sub_main1(ac_num=5):
        set_setting_temp([23] * 5)
        set_onoff([1 for i in range(ac_num)] + [0 for j in range(5-ac_num)])
        aPues = []
        for num in range(288):
            print(".", end="")
            jEnvState, jEnvState2 = get_env_state()
            print(jEnvState)
            aPues.append(jEnvState["InstantPue"])
        print()
        fMeanPue = np.mean(aPues)
        length = len(aPues)
        plt.plot(range(length), aPues)
        plt.plot(range(length), [fMeanPue] * length)
        plt.show()

    for ac_num in range(1, 5):
        print(ac_num)
        sub_main1(ac_num)

def main2():
    jEnvState, jEnvState2 = get_env_state()
    print(jEnvState)
    set_onoff([1,1,1,1,0])
    set_setting_temp([20,20,20,20,20])
    jEnvState, jEnvState2 = get_env_state()
    print(jEnvState2)

if "__main__" == __name__:
    #main1()
    main2()
