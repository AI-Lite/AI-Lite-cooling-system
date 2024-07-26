#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 本文件在 model_add_test.py 基础上修改了模型输入，将输入的送风温度，回风温度，冷通道温度，热通道温度均改为输入平均值，并且对于没有打开的空调不计入计算

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 本文件在 model.py 基础上加入了测试误差

from sklearn import tree
from joblib import dump, load
from matplotlib import pyplot as plt
import numpy as np
import random
import os
import shutil
import pandas as pd
import time as t
from sklearn.metrics import mean_squared_error
from math import sqrt
import traceback

# from virtual_env_generator_v1_2 import set_setting_temp, set_onoff, get_env_state
from virtual_env_generator_v1 import set_setting_temp, set_onoff, get_env_state
from config import  Config
from preprocess import DataLabelStruct, get_data_dict, data_env_state_normal


state_data = [] # 保存所有的环境状态数据

x = []  # 送风温度, 回风温度, 冷通道温度, 热通道温度, it负载, 空调开关列表, 空调的设定温度
y_pue = []  # pue 值
y_cold_temp = []  # shape = (:, len(rack_areas))

fixed_depth = False
max_tree_depth = 4  # 树的最大深度
model_list = []  # 训练的模型列表
setting_onoff_array = []  # 空调开关列表
setting_temp_array = []  # 空调设置温度列表

count = 0  # 计数用
train_day = 40
train_times = 288 * train_day
model_input = []  # 存储上一个时刻的平均送风温度, 平均回风温度, 平均冷通道温度, 平均热通道温度, it负载, 空调开关列表, 空调的设定温度
data_label_instance = None  # 存储标签分析后的信息, 包括空调的编号, 数量, 送回风温度标签; 机柜编号, 数量, 冷热通道温度标签, 机柜划分区域标签
drst_error = {'Status': 1, 'PredictAction': {}}
trajectory = []  # 存储每18次的状态信息
candidate_array = [-1, 0, 1]  # 待选的设置温度变化值

it_power_main = []
y_pue_main = []
y_cold_temp_main = []
onoff_all = []
set_temp_all = []
onoff_nums = []
mean_pue = []

pue_error = []
y_cold_temp_square_error = []


header_name = ['Timestamp', 'XPosMin', 'XPosMax', 'YPosMin', 'YPosMax', 'AcXPos_1', 'AcYPos_1', 'AcXPos_2', 'AcYPos_2', 'AcXPos_3', 'AcYPos_3', 'AcXPos_4',
                'AcYPos_4', 'AcXPos_5', 'AcYPos_5', 'AcXPos_6', 'AcYPos_6', 'RackXPos_1', 'RackYPos_1', 'RackXPos_2', 'RackYPos_2', 'RackXPos_3',
                'RackYPos_3', 'RackXPos_4', 'RackYPos_4', 'RackXPos_5', 'RackYPos_5', 'RackXPos_6', 'RackYPos_6', 'RackXPos_7', 'RackYPos_7', 'RackXPos_8',
                'RackYPos_8', 'AcSetTemp_1', 'AcOnoffState_1', 'AcDeliveryTemp_1', 'AcWorkTime_1', 'AcReturnTemp_1', 'AcSetTemp_2', 'AcOnoffState_2', 'AcDeliveryTemp_2',
                'AcWorkTime_2', 'AcReturnTemp_2', 'AcSetTemp_3', 'AcOnoffState_3', 'AcDeliveryTemp_3', 'AcWorkTime_3', 'AcReturnTemp_3', 'AcSetTemp_4', 'AcOnoffState_4',
                'AcDeliveryTemp_4', 'AcWorkTime_4', 'AcReturnTemp_4', 'AcSetTemp_5', 'AcOnoffState_5', 'AcDeliveryTemp_5', 'AcWorkTime_5', 'AcReturnTemp_5',
                'AcSetTemp_6', 'AcOnoffState_6', 'AcDeliveryTemp_6', 'AcWorkTime_6', 'AcReturnTemp_6', 'ItElectricPower', 'ItElectricTotalWork', 'TotalElectricPower',
                'TotalElectricTotalWork', 'InstantPue', 'RackColdTemp_1', 'RackHotTemp_1', 'RackColdTemp_2', 'RackHotTemp_2', 'RackColdTemp_3', 'RackHotTemp_3',
                'RackColdTemp_4', 'RackHotTemp_4', 'RackColdTemp_5', 'RackHotTemp_5', 'RackColdTemp_6', 'RackHotTemp_6', 'RackColdTemp_7', 'RackHotTemp_7',
                'RackColdTemp_8', 'RackHotTemp_8', 'ColdTempMean', 'HotTempMean', 'change_onoff', 'change_temp']

data_test_file = pd.read_csv('data.csv').values.tolist()

def rmse():
    global data_test_file, data_label_instance, pue_error, y_cold_temp_square_error

    input = []

    real_pue_array = []
    pre_pue_array = []

    real_cold_temp_array = []
    pre_cold_temp_array = []

    rmse_cold_temp = [0.] * data_label_instance.ac_len

    for file_index in range(0, len(data_test_file)):
        data_test_item = {}

        for col_index in range(len(data_test_file[0])):
            data_test_item[header_name[col_index]] = data_test_file[file_index][col_index]

        data = data_env_state_normal(data_test_item, data_label_instance)

        if file_index == 0:
            input = data[-5:].copy()
            continue

        for label in data_label_instance.ac_onoff_state_labels:
            input.append(data_test_item[label])

        for label in data_label_instance.ac_set_temp_labels:
            input.append(data_test_item[label])

        pre_pue_and_cold_temp = model_predict(input)

        real_pue_array.append(data_test_item['InstantPue'])
        pre_pue_array.append(pre_pue_and_cold_temp[0])

        real_cold_temp_array.append(data[2 * data_label_instance.ac_len : 3 * data_label_instance.ac_len].copy())
        pre_cold_temp_array.append(pre_pue_and_cold_temp[1:].copy())

        input = data[-5:].copy()

    pue_error.append(sqrt(mean_squared_error(real_pue_array, pre_pue_array)))

    for index in range(data_label_instance.ac_len):
        rmse_cold_temp[index] = sqrt(mean_squared_error([item[index] for item in real_cold_temp_array], [item[index] for item in pre_cold_temp_array]))

    y_cold_temp_square_error.append(rmse_cold_temp.copy())


def dump_file():
    global model_list, x, y_pue, y_cold_temp, state_data

    dump(model_list[0], f'{Config.MODEL_DIR.value}/model_pue.model')
    for index in range(1, len(model_list)):
        dump(model_list[index], f'{Config.MODEL_DIR.value}/model_cold_temp_{index - 1}.model')

    x_pd = pd.DataFrame(data=x)
    x_pd.to_csv(f'{Config.DATA_DIR.value}/x.csv', encoding='utf-8', index=False, header=None)

    pue_pd = pd.DataFrame(data=y_pue)
    pue_pd.to_csv(f'{Config.DATA_DIR.value}/y_pue.csv', encoding='utf-8', index=False, header=None)

    cold_temp_pd = pd.DataFrame(data=y_cold_temp)
    cold_temp_pd.to_csv(f'{Config.DATA_DIR.value}/y_cold_temp.csv', encoding='utf-8', index=False, header=None)

    state_pd = pd.DataFrame(data=state_data)
    TIME = t.strftime('%Y%m%d-%H%M%S', t.localtime(t.time()))
    state_pd.to_csv(f'{Config.DATA_DIR.value}/{TIME}.csv', index=False, header=state_data[0].keys())


# 训练模型
def model_train():
    global x, y_pue, y_cold_temp, mean_pue, max_tree_depth, fixed_depth

    model_list = []
    data_rows = len(y_cold_temp)
    data_columns = len(y_cold_temp[0])

    if fixed_depth:
        if len(mean_pue) > 5 and abs(mean_pue[-2]- mean_pue[-1]) > 0.001 :
            fixed_depth = False
            max_tree_depth = max(max_tree_depth, int(data_rows / 30) + 4)
    else:
        if len(mean_pue) > 5 and abs(mean_pue[-2] - mean_pue[-1]) <= 0.001 :
            fixed_depth = True

        else:
            max_tree_depth = int(data_rows / 30) + 4
    # model_list.append(tree.DecisionTreeRegressor(max_depth=int(data_rows / 30) + 4).fit(x, y_pue))

    model_list.append(tree.DecisionTreeRegressor(max_depth=max_tree_depth).fit(x, y_pue))

    for data_index in range(0, data_columns):
        model_list.append(tree.DecisionTreeRegressor(max_depth=max_tree_depth).fit(x, [row[data_index] for row in y_cold_temp]))

    return model_list


def model_predict(x_test):
    global model_list

    data_pred_list = []
    x_test = np.array(x_test)

    if x_test.ndim < 2:
        x_test = x_test[np.newaxis, :]

    for model in model_list:
        data_pred_list.append(model.predict(x_test)[0])

    return np.array(data_pred_list)


# 根据当前的状态获取下个时刻的空调的开关列表和设定的温度
def get_onoff_and_setting_temp(data):
    global setting_onoff_array, setting_temp_array, model_list, data_label_instance, candidate_array

    input = []
    cold_temp_array = data[2 * data_label_instance.ac_len: 3 * data_label_instance.ac_len].copy()

    pre_pue_list = []  # 预测的pue
    onoff_list = []    # 预测的空调开关状态
    temp_list = []     # 预测的设定温度状态
    pre_pue_and_cold_temp_list = []

    pre_mean_cold_temp_list_other = []  # 平均冷通道温度大于22℃时, 预测的平均冷通道温度
    onoff_list_other = []    # 平均冷通道温度大于22℃时, 预测的空调开关状态
    temp_list_other = []     # 平均冷通道温度大于设定温度时, 预测的设定温度状态
    pre_pue_and_cold_temp_list_other = []

    for onoff_candidate in candidate_array:

        current_onoff_array = setting_onoff_array.copy()

        if sum(current_onoff_array) + onoff_candidate >= max(Config.MIN_ONOFF_NUMS.value, round(data[-1] / 35 / 0.8) + 0.5) and \
                sum(current_onoff_array) + onoff_candidate <= data_label_instance.ac_len:

            cold_temp_array_copy = cold_temp_array.copy()
            if onoff_candidate == 1:
                for _ in range(len(cold_temp_array_copy)):
                    max_cold_temp_index = cold_temp_array_copy.index(max(cold_temp_array_copy))
                    if 0 == current_onoff_array[max_cold_temp_index]:
                        current_onoff_array[max_cold_temp_index] = 1
                        break
                    else:
                        cold_temp_array_copy[max_cold_temp_index] = 0.

            elif onoff_candidate == -1:
                for _ in range(len(cold_temp_array_copy)):
                    min_cold_temp_index = cold_temp_array_copy.index(min(cold_temp_array_copy))
                    if 1 == current_onoff_array[min_cold_temp_index]:
                        current_onoff_array[min_cold_temp_index] = 0
                        break
                    else:
                        cold_temp_array_copy[min_cold_temp_index] = float('inf')

            else:  # 此处 onoff_candidate == 0, 什么也不用做, 保持原样
                pass

        else:
            continue

#################################### 通过决策树预测 pue 和 冷通道温度 ########################################################################################
        for set_temp_candidate in candidate_array:
            current_setting_temp_array = setting_temp_array.copy()
            cold_temp_array_copy = cold_temp_array.copy()
            # 下面是 设定温度 +1, -1, 0
            if set_temp_candidate == 1:
                for _ in range(len(cold_temp_array_copy)):
                    min_cold_temp_index = cold_temp_array_copy.index(min(cold_temp_array_copy))
                    if current_setting_temp_array[min_cold_temp_index] < Config.MAX_COLD_TEMP.value:
                        current_setting_temp_array[min_cold_temp_index] += 1
                        break
                    else:
                        cold_temp_array_copy[min_cold_temp_index] = float('inf')

            elif set_temp_candidate == -1:
                for _ in range(len(cold_temp_array_copy)):
                    max_cold_temp_index = cold_temp_array_copy.index(max(cold_temp_array_copy))
                    if current_setting_temp_array[max_cold_temp_index] > Config.MIN_COLD_TEMP.value:
                        current_setting_temp_array[max_cold_temp_index] -= 1
                        break
                    else:
                        cold_temp_array_copy[max_cold_temp_index] = 0.

            else:  # 此处 set_temp_candidate == 0, 什么也不用做, 保持原样
                pass

            input.append(data[-5:].copy())  # 送风温度, 回风温度, 冷通道温度, 热通道温度, it负载
            input.append(current_onoff_array.copy())
            input.append(current_setting_temp_array.copy())
            input = sum(input, [])

            # 通过决策树预测 pue和冷通道温度
            pre_pue_and_cold_temp = model_predict(input)  # data格式：[pue, cold_temp_1, cold_temp_2, ...]

            pre_mean_cold_temp = sum([onoff_item * pre_cold_temp_item for onoff_item, pre_cold_temp_item \
                                      in zip(current_onoff_array, pre_pue_and_cold_temp[1:])]) / sum(current_onoff_array)

            if pre_mean_cold_temp < Config.PRE_MAX_MEAN_COLD_TEMP.value:  # 预测出来的平均冷通道温度 < 22℃

                pre_pue_list.append(pre_pue_and_cold_temp[0])
                onoff_list.append(current_onoff_array.copy())
                temp_list.append(current_setting_temp_array.copy())
                pre_pue_and_cold_temp_list.append(pre_pue_and_cold_temp.copy())

            else:  # 预测出来的平均冷通道温度 > 22℃
                pre_mean_cold_temp_list_other.append(pre_mean_cold_temp)
                onoff_list_other.append(current_onoff_array.copy())
                temp_list_other.append(current_setting_temp_array.copy())
                pre_pue_and_cold_temp_list_other.append(pre_pue_and_cold_temp.copy())

            input = []

    if len(pre_pue_list) > 0:
        min_pue_index = pre_pue_list.index(min(pre_pue_list))  # 获取最小的 pue 所在的索引值
        setting_onoff_array = onoff_list[min_pue_index].copy()
        setting_temp_array = temp_list[min_pue_index].copy()

    else:
        min_mean_cold_temp_index = pre_mean_cold_temp_list_other.index(min(pre_mean_cold_temp_list_other))  # 获取最小的 pue 所在的索引值
        setting_onoff_array = onoff_list_other[min_mean_cold_temp_index].copy()
        setting_temp_array = temp_list_other[min_mean_cold_temp_index].copy()
##############################################################################################################################################


def acg_control_predict(dEnvState):
    global x, y_pue, y_cold_temp, data_label_instance, count, flag, rack_areas, \
           model_input, model_list, drst_error, trajectory, pre_onoff_change, \
           pre_temp_change, setting_onoff_array, setting_temp_array, candidate_array, state_data, \
           it_power_main, y_pue_main, y_cold_temp_main, onoff_all, set_temp_all, fixed_depth

    dRst = {'Status': 0, 'PredictAction': {}}
    env_state_dict = get_data_dict(dEnvState)
    print(f'it_power: {env_state_dict["ItElectricPower"]}')

    state_data.append(env_state_dict)

    if 0 == count:
        data_label_instance = DataLabelStruct(dEnvState)
        print(data_label_instance)
        drst_error['PredictAction']['EquipIdArray'] = data_label_instance.ac_id_list
        drst_error['PredictAction']['PredOnoffArray'] = [1] * data_label_instance.ac_len
        drst_error['PredictAction']['PredTempSetArray'] = [21.] * data_label_instance.ac_len

    data = data_env_state_normal(env_state_dict, data_label_instance)


    it_power_main.append(data[-1])
    y_pue_main.append(env_state_dict['InstantPue'])
    y_cold_temp_main.append(data[2 * data_label_instance.ac_len: 3 * data_label_instance.ac_len].copy())


    # 开始时模型没有训练数据, 只能随机生成，数量为 Config.INIT_DATA_NUMS.value
    if count < Config.INIT_DATA_NUMS.value:
        print(f'初始的{Config.INIT_DATA_NUMS.value}数据')
        ac_on_nums = max(random.randint(3, 5), Config.MIN_ONOFF_NUMS.value, round(data[-1] / 35 / 0.8 + 0.5))  # 至少开 2 台空调
        cold_mean_temp = data[2 * data_label_instance.ac_len: 3 * data_label_instance.ac_len].copy()
        setting_onoff_array = [0] * data_label_instance.ac_len
        setting_temp_array = [random.randint(Config.MIN_SET_TEMP.value, Config.MAX_SET_TEMP.value)] * data_label_instance.ac_len

        for _ in range(ac_on_nums):
            max_cold_mean_index = cold_mean_temp.index(max(cold_mean_temp))
            setting_onoff_array[max_cold_mean_index] = 1
            cold_mean_temp[max_cold_mean_index] = 0.

        for item in data[-5:].copy():
            model_input.append(item)

        for onoff_item in setting_onoff_array:
            model_input.append(onoff_item)

        for setting_temp_item in setting_temp_array:
            model_input.append(setting_temp_item)

    elif 0 == count % 18:
        print('模型预测的数据')
        get_onoff_and_setting_temp(data)

        for item in data[-5:].copy():
            model_input.append(item)

        for onoff_item in setting_onoff_array:
            model_input.append(onoff_item)

        for setting_temp_item in setting_temp_array:
            model_input.append(setting_temp_item)

    else:  # 若发现某个区域温度超出阈值, 则立刻进行控制, 此处为 count % 18 >= 1 and count % 18 <= 17
        print('二次调整')
        cold_mean_temp = data[2 * data_label_instance.ac_len: 3 * data_label_instance.ac_len].copy()

        if (count % 18 >= 14 and count % 18 <= 17):
            tmp = []

            tmp.append(env_state_dict['InstantPue'])
            for cold_temp_item in cold_mean_temp:
                tmp.append(cold_temp_item)

            trajectory.append(tmp)

        # 平均冷通道温度大于目标温度, 温度设置太高, 找到机柜最热的区域, 降低温度, 开启空调
        if env_state_dict['ColdTempMean'] > Config.TARGET_TEMP.value:

            changed = False

            for _ in range(data_label_instance.ac_len):
                max_cold_mean_index = cold_mean_temp.index(max(cold_mean_temp))
                # if cold_mean_temp[max_cold_mean_index] > Config.MAX_COLD_TEMP.value and 1 == setting_onoff_array[max_cold_mean_index]:
                if 1 == setting_onoff_array[max_cold_mean_index]:
                    if setting_temp_array[max_cold_mean_index] > Config.MIN_COLD_TEMP.value:
                        setting_temp_array[max_cold_mean_index] -= 1
                        changed = True
                        break
                    else:
                        cold_mean_temp[max_cold_mean_index] = 0.
                else:
                    cold_mean_temp[max_cold_mean_index] = 0.

            if not changed and sum(setting_onoff_array) < data_label_instance.ac_len:
                cold_mean_temp = data[2 * data_label_instance.ac_len: 3 * data_label_instance.ac_len].copy()
                for _ in range(data_label_instance.ac_len):
                    max_cold_mean_index = cold_mean_temp.index(max(cold_mean_temp))
                    if  0 == setting_onoff_array[max_cold_mean_index]:
                        setting_onoff_array[max_cold_mean_index] = 1
                        break
                    else:
                        cold_mean_temp[max_cold_mean_index] = 0.

        # 平均冷通道温度小于冷通道温度下限, 温度设置太低, 找到机柜最冷的区域, 升高温度, 关闭空调
        elif env_state_dict['ColdTempMean'] < Config.MIN_COLD_TEMP.value:

            changed = False
            cold_mean_temp = data[2 * data_label_instance.ac_len: 3 * data_label_instance.ac_len].copy()

            for _ in range(data_label_instance.ac_len):
                min_cold_mean_index = cold_mean_temp.index(min(cold_mean_temp))
                # if cold_mean_temp[min_cold_mean_index] < Config.MIN_COLD_TEMP.value and 1 == setting_onoff_array[min_cold_mean_index]:
                if 1 == setting_onoff_array[min_cold_mean_index]:
                    if setting_temp_array[min_cold_mean_index] < Config.MAX_SET_TEMP.value:
                        setting_temp_array[min_cold_mean_index] += 1
                        changed = True
                        break
                    else:
                        cold_mean_temp[min_cold_mean_index] = float('inf')
                else:
                    cold_mean_temp[min_cold_mean_index] = float('inf')

            if not changed and sum(setting_onoff_array) > max(Config.MIN_ONOFF_NUMS.value, round(data[-1]/35/0.8 + 0.5)):
                cold_mean_temp = data[2 * data_label_instance.ac_len: 3 * data_label_instance.ac_len].copy()
                for _ in range(data_label_instance.ac_len):
                    min_cold_mean_index = cold_mean_temp.index(min(cold_mean_temp))
                    if  1 == setting_onoff_array[min_cold_mean_index]:
                        setting_onoff_array[min_cold_mean_index] = 0
                        break
                    else:
                        cold_mean_temp[min_cold_mean_index] = float('inf')


    dRst['PredictAction']['EquipIdArray'] = data_label_instance.ac_id_list
    dRst['PredictAction']['PredOnoffArray'] = setting_onoff_array
    dRst['PredictAction']['PredTempSetArray'] = setting_temp_array

    onoff_all.append(setting_onoff_array.copy())
    set_temp_all.append(setting_temp_array.copy())
    onoff_nums.append(sum(setting_onoff_array.copy()))

    # -------------------------------- 模型训练 ----------------------------------------------------

    if count < Config.INIT_DATA_NUMS.value:
        x.append(model_input.copy())
        y_pue.append(env_state_dict['InstantPue'])
        y_cold_temp.append(data[2 * data_label_instance.ac_len: 3 * data_label_instance.ac_len].copy())

        if count == Config.INIT_DATA_NUMS.value - 1:
            model_list = model_train()

        model_input = []

    elif count % 18 == 17:
        if len(trajectory) == 4:

            if fixed_depth:
                del x[0]
                del y_pue[0]
                del y_cold_temp[0]

            x.append(model_input.copy())
            y_pue.append(sum([item[0] for item in trajectory]) / len(trajectory))

            tmp = []
            for index in range(1, len(trajectory[0])):
                tmp.append(sum([item[index] for item in trajectory]) / len(trajectory))
            y_cold_temp.append(tmp.copy())

            model_list = model_train()
            if count % 288 == 287:
                dump_file()
                rmse()
                state_data = []

        trajectory = []
        model_input = []

    print(f'pue: {env_state_dict["InstantPue"]}')
    print(f"cold_temp_list: {data[2 * data_label_instance.ac_len: 3 * data_label_instance.ac_len]}")

    count += 1

    return dRst


def acg_control_init():
    global model_list, x, y_pue, y_cold_temp

    try:
        if os.path.exists(Config.MODEL_DIR.value) and os.path.isdir(Config.MODEL_DIR.value):
            if os.path.exists(f'{Config.MODEL_DIR.value}/model_pue.model'):
                model_list.append(load(f'{Config.MODEL_DIR.value}/model_pue.model'))

                cold_temp_model_nums = 0
                for _, _, files in os.walk(Config.MODEL_DIR.value):
                    for file_name in files:
                        if file_name.startswith('model_cold_temp_'):
                            cold_temp_model_nums += 1

                for index in range(cold_temp_model_nums):
                    model_list.append(load(f'{Config.MODEL_DIR.value}/model_cold_temp_{index}.model'))

        else:
            os.mkdir(Config.MODEL_DIR.value)

        if os.path.exists(Config.DATA_DIR.value) and os.path.isdir(Config.DATA_DIR.value):
            if os.path.exists(f'{Config.DATA_DIR.value}/x.csv'):
                x = pd.read_csv(f'{Config.DATA_DIR.value}/x.csv', header=None).values.tolist()
                y_pue = pd.read_csv(f'{Config.DATA_DIR.value}/y_pue.csv', header=None).values.tolist()
                y_cold_temp = pd.read_csv(f'{Config.DATA_DIR.value}/y_cold_temp.csv', header=None).values.tolist()

        else:
            os.mkdir(Config.DATA_DIR.value)

        return True

    except Exception:

        return False


# 删除所有的模型和数据
def acg_control_reset():

    try:
        if os.path.exists(Config.MODEL_DIR.value) and os.path.isdir(Config.MODEL_DIR.value):
                shutil.rmtree(Config.MODEL_DIR.value)

        if os.path.exists(Config.DATA_DIR.value) and os.path.isdir(Config.DATA_DIR.value):
                shutil.rmtree(Config.DATA_DIR.value)

        os.mkdir(Config.DATA_DIR.value)
        os.mkdir(Config.MODEL_DIR.value)

        return True

    except Exception as e:
        traceback.print_exc()
        return False

def main():
    global model_list, rack_areas
    for time in range(train_times):
        print(f'----------------------------------------{time}------------------------------------------------------')
        _, env_state = get_env_state()
        dRst = acg_control_predict(env_state)
        if (0 == dRst['Status']):
            set_onoff(dRst['PredictAction']['PredOnoffArray'])
            set_setting_temp(dRst['PredictAction']['PredTempSetArray'])
            print(
                f"onoff: {dRst['PredictAction']['PredOnoffArray']}, setting_temp: {dRst['PredictAction']['PredTempSetArray']}")
        else:
            print('设备出现故障')


if __name__ == '__main__':

    if acg_control_reset():
        print('重置完成')
    else:
        print('重置失败')

    if acg_control_init():
        print('初始化完成')
    else:
        print('初始化失败')

    main()

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    add = 0
    for y_pue_main_index in range(len(y_pue_main)):
        add += y_pue_main[y_pue_main_index]
        if y_pue_main_index != 0 and y_pue_main_index % 287 == 0:
            mean_pue.append(add / 288)
            add = 0

    plt.figure(figsize=(10, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt_rows = 5
    plt_cols = 2
    ax1 = plt.subplot(plt_rows, plt_cols, 1)
    ax2 = plt.subplot(plt_rows, plt_cols, 2)
    ax3 = plt.subplot(plt_rows, plt_cols, 3)
    ax4 = plt.subplot(plt_rows, plt_cols, 4)
    ax5 = plt.subplot(plt_rows, plt_cols, 5)
    ax6 = plt.subplot(plt_rows, plt_cols, 6)
    ax7 = plt.subplot(plt_rows, plt_cols, 7)
    ax8 = plt.subplot(plt_rows, plt_cols, 8)
    ax9 = plt.subplot(plt_rows, plt_cols, 9)
    ax10 = plt.subplot(plt_rows, plt_cols, 10)

    plt.sca(ax1)
    plt.title(f'dst每5分钟it负载变化曲线 {train_day}天')
    plt.ylabel('it_power')
    plt.plot(range(len(y_pue_main)), it_power_main, 'r')

    plt.sca(ax2)
    plt.title(f'每5分钟Pue值变化曲线 {train_day}天')
    plt.ylabel('pue')
    plt.plot(range(len(y_pue_main)), [1.3] * len(y_pue_main))
    plt.plot(range(len(y_pue_main)), y_pue_main, 'r')

    plt.sca(ax3)
    plt.title(f'每5分钟平均冷通道温度变化曲线 {train_day}天')
    plt.ylabel('mean_cold_temp')
    plt.plot(range(len(y_pue_main)), y_cold_temp_main)

    plt.plot(range(len(y_pue_main)), [25] * len(y_pue_main))
    plt.plot(range(len(y_pue_main)), [24] * len(y_pue_main))
    plt.plot(range(len(y_pue_main)), [20] * len(y_pue_main))

    plt.sca(ax4)
    plt.title(f'每5分钟空调设置温度变化曲线 {train_day}天')
    plt.ylabel('set_temp')
    for model_index in range(0, data_label_instance.ac_len):

        plt.plot(range(len(y_pue_main)), [row[model_index] + 0.01 * model_index for row in set_temp_all])

    plt.sca(ax5)
    plt.title(f'每5分钟空调开关变化曲线 {train_day}天')
    plt.ylabel('onoff')
    for model_index in range(0, data_label_instance.ac_len):
        plt.plot(range(len(y_pue_main)), [row[model_index] + 0.01 * model_index for row in onoff_all])

    plt.sca(ax6)
    plt.title(f'每5分钟开关数量变化曲线 {train_day}天')
    plt.ylabel('onoff_nums')
    plt.plot(range(len(y_pue_main)), onoff_nums)

    plt.sca(ax7)
    plt.title(f'每天平均Pue {train_day}天')
    plt.plot(range(len(mean_pue)), mean_pue)

    plt.sca(ax8)
    plt.title(f'每天 pue 均方误差 rmse_pue {train_day}天')
    plt.plot(range(len(pue_error)), pue_error)

    plt.sca(ax9)
    plt.title(f'每天各区域冷通道温度均方误差 rmse_cold_temp {train_day}天')
    for model_index in range(0, data_label_instance.ac_len):
        plt.plot(range(len(y_cold_temp_square_error)), [row[model_index] + 0.01 * model_index for row in y_cold_temp_square_error])

    if len(y_pue_main) > 1440:
        plt.sca(ax10)
        plt.title('最后5天的空调开关变化曲线, 从第5副图截取的部分')
        plt.ylabel('onoff')
        for model_index in range(0, data_label_instance.ac_len):
            plt.plot(range(1440), [row[model_index] + 0.01 * model_index for row in onoff_all[-1440:]])

        plt.plot([288 - 1] * 6, [-0.2, 0.2, 0.4, 0.6, 0.8, 1.2])
        plt.plot([288*2 - 1] * 6, [-0.2, 0.2, 0.4, 0.6, 0.8, 1.2])
        plt.plot([288*3 - 1] * 6, [-0.2, 0.2, 0.4, 0.6, 0.8, 1.2])
        plt.plot([288*4 - 1] * 6, [-0.2, 0.2, 0.4, 0.6, 0.8, 1.2])
        plt.plot([288*5 - 1] * 6, [-0.2, 0.2, 0.4, 0.6, 0.8, 1.2])

    plt.show()

    print(f'pue: {sum(y_pue_main) / len(y_pue_main)}')