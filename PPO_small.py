import tensorflow.compat.v1 as tf
import numpy as np
import os
import math
import shutil
import heapq
import logging
from ppo_agent import PPO
from nn_class import Graph_2
from interface_diy import get_pos
from interface_diy import get_new_label
from interface_diy import get_new_dEnvState
from interface_diy import devide_area
from interface_diy import get_ac_temp_label
from embedding import get_state_input_3 # 获取环境状态函数
from embedding import vec_embedding_5 # 编码函数
from embedding import vec_embedding_1 # 编码函数
from embedding import vec_embedding_2 # 编码函数
from embedding import vec_embedding_6 # 编码函数
from embedding import discount_rewards
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
# from virtual_env_generator_v1 import get_env_state
# from virtual_env_generator_v1 import set_onoff
# from virtual_env_generator_v1 import set_setting_temp
from virtual_env_generator_v1 import get_env_state
from virtual_env_generator_v1 import set_onoff
from virtual_env_generator_v1 import set_setting_temp
from matplotlib import pyplot as plt
import xgboost as xgb
import joblib
from sklearn.multioutput import MultiOutputRegressor


######################################### 用户自定义内容 ###############################################
# max_sample = 7
# explore_day = 2loss0
explore_day = 100
train_data = 0.5
Max_ac_num = 6 # 最大空调台数
Min_ac_num = 2 # 最低空调台数
Max_temp = 22 # 空调温度设置上限
Min_temp = 18 # 空调温度设置下限
Ac_num_lower = 2 # 最低空调开启台数
ALPHA = 0.0  # 冷通道温度超过设定的TARGET_TEMP损失参数
TARGET_TEMP = 22
violation = 3
epsilon = 1
Temp_upper = 25
collect_smaple_date = 0.5
encode_len = 4
#pue_base_collect = 2
#####################################################################################################

train_PPO_num = 0
######################################## 神经网络相关定义 #############################################
AC_CHOOSE = [0,-1,1] # 最低开启两台空调
TEMP_CHOOSE = [t for t in range(Min_temp,Max_temp + 1)] # 空调温度设置网络输出
# Input_dim_1 = encode_len * 5 # the dimension of the input of the graph1
Input_dim_1 = int(encode_len) * 2 + encode_len * (4) + encode_len
#Input_dim_1 = 88
Input_dim_2 = Input_dim_1 + Max_ac_num   # the dimension of the input of the graph1
Num_hidden_1 = 2  # the number of dimension of graph1
Num_hidden_2 = 4  # the number of dimension of graph2
Out_dim_1 = Max_ac_num - 1  # the number of output of graph1
Out_dim_2 = len(TEMP_CHOOSE)  # the number of output of graph2
####################################################################################################

######################################### 可调整的参数 ###############################################
# Lr_1 = 0.16 # the step_size(learning rate) of graph1
# Lr_1 = 0.3 # the step_size(learning rate) of graph1
Lr_1 = 0.3 # the step_size(learning rate) of graph1
Lr_2 = 0.03  # the step_size(learning rate) of graph2
Rew_discount_1 = 0.2 # 奖励折扣
# Rew_discount_1 = 1
Rew_discount_2 = 0.08
Max_train_step = 200 # 暂定最大训练伦次
Len_epi = 1 # episode的长度，为1时表明不使用累积奖励
Num_epi = 1 # 一次迭代中，episode的个数
Min_lrmult = 0.1
# Min_lrmult = 1
Gamma = 0.9 # 累计及奖励的折扣值
Baselinefactor = 1.0 # 获取pue时的基线
#Baselinefactor = 1.02 # 获取pue时的基线
####################################################################################################

weight_temp=0.5
temp_epsilon=4
pue_fraction = 0.25
# label_1 = Max_ac_num - Ac_num_lower
######################################### 全局变量 ##################################################
TIMES = 0 # 统计采样次数
PUE_LIST = [] # 统计此次空调设置的pue
AC_LIST = [1] * Max_ac_num # 默认状态下的空调设置
AC_TEMP = [21] * Max_ac_num # 初始空调配置
STATE_DATA = []  # 记录多个环境状态，减少环境噪声
STATE_INPUT = [] # 记录输入，待神经网络决策后，传入相应列表存储
LOSS_AC_NUM = 0 # 开启空调台数决策超标时损失
TRAINED = False # 根据是否训练并给出决策
EP_OBS_1=[] # 存储episode状态
EP_ACT_1=[] # 存储graph1的episode动作
EP_REW_1=[] # 存储累积的奖励
DATA_X = []  # 数据收集
DATA_Y = []  #
COLD_TEMP_MEAN = []
ALL_PUE = []

# previous_action = Ac_num_lower
old_ac_penalty = 0
old_label=-1
####################################################################################################

########################################## 特定功能函数 ##############################################
def get_rack_cold_temp(jEnvState,Area): # 得到区域平均冷通道温度
    AreaColdTemp = []
    for i in range(len(Area)):
        cold_temp = []
        for j in range(len(Area[i][1])):
            if Area[i][1][j] in jEnvState:
                cold_temp.append(jEnvState[Area[i][1][j]])
        if (len(cold_temp) == 0):
            return False
        AreaColdTemp.append(sum(cold_temp)/len(cold_temp))
    return (AreaColdTemp)

def safe_control(jEnvState,ac_list,ac_temp,Area,DRST): # 进行温度控制
    AC_cold_temp = get_rack_cold_temp(jEnvState,Area)
    if type(AC_cold_temp) != list:
        logging.error("Get rack cold temp encounter error 2.")
        return DRST
    ac_l = ac_list.copy()
    ac_t = ac_temp.copy()
    temp_changed = 0 # 临时记录是否调整了温度
    for j in range(len(AC_cold_temp)):
        pos = AC_cold_temp.index(max(AC_cold_temp))
        l = Area[pos][0][0]  # 得到对应空调的编号
        if ac_t[l] > 20 and ac_l[l] == 1:
            ac_t[l] -= 1
            temp_changed = 1
            break
        else:
            AC_cold_temp[pos] = 0
    if temp_changed == 0: # 若没有降温，表明需要打开关闭的空调
        AC_cold_temp = get_rack_cold_temp(jEnvState,Area)
        for k in range(len(AC_cold_temp)):
            pos = AC_cold_temp.index(max(AC_cold_temp))
            l = Area[pos][0][0]  # 得到对应空调的编号
            if ac_l[l] == 0:
                ac_l[l] = 1
                break
            else:
                AC_cold_temp[pos] = 0
    return ac_l,ac_t

def get_ac_list(ac_action,ac_l,Area,AC_cold_temp,Ac_num): # 根据空调开关神经网络输出结果，自动得到相应的空调设置
    global LOSS_AC_NUM
    if ac_action > Ac_num :
        ac_action = Ac_num
        LOSS_AC_NUM = 0.8
    a = ac_action - sum(ac_l)
    print("a:", a)
    if a > 0:
        for a_i in range(a):
            for i in range(len(AC_cold_temp)):
                pos_max_cold = AC_cold_temp.index(max(AC_cold_temp))
                if ac_l[Area[pos_max_cold][0][0]] == 0:
                    ac_l[Area[pos_max_cold][0][0]] = 1
                    break
                else:
                    AC_cold_temp[pos_max_cold] = 0
    elif a < 0:
        for a_i in range(abs(a)):
            for i in range(len(AC_cold_temp)):
                pos_min_cold = AC_cold_temp.index(min(AC_cold_temp))
                if ac_l[Area[pos_min_cold][0][0]] == 1:
                    ac_l[Area[pos_min_cold][0][0]] = 0
                    break
                else:
                    AC_cold_temp[pos_min_cold] = 50
    else:
        pass
    return ac_l
def get_train_data(input_2, Max_temp, Min_temp):
    train_state = []
    train_temp = []
    temp_choice = []
    for i in range(0, Max_temp - Min_temp + 1):
        train_state.append(input_2.copy())
        #for j in range(0, Max_temp - Min_temp + 1):
        temp_choice.append(Min_temp + i)
        train_temp.append(temp_choice.copy())
        temp_choice = []
    train_temp = np.array(train_temp)
    train_state = np.array(train_state)
    train_x = np.concatenate((train_state,train_temp),axis = 1)
    #print(train_state)
    #print(train_temp)
    #print(train_x)
    return train_x

def get_decisiontree_input(jEnvState,AC,Area,Ac_num,ac_list):  # 热通道温度减去冷通道温度，结合空调设置温度
    temp_state = [jEnvState['ItElectricPower']]
    deliver_temp = []
    return_temp = []
    for h in range(Ac_num):
        if AC[0][h] in jEnvState:
            if (ac_list[h] == 1):
                deliver_temp.append(jEnvState[AC[0][h]])
        else:
            return False  # 缺少任意空调送回风温度，报错
        if (AC[1][h] in jEnvState):
            if (ac_list[h] == 1) :
                return_temp.append(jEnvState[AC[1][h]])
        else:
            return False  # 缺少任意空调送回风温度，报错
    # 六个区域的平均冷通道温度
    temp_state.append(np.mean(deliver_temp))
    temp_state.append(np.mean(return_temp))
    #AreaColdTemp = []
    #AreaHotTemp = []
    cold_temp = []
    hot_temp = []
    for i in range(len(Area)):
        for j in range(len(Area[i][1])):
            if Area[i][1][j] in jEnvState:
                cold_temp.append(jEnvState[Area[i][1][j]])
                hot_temp.append(jEnvState[Area[i][2][j]])
        #if (len(cold_temp) == 0) or (len(hot_temp) == 0):
            #return False
    temp_state.append(sum(cold_temp) / len(cold_temp))
    temp_state.append(sum(hot_temp) / len(hot_temp))
    #if (len(AreaColdTemp) != len(Area)) or (len(AreaHotTemp) != len(Area)):  # 整块区域温度缺失，报错
        #return False
    # 将冷通道温度和热通道温度添加到列表temp_state
    #print(len(temp_state))
    return temp_state

def temp_decision(y_pre,target_temp):
    pre = np.reshape(y_pre,(-1))
    #print("*****************",y_pre)
    pre_list = pre
    dis_target_temp = []
    for i in range(len(pre_list)):
        dis_target_temp.append(abs(pre_list[i] - target_temp))
    min_pos = dis_target_temp.index(min(dis_target_temp))
    # a = min(dis_target_temp)
    # min_pos = np.max(dis_target_temp.index(a))
    return min_pos


def acg_control_init():
    """
    加载现有模型，用于突发故障等意外因素，
    加载当前状态最新的模型，若此时没有训练的模型，
    则需要用 acg_control_reset函数进行初始化
    """
    try:
        sModelPath1 = os.path.join(os.path.dirname(__file__), 'model_pg/graph_1_ckpt/')
        if os.path.exists(sModelPath1):
            G1.saver_1.restore(G1.sess1, tf.train.latest_checkpoint(sModelPath1))
        else:
            G1.saver_1.restore(G1.sess1, os.path.join(os.path.dirname(__file__), \
                                                'model_initial/graph_1_ckpt_initial/model.ckpt-4'))
        return True
    except:
        return False

def acg_control_predict(dEnvState,G1):
    global TIMES,PUE_LIST,AC_LIST,AC_TEMP,STATE_DATA,STATE_INPUT,TRAINED,ALL_PUE
    global EP_OBS_1, EP_ACT_1,EP_REW_1,LOSS_AC_NUM,DATA_X,COLD_TEMP_MEAN,BASELINE,previous_action,ac_action,old_ac_penalty,old_label
    global train_PPO_num, old_input, old_action, old_reward,old_pue_av,old_elect
    Ac_num, Rc_num, Pos_device,Ac_name_list = get_pos(dEnvState) # 获得设备信息
    New_pos = get_new_label(Pos_device) # 生成位置对应
    New_EnvState = get_new_dEnvState(dEnvState, Pos_device, New_pos)  # 得到新的EnvState
    Area = devide_area(New_pos) # 对机柜和空调进行分区
    # print('Area:',Area)
    if len(AC_LIST) > Ac_num:
        for i in range(Ac_num,len(AC_LIST)):
            AC_LIST[i] = 0
    DRST = {'Status': 1, 'PredictAction': {'EquipIdArray': Ac_name_list, "PredOnoffArray": [1] * Ac_num,
                                           'PredTempSetArray': [21.0] * Ac_num}} # 设备故障时的保障决策
    AC = get_ac_temp_label(New_EnvState)  # 得到空调的送回风温度标签，用于读取对应的数据
    dRst = {'Status': 0, 'PredictAction': {}}
    dRst['PredictAction']['EquipIdArray'] = Ac_name_list # 空调标号列表，亦可由代码自动生成，再次先手动填写
    dRst['PredictAction']["PredOnoffArray"] = [int(AC_LIST[each]) for each in range(Ac_num)]
    dRst['PredictAction']['PredTempSetArray'] = [float(AC_TEMP[each]) for each in range(Ac_num)]
    jEnvState = get_new_dEnvState(dEnvState, Pos_device, New_pos) # 通过两个字典搭建桥梁
    if type(jEnvState) != dict:
        logging.error("Translate data encounter error.")
        return DRST
    STATE_DATA.append(jEnvState.copy())
    TRAINED = False #将Trained指示变量置为False
    TIMES += 1  # 统计采样次数，便于在相应次数进行调整空调设置
    if (TIMES % 18) >= 15 or (TIMES % 18) == 0:
        #print('#####',len(STATE_INPUT))
        PUE_LIST.append(jEnvState['InstantPue'])  # 存储当前pue
        COLD_TEMP_MEAN.append(jEnvState['ColdTempMean'])
        state_temp = get_state_input_3(jEnvState, AC, Area, Ac_num, AC_LIST)
        if type(state_temp) == list:
            STATE_INPUT.append(state_temp.copy())
        else:
            return DRST
        #print(len(STATE_INPUT))
        if (len(STATE_INPUT) < 4) and (TIMES % 18 == 0): # 传感器失灵，此轮不训练，则将相应变量清空
            STATE_INPUT, PUE_LIST, COLD_TEMP_MEAN= [], [], []
            LOSS_AC_NUM = 0 # 归零
            EP_OBS_1.pop() # 移除最后一个存储的observation
            EP_ACT_1.pop() # 移除最后一个存储的action
            return DRST
        elif len(STATE_INPUT) == 4:
            #print(len(STATE_INPUT))
            state_mean = np.mean(STATE_INPUT, axis=0).tolist()
            inputs = state_mean
            if TIMES > 288 * collect_smaple_date:
                train_PPO_num += 1
                # input_1 = vec_embedding_1(inputs, old_elect, encode_len)
                input_1 = vec_embedding_2(inputs, ac_action, encode_len)
                # input_1 = vec_embedding_5(inputs, encode_len)
                # print(np.array(input_1[0]))
                # ac_select = G1.actor(input_1)
                input_1 = input_1.astype(np.float32)
                # print((input_1))
                ac_select = G1.get_action([input_1])
                cold_temp_mean = np.mean(COLD_TEMP_MEAN)
                print(ac_select)
                if cold_temp_mean <= TARGET_TEMP - violation:
                    # print(ac_select[0])
                    # print(Max_ac_num/2 + 1-Ac_num_lower)
                    ac_select[0][int(Max_ac_num/2)-Ac_num_lower:] = 0
                    if sum(ac_select[0]) > 0:
                        ac_select = ac_select/sum(ac_select[0])
                    else:
                        ac_select[0][:int(Max_ac_num/2)-Ac_num_lower] = [1/(int(Max_ac_num/2)-Ac_num_lower)
                                                                           for i in range(0,int(Max_ac_num/2)-Ac_num_lower)]
                        # print(ac_select[0])
                # print(ac_select)
                # if TIMES <= 288 * explore_day:
                ac_action_choose = np.random.choice(AC_CHOOSE, p=ac_select.ravel())
                # print("afdsddsdf",ac_action)
                label_1 = AC_CHOOSE.index(ac_action_choose)
                print(label_1)
                # ac_action_temp = ac_action
                # print("the action is", ac_action_choose)
                ac_action = ac_action + ac_action_choose
                # print("the action is", ac_action_choose, "##previous action is", ac_action_temp, "##current action is",
                #       ac_action)
                if ac_action > Max_ac_num:
                    ac_action = Max_ac_num
                    ac_penalty = 0.8
                elif ac_action < Min_ac_num:
                    ac_action = Min_ac_num
                    ac_penalty = 0.8
                else:
                    ac_penalty = 0
                AC_cold_temp = get_rack_cold_temp(jEnvState,Area)
                ac_l = AC_LIST.copy()
                AC_LIST = get_ac_list(ac_action,ac_l,Area,AC_cold_temp,Ac_num)  # 得到空调开关设置
                print('第 %d 次:'%TIMES)
                print("ac_action:",ac_action)
                print(AC_LIST)
                dRst['PredictAction']["PredOnoffArray"] = [int(AC_LIST[each]) for each in range(Ac_num)]
                #dRst["PredictAction"]["PredOnoffArray"] = [int(each) for each in AC_LIST]  # 准备输出
                # set_onoff(ac_list)
                ac_array = np.array(sum(AC_LIST))
                ac_array = np.reshape(ac_array, (1, 1))
                # 得到第二个模型的输入
                state = np.array(state_mean)
                state = np.reshape(state,(1,-1))
                input_2 = np.concatenate((state, ac_array), axis=1)
                input_2 = np.reshape(input_2,(-1))
                train_x = get_train_data(input_2,Max_temp,Min_temp)
                #print(train_x)
                print(np.mean(COLD_TEMP_MEAN))
                DATA_Y.append(cold_temp_mean)
                Train_y = np.array(DATA_Y)
                Train_y = np.reshape(Train_y, (-1, 1))
                Train_x = np.array(DATA_X)
                other_params = {'max_depth': 5, 'learning_rate': 0.2,'silent': False}
                # multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:gamma', **other_params)).fit(Train_x,
                # Train_y)
                multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(**other_params)).fit(Train_x,
                                                                                 Train_y)
                joblib.dump(multioutputregressor, filename='./model/decision_tree.pkl')

                model = joblib.load('./model/decision_tree.pkl')
                pre_y = model.predict(train_x)
                print("pre_y:",pre_y)
                y_temp = temp_decision(pre_y, TARGET_TEMP)
                print("##############:",pre_y[y_temp])
                x_selected = train_x[y_temp] # 代表此次被选中的空调开关和温度设置
                temp = Min_temp + y_temp
                AC_TEMP = [temp] * (Max_ac_num)  # 得到空调温度设置
                #print('#####################################################################')
                print("temp:", temp)
                print(AC_TEMP)
                dRst['PredictAction']['PredTempSetArray'] = [float(AC_TEMP[each]) for each in range(Ac_num)]
                # dRst['PredictAction']['PredTempSetArray'] = [float(each) for each in AC_TEMP] # 输入到输出变量中
                pue_av = np.mean(PUE_LIST)  # 获取pue均值
                ALL_PUE.append(np.mean(pue_av))
                STATE_INPUT, PUE_LIST, COLD_TEMP_MEAN = [], [], []  # 清空相应变量
                temp_penalty = weight_temp*math.log(1+math.exp(abs(TARGET_TEMP - cold_temp_mean)))
                rew_1 = 2 - (pue_av + LOSS_AC_NUM + old_ac_penalty)
                # rew_1 = old_pue_av - pue_av
                if cold_temp_mean <= TARGET_TEMP - temp_epsilon:
                    rew_1 -= temp_penalty # reward
                print('pue:', pue_av)
                LOSS_AC_NUM = 0

                if train_PPO_num > 1:
                    G1.perceive(old_input, old_action, rew_1, input_1)
                old_input = input_1.copy()
                old_action = label_1
                old_ac_penalty = ac_penalty
                old_pue_av = pue_av
                old_elect = inputs[0]
                # old_reward = rew_1
                DATA_X.append(x_selected.copy())
                TRAINED = True # 表示已决策
            elif TIMES <= 288 * collect_smaple_date:
                ALL_PUE.append(np.mean(PUE_LIST))
                ac_action = np.random.choice([2,3,4,5])
                AC_cold_temp = get_rack_cold_temp(jEnvState, Area)
                ac_l = AC_LIST.copy()
                AC_LIST = get_ac_list(ac_action, ac_l, Area, AC_cold_temp, Ac_num)  # 得到空调开关设置
                print('第 %d 次:' % TIMES)
                print("ac_action:", ac_action)
                print(AC_LIST)
                dRst['PredictAction']["PredOnoffArray"] = [int(AC_LIST[each]) for each in range(Ac_num)]
                # dRst["PredictAction"]["PredOnoffArray"] = [int(each) for each in AC_LIST]  # 准备输出
                # set_onoff(ac_list)
                ac_array = np.array(sum(AC_LIST))
                ac_array = np.reshape(ac_array, (1, 1))
                # 得到第二个模型的输入
                state = np.array(state_mean)
                state = np.reshape(state, (1, -1))
                input_2 = np.concatenate((state, ac_array), axis=1)
                #input_2 = np.reshape(input_2, (-1))
                #train_x = get_train_data(input_2, Max_temp, Min_temp)
                #input_2 = np.concatenate((input_1, ac_array), axis=1)
                #input_2 = np.reshape(input_2, (-1))
                #train_x = get_train_data(input_2, Max_temp, Min_temp)
                ac_temp = 20
                if ac_action == 2:
                    ac_temp = np.random.choice([17,18,19])
                elif ac_action == 3:
                    ac_temp = np.random.choice([18,19,20,21])
                elif ac_action == 4:
                    ac_temp = np.random.choice([19,20,21,22])
                elif ac_action == 5:
                #     # ac_temp = np.random.choice([18,19,20,21,22])
                    ac_temp = np.random.choice([21,22])
                temp_list = []
                """
                for i in range(0, Max_temp - Min_temp + 1):
                    if i == (ac_temp - 1):
                        temp_list.append(1)
                    else:
                        temp_list.append(0)
                temp_list = np.array(temp_list)
                temp_list = np.reshape(temp_list,(1,-1))
                """
                #for i in range(0, Max_temp - Min_temp + 1):
                temp_list.append(ac_temp)
                temp_list = np.array(temp_list)
                temp_list = np.reshape(temp_list, (1, -1))
                x_train = np.concatenate((input_2,temp_list),axis=1)
                x_train = np.reshape(x_train,(-1))
                AC_TEMP = [ac_temp] * (Max_ac_num)  # 得到空调温度设置
                print("temp:", ac_temp)
                print(AC_TEMP)
                dRst['PredictAction']['PredTempSetArray'] = [float(AC_TEMP[each]) for each in range(Ac_num)]
                cold_temp_mean = np.mean(COLD_TEMP_MEAN)

                if len(DATA_X)>0:
                    DATA_Y.append(cold_temp_mean)
                LOSS_AC_NUM = 0
                old_pue_av = np.mean(PUE_LIST)
                old_elect = inputs[0]
                DATA_X.append(x_train.copy())
                STATE_INPUT, PUE_LIST, COLD_TEMP_MEAN = [], [], []  # 清空相应变量
                #print('len(DATA_X):', len(DATA_X))
                #print('len(DATA_Y):', DATA_Y)
                #print('###############################################')
    if (jEnvState['ColdTempMean'] > Temp_upper) and (TRAINED == False):  # 若总体冷通道平均温度大于 target_temp
        AC_LIST, AC_TEMP = safe_control(jEnvState, AC_LIST, AC_TEMP, Area,DRST)
        dRst['PredictAction']["PredOnoffArray"] = [int(AC_LIST[each]) for each in range(Ac_num)]
        dRst['PredictAction']['PredTempSetArray'] = [float(AC_TEMP[each]) for each in range(Ac_num)]
    #print(TIMES)
    #print(DATA_X)
    #print(DATA_Y)
    return (dRst,DATA_X,DATA_Y)




if "__main__" == __name__:
     G1 = PPO(Input_dim_1, Num_hidden_1, len(AC_CHOOSE))  # generate graph of setting_onoff
     times = 288 * 20
     collect_smaple = 1
     ratio_temp = []
     control_trace = []
     PUE_LIST_FULL = []
     cold_temp = []
     ac = []
     X = []
     Y = []
     for i in range(times):
         jEnvState, jEnvState2 = get_env_state()
         drst, X, Y = acg_control_predict(jEnvState2,G1)
         PUE_LIST_FULL.append(jEnvState['InstantPue'])
         set_onoff(drst['PredictAction']['PredOnoffArray'])
         set_setting_temp(drst['PredictAction']['PredTempSetArray'])
         # print("ItElectricPower:",jEnvState['ItElectricPower'])
         cold_temp.append(jEnvState['ColdTempMean'])
     for j in range(collect_smaple, len(X)):
         control_trace.append((X[j])[5:7])
     print("The action is:", control_trace)
     print(Y)
     # print("PUE list is:", PUE_LIST_FULL)
     t = 0
     for j in range(len(cold_temp)):
         if cold_temp[j] > 25:
             t += 1
     pue_average = []
     ave_day = []
     for i in range(int(len(PUE_LIST_FULL) / 288)):
         ave_day.append(np.mean(PUE_LIST_FULL[i * 288:(i + 1) * 288]))
     print("高于25度的次数：", t)
     print("高于25度的比例：", t / (times))

     ratio_temp.append(t / (times))
     # a = acg_control_reset()
     for i in range(len(ave_day)):
         print('第', i + 1, '天平均pue为：', ave_day[i])
     z = range(len(ave_day))
     plt.title("Matplotlib demo")
     plt.xlabel("Times")
     plt.ylabel("InstantPue")
     plt.ylim(1.25, 1.35)
     plt.plot(z, ave_day)
     # plt.plot(x,y)
     plt.plot()
     plt.show()
     print(ratio_temp)
