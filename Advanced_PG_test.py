#coding:utf-8
# from virtual_env_generator_v1_2 import get_env_state
# from virtual_env_generator_v1_2 import set_onoff
# from virtual_env_generator_v1_2 import set_setting_temp
from virtual_env_generator_v1 import get_env_state
from virtual_env_generator_v1 import set_onoff
from virtual_env_generator_v1 import set_setting_temp
# from virtual_env_simulation import get_env_state
# from virtual_env_simulation import set_onoff
# from virtual_env_simulation import set_setting_temp
#from PG_method_train_second_mixed import acg_control_init
from Advanced_PG import acg_control_predict
# from PG_method_train_second_mixed import acg_control_reset
from Advanced_PG import acg_control_initial
import numpy as np
from matplotlib import pyplot as plt
#a=acg_control_reset()
#print (a)
times = 288 * 30
collect_smaple = 1
ratio_temp = []
# control_trace=[]

for k in range(1):
    # G1 = acg_control_reset()
    G1 = acg_control_initial()
    control_trace = []
    PUE_LIST = []
    cold_temp = []
    ac = []
    X = []
    Y = []
    for i in range(times):
        jEnvState, jEnvState2 = get_env_state()
        # print('第', i + 1, '次采样：')
        drst, X, Y = acg_control_predict(G1, jEnvState2)
        PUE_LIST.append(jEnvState['InstantPue'])
        # print(pue)
        # print(drst['PredictAction']['PredOnoffArray'])
        # print(drst['PredictAction']['PredTempSetArray'])
        set_onoff(drst['PredictAction']['PredOnoffArray'])
        set_setting_temp(drst['PredictAction']['PredTempSetArray'])
        # print("ItElectricPower:",jEnvState['ItElectricPower'])
        cold_temp.append(jEnvState['ColdTempMean'])
    for j in range(collect_smaple, len(X)):
        control_trace.append((X[j])[5:7])
    print("The action is:",  control_trace)
    # print(Y)
    t = 0
    for j in range(len(cold_temp)):
        if cold_temp[j] > 25:
            t += 1
    pue_average = []
    ave_day = []
    for i in range(int(len(PUE_LIST) / 288)):
        ave_day.append(np.mean(PUE_LIST[i * 288:(i + 1) * 288]))
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