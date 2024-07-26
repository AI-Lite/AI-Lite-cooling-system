#coding:utf-8
import re
import math

def get_pos(dEnvState):
    label = dEnvState['ParamLabel'].copy()
    data = dEnvState['ParamData'].copy()
    #print(label)
    ac_num=0 # 统计空调数目
    rc_num=0 # 统计机柜数目
    pos_device = {}
    ac_name_list = []
    for i in range(len(label)):  # 根据得到的数据得到各设备的位置信息，便于之后的数据转换
        if label[i][0:6] == 'AcXPos':   # 找到空调的坐标
            ac_name_list.append(re.findall(r"\d+\.?\d*", label[i])[0])
            ac_num+=1
            for j in range(len(label)):
                if (label[j][0:6] == 'AcYPos') and (
                        re.findall(r"\d+\.?\d*", label[j])[0] == re.findall(r"\d+\.?\d*", label[i])[0]):
                    #print(dEnvState['ParamLabel'][i], dEnvState['ParamData'][i])
                    #print(dEnvState['ParamLabel'][j], dEnvState['ParamData'][j])
                    pos_device[label[i][0:2] + re.findall(r"\d+\.?\d*", label[i])[0]] = str(data[i]) + '_' + str(data[j])
        elif label[i][0:8] == 'RackXPos': # 找到机柜的坐标
            rc_num+=1
            for k in range(len(label)):
                if (label[k][0:8] == 'RackYPos') and (
                        re.findall(r"\d+\.?\d*", label[k])[0] == re.findall(r"\d+\.?\d*", label[i])[0]):
                    #print(dEnvState['ParamLabel'][i], dEnvState['ParamData'][i])
                    #print(dEnvState['ParamLabel'][k], dEnvState['ParamData'][k])
                    pos_device[label[i][0:4] + re.findall(r"\d+\.?\d*", label[i])[0]] = str(data[i]) + '_' + str(data[k])
    return ac_num,rc_num,pos_device,ac_name_list
def get_new_label(pos_device): # 生成形式的空调机柜编号
    new_= {}
    ac = 0
    rc = 0
    for label in pos_device.keys():
        if label[0:2] == 'Ac':
            new_[pos_device[label]] = 'Ac' + str(ac)
            ac += 1
        else:
            new_[pos_device[label]] = 'Rack' + str(rc)
            rc += 1
    #print("ac:",ac)
    #print("rc:",rc)
    return new_

def get_new_dEnvState(dEnvState,pos_device,new_):
    label = dEnvState['ParamLabel'].copy()
    data = dEnvState['ParamData'].copy()
    new_EnvState = {}
    for i in range(len(label)):
        if label[i][0:2] == 'Ac':
            key_tmp = 'Ac' + re.findall(r"\d+\.?\d*", label[i])[0]  # 得到实际的空调编号 Ac*
            target_num = new_[pos_device[key_tmp]][2:] # 得到虚拟的空调编号Ac*
            target_key = re.sub(r'[^A-Za-z]', '', label[i]) + '_' + target_num
            new_EnvState[target_key] = data[i]
        elif label[i][0:4] == 'Rack':
            key_tmp = 'Rack' + re.findall(r"\d+\.?\d*", label[i])[0] # 得到实际的机柜编号 Rack*
            target_num = new_[pos_device[key_tmp]][4:] # 得到虚拟的机柜编号 Rack*
            target_key = re.sub(r'[^A-Za-z]', '', label[i]) + '_' + target_num
            new_EnvState[target_key] = data[i]
        else:
            new_EnvState[label[i]] = data[i]
    if 'ItElectricPower' not in new_EnvState: # 负载丢失，报错
        return False
    return new_EnvState
def trans_d_to_j(dEnvState):  # 转换dEnvState格式
    jEnvState={}
    for i in range(len(dEnvState['ParamLabel'])):
        jEnvState[dEnvState['ParamLabel'][i]]=dEnvState['ParamData'][i]
    if 'ItElectricPower' not in jEnvState: # 负载丢失，报错
        return False
    load=jEnvState['ItElectricPower']
    # 用于得到负载Load
    jEnvState['ItElectricPower']=load
    #print(jEnvState)
    return (jEnvState)
"""
    示例：
    Area=[[[3],['RackColdTemp_22','RackColdTemp_23','RackColdTemp_24','RackColdTemp_25'],
        ['RackHotTemp_22','RackHotTemp_23','RackHotTemp_24','RackHotTemp_25']],
        [[2],['RackColdTemp_27','RackColdTemp_29','RackColdTemp_30','RackColdTemp_31'],
        ['RackHotTemp_27','RackHotTemp_29','RackHotTemp_30','RackHotTemp_31']],
        [[4],['RackColdTemp_34','RackColdTemp_35'],['RackHotTemp_34','RackHotTemp_35']],
        [[3],['RackColdTemp_3','RackColdTemp_4'],['RackHotTemp_3','RackHotTemp_4']],
        [[1],['RackColdTemp_7','RackColdTemp_8','RackColdTemp_9'],
        ['RackHotTemp_7','RackHotTemp_8','RackHotTemp_9']],
        [[0],['RackColdTemp_12','RackColdTemp_13','RackColdTemp_14','RackColdTemp_15'],
        ['RackHotTemp_12','RackHotTemp_13','RackHotTemp_14','RackHotTemp_15']]]
"""
def devide_area(new_pos): # 将设备分区
    ac_dict = {}
    rc_dict = {}
    for key in new_pos.keys():
        if len(new_pos[key]) == 3:
            position = re.findall(r"\d+\.?\d*", key)
            ac_dict[new_pos[key]] = [int(position[0]),int(position[1])]
        else:
            position = re.findall(r"\d+\.?\d*", key)
            rc_dict[new_pos[key]] = [int(position[0]), int(position[1])]
    #print("ac_dict:",ac_dict)
    #print("rc_dict:",rc_dict)
    rack_ac = {}
    rack_dis = {}
    for rc_key in rc_dict.keys(): # 计算各机柜同对面空调的距离
        rack_ac[rc_key] = []
        rack_dis[rc_key] = []
        for ac_key in ac_dict.keys():
            if rc_dict[rc_key][1] != ac_dict[ac_key][1]:
                dis = math.sqrt(math.pow((rc_dict[rc_key][0] - ac_dict[ac_key][0]),2) +
                                math.pow((rc_dict[rc_key][1] - ac_dict[ac_key][1]),2))
                rack_ac[rc_key].append(ac_key)
                rack_dis[rc_key].append(dis)
    rc_to_ac = {}
    for rack in rack_ac.keys():
        min_ac = rack_dis[rack].index(min(rack_dis[rack]))
        if rack_ac[rack][min_ac] not in rc_to_ac.keys():
            rc_to_ac[rack_ac[rack][min_ac]] = [rack]
        else:
            rc_to_ac[rack_ac[rack][min_ac]].append(rack)
    #print("rc_to_ac:",rc_to_ac)
    distribution = []
    for ac in rc_to_ac.keys():
        ac_to_rc = [[int(re.findall(r"\d+\.?\d*", ac)[0])],[],[]]
        for i in range(len(rc_to_ac[ac])):
            ac_to_rc[1].append('RackColdTemp_' + re.findall(r"\d+\.?\d*", rc_to_ac[ac][i])[0])
            ac_to_rc[2].append('RackHotTemp_' + re.findall(r"\d+\.?\d*", rc_to_ac[ac][i])[0])
        distribution.append(ac_to_rc.copy())
    return distribution
def get_ac_temp_label(new_EnvState): # 得到空调编号
    ac = [[],[]]
    for key in new_EnvState.keys():
        if re.sub(r'[^A-Za-z]', '', key) == 'AcDeliveryTemp' :
            ac[0].append('AcDeliveryTemp_' + re.findall(r"\d+\.?\d*", key)[0])
            ac[1].append('AcReturnTemp_' + re.findall(r"\d+\.?\d*", key)[0])
    return ac

def main():
    dEnvState = {'ParamLabel': ['EnvTemp', 'ItElectricPower', 'Humidity', 'AcDeliveryTemp_24', 'AcDeliveryTemp_25',
                                'AcDeliveryTemp_27', 'AcDeliveryTemp_28', 'AcDeliveryTemp_43', 'AcReturnTemp_24',
                                'AcReturnTemp_25', 'AcReturnTemp_27', 'AcReturnTemp_28', 'AcReturnTemp_43', 'RackColdTemp_12',
                                'RackColdTemp_13', 'RackColdTemp_14', 'RackColdTemp_15', 'RackColdTemp_22', 'RackColdTemp_23',
                                'RackColdTemp_24', 'RackColdTemp_25', 'RackColdTemp_27', 'RackColdTemp_29', 'RackColdTemp_3',
                                'RackColdTemp_30', 'RackColdTemp_31', 'RackColdTemp_34', 'RackColdTemp_35', 'RackColdTemp_4',
                                'RackColdTemp_7', 'RackColdTemp_8', 'RackColdTemp_9', 'RackHotTemp_12', 'RackHotTemp_13',
                                'RackHotTemp_14', 'RackHotTemp_15', 'RackHotTemp_22', 'RackHotTemp_23', 'RackHotTemp_24',
                                'RackHotTemp_25', 'RackHotTemp_27', 'RackHotTemp_29', 'RackHotTemp_3', 'RackHotTemp_30',
                                'RackHotTemp_31', 'RackHotTemp_34', 'RackHotTemp_35', 'RackHotTemp_4', 'RackHotTemp_7',
                                'RackHotTemp_8', 'RackHotTemp_9', 'InstantPue', 'TotalElectricPower', 'AcWorkTime_24',
                                'AcWorkTime_25', 'AcWorkTime_27', 'AcWorkTime_28', 'AcWorkTime_43', 'AcSetTemp_24',
                                'AcSetTemp_25', 'AcSetTemp_27', 'AcSetTemp_28', 'AcSetTemp_43', 'AcOnoffState_24',
                                'AcOnoffState_25', 'AcOnoffState_27', 'AcOnoffState_28', 'AcOnoffState_43', 'ColdTempMean',
                                'AcXPos_24', 'AcXPos_25', 'AcXPos_27', 'AcXPos_28', 'AcXPos_43', 'AcYPos_24', 'AcYPos_25',
                                'AcYPos_27', 'AcYPos_28', 'AcYPos_43', 'RackXPos_12', 'RackXPos_13', 'RackXPos_14',
                                'RackXPos_15', 'RackXPos_2', 'RackXPos_22', 'RackXPos_23', 'RackXPos_24', 'RackXPos_25',
                                'RackXPos_27', 'RackXPos_29', 'RackXPos_3', 'RackXPos_30', 'RackXPos_31', 'RackXPos_34',
                                'RackXPos_35', 'RackXPos_4', 'RackXPos_7', 'RackXPos_8', 'RackXPos_9', 'RackYPos_12',
                                'RackYPos_13', 'RackYPos_14', 'RackYPos_15', 'RackYPos_2', 'RackYPos_22', 'RackYPos_23',
                                'RackYPos_24', 'RackYPos_25', 'RackYPos_27', 'RackYPos_29', 'RackYPos_3', 'RackYPos_30',
                                'RackYPos_31', 'RackYPos_34', 'RackYPos_35', 'RackYPos_4', 'RackYPos_7', 'RackYPos_8',
                                'RackYPos_9'],
                 'ParamData': [29.1, 26000, 83, 19.2277, 18.5495, 20.0901, 18.6801, 23.8992, 25.2316, 25.3941, 24.276,
                               24.8201, 26.0606, 21.189, 21.733, 21.7707, 21.5087, 22.2106, 22.0545, 20.8345, 21.4246,
                               23.7925, 21.3001, 24.2259, 21.1071, 21.4088, 24.0461, 24.0819, 23.8659, 21.9263, 21.5559,
                               21.2783, 26.6294, 22.3335, 22.9156, 25.3649, 23.2484, 23.0157, 25.049, 21.7984, 24.6786,
                               27.8393, 24.8563, 22.7249, 25.7371, 26.9495, 24.8187, 32.128, 24.0094, 21.6256, 24.4199,
                               1.3663, 35523.8, 1, 1, 1, 1, -1, 20, 20, 20, 20, 20, 1, 1, 1, 1, 0, 22.2, 1092, 752, 752,
                               412, 1160, 350, 350, 100, 100, 100, 820, 888, 956, 1024, 140, 344, 412, 480, 548, 684,
                               820, 208, 888, 956, 1160, 1228, 276, 480, 548, 616, 100, 100, 100, 100, 100, 350, 350,
                               350, 350, 350, 350, 100, 350, 350, 350, 350, 100, 100, 100, 100]}

    a,r,p,n = get_pos(dEnvState)
    #print(a)
    #print(r)
    print("AC_NAME:",n)
    print(p)
    new_reveal = get_new_label(p)
    print(new_reveal)
    new_EnvState = get_new_dEnvState(dEnvState,p,new_reveal)
    #jEnvState = trans_d_to_j(dEnvState)
    #print("jEnvState:\n",jEnvState)
    print("new_EnvState:\n",new_EnvState)
    AC = get_ac_temp_label(new_EnvState)
    print('AC:',AC)
    d=devide_area(new_reveal)
    print(d)

if "__main__" == __name__:
    main()


