import numpy as np
max_ac_number = 7

def vec_embedding_1(state,old_elect,encode_len):  # the embedding function of input of graph1
    l = state[0]
    # print(l)
    # load_step #= 110*1000/(encode_len/2)
    # load_step = 150*1000 / (encode_len)
    load_step = 150/encode_len/3
    # load_step = 150*1000 / encode_len / 3
    g = l / load_step
    inputs = []
    #inputs.append(round(g,3))

    for h in range(1, int(encode_len*3)+1):
        if h <= g:
            inputs.append(1)
        elif g < h < g + 1 and g % 1 > 0:
            inputs.append(g % 1)
        else:
            inputs.append(0)

    ac_step = max_ac_number / encode_len
    old_ele = old_elect / ac_step

    for h in range(1, int(encode_len) + 1):
        if h <= old_ele:
            inputs.append(1)
        elif old_ele < h < old_ele + 1 and old_ele % 1 > 0:
            inputs.append(old_ele % 1)
        else:
            inputs.append(0)

    temp = state[1:]
    num = len(temp)
    encode_temp_step = 25/encode_len
    for m in range(num):
        temp[m] = (temp[m] - 13)/encode_temp_step
    for n in range(num):
        s = temp[n]
        if s < 0:
            inputs.append(-1)
            for l in range(encode_len - 1):
                inputs.append(0)
        else:
            for k in range(1, encode_len + 1):
                if k <= s:
                    inputs.append(1)
                elif s < k < s + 1 and s % 1 > 0:
                    inputs.append(s % 1)
                else:
                    inputs.append(0)

    # inputs.append(previous_action/max_ac_number)
    #print("#####len:",len(inputs))
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (1, -1))
    return inputs

def vec_embedding_2(state,ac_action,encode_len):  # the embedding function of input of graph1
    l = state[0]
    print(l)
    # load_step = 110*1000/(encode_len/2)
    # load_step = 150*1000 / (encode_len)
    load_step = 150 / encode_len
    g = l / load_step
    inputs = []
    # inputs.append(round(g,3))

    for h in range(1, int(encode_len) + 1):
        if h <= g:
            inputs.append(1)
        elif g < h < g + 1 and g % 1 > 0:
            inputs.append(g % 1)
        else:
            inputs.append(0)
    # """
    # #temp = state[1 + Ac_num * 2:]
    temp = state[1:]
    num = len(temp)
    encode_temp_step = 25 / encode_len
    for m in range(num):
        temp[m] = (temp[m] - 13) / encode_temp_step
    for n in range(num):
        s = temp[n]
        if s < 0:
            inputs.append(-1)
            for l in range(encode_len - 1):
                inputs.append(0)
        else:
            for k in range(1, encode_len + 1):
                if k <= s:
                    inputs.append(1)
                elif s < k < s + 1 and s % 1 > 0:
                    inputs.append(s % 1)
                else:
                    inputs.append(0)

    ac_step =   max_ac_number/encode_len/2
    action =  ac_action / ac_step

    for h in range(1, int(encode_len*2) + 1):
        if h <= action:
            inputs.append(1)
        elif action < h < action + 1 and action % 1 > 0:
            inputs.append(action % 1)
        else:
            inputs.append(0)

    # inputs.append(previous_action/max_ac_number)
    # print("#####len:",len(inputs))
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (1, -1))
    return inputs


def vec_embedding_3(state):  # the embedding function of input of graph1
    l = state[0]
    g = l / 10
    inputs = []
    #inputs.append(round(g,3))

    for h in range(1, 17):
        if h <= g:
            inputs.append(1)
        elif g < h < g + 1 and g % 1 > 0:
            inputs.append(g % 1)
        else:
            inputs.append(0)
    print(inputs)

    """
    inputs = inputs + state[1:1+Ac_num] # 添加空调开关设置
    set_temp = state[1 + Ac_num:1 + Ac_num + Ac_num]
    temp = int(np.mean(set_temp)) # 对当前空调设置温度平均取整
    for i in range(1,5):
        if i == (temp - 14):
            inputs.append(1)
        else:
            inputs.append(0)
    """
    #temp = state[1 + Ac_num * 2:]
    temp = state[1:]
    num = len(temp)
    for m in range(num):
        temp[m] -= 17
    for n in range(num):
        s = temp[n]
        if s < 0:
            inputs.append(-1)
            for l in range(17):
                inputs.append(0)
        else:
            for k in range(1, 19):
                if k <= s:
                    inputs.append(1)
                elif s < k < s + 1 and s % 1 > 0:
                    inputs.append(s % 1)
                else:
                    inputs.append(0)
    #print("#####len:",len(inputs))
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (1, -1))
    return inputs

def vec_embedding_4(state):  # the embedding function of input of graph1
    l = state[0]
    inputs = []
    inputs.append(l / 150)
    #inputs.append(round(g,3))
    temp = state[1:]
    num = len(temp)
    for m in range(num):
        temp[m] = (temp[m] - 13) / 25
        inputs.append(temp[m])
    inputs = np.reshape(inputs, (1, -1))
    return inputs


def vec_embedding_5(state,old_elect,encode_len):  # the embedding function of input of graph1
    l = state[0]
    # print(l)
    # load_step #= 110*1000/(encode_len/2)
    # load_step = 150*1000 / (encode_len)
    load_step = 150 / encode_len / 3
    #load_step = 150*1000 / encode_len / 3
    g = l / load_step
    inputs = []
    # inputs.append(round(g,3))

    for h in range(1, int(encode_len * 3) + 1):
        if h <= g:
            inputs.append(1)
        elif g < h < g + 1 and g % 1 > 0:
            inputs.append(1*(g % 1))
        else:
            inputs.append(0)

    ac_step = max_ac_number / encode_len / 3
    old_ele = old_elect / ac_step

    for h in range(1, int(encode_len * 3) + 1):
        if h <= old_ele:
            inputs.append(1)
        elif old_ele < h < old_ele + 1 and old_ele % 1 > 0:
            inputs.append(old_ele % 1)
        else:
            inputs.append(0)

    temp = state[1:]
    num = len(temp)
    encode_temp_step = 25 / encode_len
    for m in range(num):
        temp[m] = (temp[m] - 13) / encode_temp_step
    for n in range(num):
        s = temp[n]
        if s < 0:
            inputs.append(-1)
            for l in range(encode_len - 1):
                inputs.append(0)
        else:
            for k in range(1, encode_len + 1):
                if k <= s:
                    inputs.append(1)
                elif s < k < s + 1 and s % 1 > 0:
                    inputs.append(s % 1)
                else:
                    inputs.append(0)

    # inputs.append(previous_action/max_ac_number)
    # print("#####len:",len(inputs))
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (1, -1))
    return inputs

def vec_embedding_6(state,encode_len):  # the embedding function of input of graph1
    l = state[0]
    print(l)
    # load_step = 110*1000/(encode_len/2)
    load_step = 150 / encode_len / 3
    # load_step = 150*1000 / encode_len / 3
    g = l / load_step
    inputs = []
    # inputs.append(round(g,3))

    for h in range(1, int(encode_len * 3) + 1):
        if h <= g:
            inputs.append(1)
        elif g < h < g + 1 and g % 1 > 0:
            inputs.append(g % 1)
        else:
            inputs.append(0)

    temp = state[1:]
    num = len(temp)
    encode_temp_step = 25/encode_len
    for m in range(num):
        temp[m] = (temp[m] - 13)/encode_temp_step
    for n in range(num):
        s = temp[n]
        if s < 0:
            inputs.append(-1)
            for l in range(encode_len - 1):
                inputs.append(0)
        else:
            for k in range(1, encode_len + 1):
                if k <= s:
                    inputs.append(1)
                elif s < k < s + 1 and s % 1 > 0:
                    inputs.append(s % 1)
                else:
                    inputs.append(0)
    # print("#####len:",len(inputs))
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (1, -1))
    return inputs

def get_state_input_3(jEnvState,AC,Area,Ac_num,ac_list):  # 热通道温度减去冷通道温度，结合空调设置温度
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

def discount_rewards(ep_rs,gamma,len_epi,num_epi):
    discounted_ep_rs = []
    for i in range(num_epi):
        discounted_ep_rs.append(np.zeros_like(ep_rs[(i*len_epi):((i+1)*len_epi)]))
        #a=np.array(ep_rs[(i * 4):((i + 1) * 4)])
        #print('######:',a)
    for j in range(num_epi):
        running_add = 0
        for t in reversed(range(0, len(ep_rs[(j*len_epi):((j+1)*len_epi)]))):
            running_add = running_add * gamma + ep_rs[(j*len_epi):((j+1)*len_epi)][t]  # 计算累计奖励
            #print(running_add)
            discounted_ep_rs[j][t] = running_add
    # discounted_mean = np.mean(discounted_ep_rs,axis=0)
    # mean_repeat = [discounted_mean for i in range(0,num_epi)]
    # matrix = np.array(discounted_ep_rs) - np.array(mean_repeat)
    # column_ave = np.mean(discounted_ep_rs,axis = 0)
    # #print (column_ave)
    # matrix = discounted_ep_rs - column_ave
    # discounted_ep_rs -= np.mean(discounted_ep_rs)
    # discounted_ep_rs /= np.std(discounted_ep_rs)
    matrix = np.array(discounted_ep_rs)
    #print("###:",matrix)
    return matrix

def add_vtarg_and_adv( seg, gamma=0.99, lam=0.95):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"]) # 比ac列长1 len(vpred) = len(ac）+1
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"] # 经过处理的pue序列 -(math.exp(pue)-3.66)*0.1
    lastgaelam = 0
    for t in reversed(range(T)): # 类似于累计奖励  # 这部分还是有点绕，不清楚为啥这样处理，还要理解，后天处理
        nonterminal = 1-new[t+1] # 全都是False，也就是没有终点
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]




def main():
    l = []
    for i in range(12):
        l.append(i+1.0)
    print(l)
    a = discount_rewards(l,0.5,3,4) *0.1

    print(a)

if "__main__" == __name__:
    main()





