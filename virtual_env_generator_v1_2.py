#coding:utf-8
'''
Purpose:
    用于为虚拟测试提供虚拟环境
Author：
    liu.jingmin.andy
Date:
    2020.7.30
Note:
                    SA结构：
    ------------------------------------------
                    热通道
                                  
    负载1 空调1 负载2 空调2 负载3 空调3 负载4
      
                    冷通道              
                    
    负载5 空调4 负载6 空调5 负载7 空调6 负载8
      
                    热通道                
    ------------------------------------------
    空调个数：6，equipId为：1，2，3，4，5，6
    机柜个数：8，equipId为：1，2，3，4，5，6，7，8
    每个空调额定制冷量为：35kw，能效系数 2.5
    每个空调最佳制冷量为：21kw(60%)，能效系数 4.0
    每个空调最小制冷量为：12kw(34%)，能效系数 3.5
    ！！！！！！由上面功率系数可以知道，本虚拟系统最佳pue为1.25，最差为1.4且最差时会造成环境安全问题！！！！！！
    总额定制冷量：210kw 总最小制冷量：72kw
    空调能效系数由制冷功率非线性确定
    由空调限制的单负载最大值：26.25kw
    机柜负载24小时周期性变化，为预先设置的固定值，单位为kw
    负载功率最高时：负载1为13.5kw，其它9.5kw，共80kw，占比38%
    负载功率最低时：负载1为11kw，其它8kw，共67kw，占比32%
    it功率为负载功率和
    开启状态空调的制冷功率=it功率/空调开启数
    开启状态空调的电能由能效系数和制冷功率确定
    空调送风温度为设定温度
    空调回风温度由设定温度和制冷功率线性确定
    机柜冷通道温度由送风温度、与各空调的距离和负载功率确定
    机柜热通道由机柜冷通道温度和负载功率线性确定。机柜冷通道温度按负载功率加上一个值。
    
    空调电能为所有空调电能之和
    总电能为it电能与空调电能和
    
    假定空调只有制冷功能
    空调支持的最大送风回风温差24度，每度使环境下降1/4度
    每台空调最大可以使环境下降6度
    送风回风温差为12时pue最大，向两侧递增
    任一空调对环境的贡献相等
    设定温度域值12-34
    送风温度设定为和设定温度相等，回风温度由计算得
'''
import copy
import random
import numpy as np

from config import PARAM_NAMESPACE


class EnvItPower(object):
    def __init__(self):
        '''
        机柜1负载最大为13.5kw，最小为11kw
        其它机柜负载最大为9.5kw，最小为8kw
        每组288个数，每小时12个，恰好一天。
        负载功率从夜间零点开始记录，先升后降，伪周期性循环。
        '''
        self.__it_power_generator_1 = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11.2, 11.2, \
                11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.4, 11.4, 11.4, 11.4, 11.4, \
                11.4, 11.4, 11.4, 11.4, 11.4, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, \
                11.5, 11.5, 11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 11.9, \
                11.9, 11.9, 11.9, 11.9, 11.9, 11.9, 11.9, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, \
                12.1, 12.1, 12.1, 12.1, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, \
                12.3, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.6, 12.6, \
                12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.8, 12.8, 12.8, 12.8, 12.8, \
                12.8, 12.8, 12.8, 12.8, 12.8, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13.1, \
                13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.3, 13.3, 13.3, 13.3, \
                13.3, 13.3, 13.3, 13.3, 13.3, 13.3, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, \
                13.5, 13.5, 13.5, 13.3, 13.3, 13.3, 13.3, 13.3, 13.3, 13.3, 13.3, 13.1, 13.1, \
                13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13, 13, 13, 13, 13, \
                13, 13, 13, 13, 13, 12.8, 12.8, 12.8, 12.8, 12.8, 12.8, 12.8, 12.8, 12.8, 12.8, \
                12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.4, 12.4, 12.4, \
                12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.4, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, \
                12.3, 12.3, 12.3, 12.3, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, \
                12.1, 11.9, 11.9, 11.9, 11.9, 11.9, 11.9, 11.9, 11.9, 11.9, 11.9, 11.7, 11.7, \
                11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 11.5, 11.5, 11.5, 11.5, 11.5, \
                11.5, 11.5, 11.5, 11.5, 11.5, 11.4, 11.4, 11.4, 11.4, 11.4, 11.4, 11.4, 11.4, \
                11.4, 11.4, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11.2, 11, 11, \
                11, 11, 11, 11, 11, 11, 11, 11]
        generator_else_1 = [8, 8, 8, 8, 8, 8, 8, 8, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, \
                8.1, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.3, 8.3, 8.3, \
                8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, \
                8.4, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.6, 8.6, 8.6, 8.6, 8.6, \
                8.6, 8.6, 8.6, 8.6, 8.6, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.8, \
                8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, \
                8.9, 8.9, 8.9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, \
                9.1, 9.1, 9.1, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.4, 9.4, 9.4, \
                9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, \
                9.5, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.2, 9.2, 9.2, 9.2, 9.2, \
                9.2, 9.2, 9.2, 9.2, 9.2, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9, 9, \
                9, 9, 9, 9, 9, 9, 9, 9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.8, 8.8, 8.8, \
                8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, \
                8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, \
                8.5, 8.5, 8.5, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.3, 8.3, 8.3, 8.3, \
                8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.1, \
                8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        generator_else_2 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, \
                8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, \
                8.3, 8.3, 8.3, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.5, 8.5, 8.5, 8.5, \
                8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.7, \
                8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, \
                8.8, 8.8, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 9, 9, 9, 9, 9, 9, 9, 9, \
                9, 9, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, \
                9.2, 9.2, 9.2, 9.2, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.5, 9.5, 9.5, \
                9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, \
                9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.2, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, \
                9.1, 9.1, 9.1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, 8.9, \
                8.9, 8.9, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 8.7, 8.7, 8.7, 8.7, 8.7, \
                8.7, 8.7, 8.7, 8.7, 8.7, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6, 8.5, 8.5, \
                8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, \
                8.4, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, \
                8.2, 8.2, 8.2, 8.2, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8, 8, 8, 8, 8, \
                8, 8, 8, 8, 8]
        self.__it_power_generator_else = [generator_else_1, generator_else_2]
        self.__generator_else_idx = 0
        self.__cur_idx = 0

    def get(self):
        it_power_1    = self.__it_power_generator_1[self.__cur_idx] + np.clip(random.gauss(0, 0.05), -0.07, 0.07)
        it_power_else = self.__it_power_generator_else[self.__generator_else_idx][self.__cur_idx] + \
                np.clip(random.gauss(0, 0.05), -0.07, 0.07)
        if self.__cur_idx != 288 - 1:
            self.__cur_idx += 1
        else:
            self.__cur_idx = 0
            self.__generator_else_idx = np.random.choice(list(range(len(self.__it_power_generator_else))))
        aItPoweres = [it_power_1]
        aItPoweres.extend([it_power_else]*(8-1))
        return aItPoweres
_env_it_power_ins = EnvItPower()
def _get_env_it_power():
    return _env_it_power_ins.get()


_SettingTempArray = [23, 23, 23, 23, 23, 23]
_OnoffFlagArray = [1, 1, 1, 1, 1, 1]
_AcWorkTimeArray = [1, 1, 1, 1, 1, 1]
_EnvState = {}

'''
上一次的冷量的过剩，会导致上一次的冷通道温度持续下降，但不应该影响到现在。冷量过剩的表现，
pue不变，温度持续下降，但同时记录和计算温度是困难的，所以我这里记录冷量过剩或不足的数值，再
反过来影响冷通道温度。

用于记录冷量过剩或冷量不足状态持续状态
flag:
    0：正常状态
    1：异常状态。冷量过剩，即空调的最小制冷量都比需求冷量大。
    -1：异常状态。冷量不足，即空调的最大制冷量都比需求冷量小。
num_more:
    int值：冷量过剩状态的持续时长。
'''
_OutOfControlRecord = {"flag":0, "num_more":0, "num_less":0}


class EnvGen(object):
    def __init__(self):
        self.__iTargetTemp = 23
        self.__iAcNum = len(_OnoffFlagArray)
        self.__iRackNum = 8
    
    def __get_pos_info(self):
        '''
        ------------------------------------------
                        热通道
                                      
        负载1(100,200) 空调1(120,200) 负载2(140,200) 空调2(160,200) 负载3(180,200) 空调3(200,200) 负载4(220,200)
          
                        冷通道              
                        
        负载5(100,100) 空调4(120,100) 负载6(140,100) 空调5(160,100) 负载7(180,100) 空调6(200,100) 负载8(220,100)
          
                        热通道                
        ------------------------------------------
        (x, y)
        '''
        dAcPos = {1:[120, 200], 2:[160, 200], 3:[200, 200], 4:[120, 100], 5:[160, 100], 6:[200, 100]}
        dRackPos = {1:[100, 200], 2:[140, 200], 3:[180, 200], 4:[220, 200], \
                5:[100, 100], 6:[140, 100], 7:[180, 100], 8:[220, 100]}
        aPosThresh = [100, 220, 100, 200]
        aPosLabeles = [PARAM_NAMESPACE.sXPosMin, PARAM_NAMESPACE.sXPosMax, PARAM_NAMESPACE.sYPosMin,\
                PARAM_NAMESPACE.sYPosMax]
        aPosValues = copy.deepcopy(aPosThresh)
        aPosLabeles.extend(["%s1"%PARAM_NAMESPACE.sAcXPosPrefix, "%s1"%PARAM_NAMESPACE.sAcYPosPrefix])
        aPosValues.extend(dAcPos[1])
        aPosLabeles.extend(["%s2"%PARAM_NAMESPACE.sAcXPosPrefix, "%s2"%PARAM_NAMESPACE.sAcYPosPrefix])
        aPosValues.extend(dAcPos[2])
        aPosLabeles.extend(["%s3"%PARAM_NAMESPACE.sAcXPosPrefix, "%s3"%PARAM_NAMESPACE.sAcYPosPrefix])
        aPosValues.extend(dAcPos[3])
        aPosLabeles.extend(["%s4"%PARAM_NAMESPACE.sAcXPosPrefix, "%s4"%PARAM_NAMESPACE.sAcYPosPrefix])
        aPosValues.extend(dAcPos[4])
        aPosLabeles.extend(["%s5"%PARAM_NAMESPACE.sAcXPosPrefix, "%s5"%PARAM_NAMESPACE.sAcYPosPrefix])
        aPosValues.extend(dAcPos[5])
        aPosLabeles.extend(["%s6"%PARAM_NAMESPACE.sAcXPosPrefix, "%s6"%PARAM_NAMESPACE.sAcYPosPrefix])
        aPosValues.extend(dAcPos[6])
        aPosLabeles.extend(["%s1"%PARAM_NAMESPACE.sRackXPosPrefix, "%s1"%PARAM_NAMESPACE.sRackYPosPrefix])
        aPosValues.extend(dRackPos[1])
        aPosLabeles.extend(["%s2"%PARAM_NAMESPACE.sRackXPosPrefix, "%s2"%PARAM_NAMESPACE.sRackYPosPrefix])
        aPosValues.extend(dRackPos[2])
        aPosLabeles.extend(["%s3"%PARAM_NAMESPACE.sRackXPosPrefix, "%s3"%PARAM_NAMESPACE.sRackYPosPrefix])
        aPosValues.extend(dRackPos[3])
        aPosLabeles.extend(["%s4"%PARAM_NAMESPACE.sRackXPosPrefix, "%s4"%PARAM_NAMESPACE.sRackYPosPrefix])
        aPosValues.extend(dRackPos[4])
        aPosLabeles.extend(["%s5"%PARAM_NAMESPACE.sRackXPosPrefix, "%s5"%PARAM_NAMESPACE.sRackYPosPrefix])
        aPosValues.extend(dRackPos[5])
        aPosLabeles.extend(["%s6"%PARAM_NAMESPACE.sRackXPosPrefix, "%s6"%PARAM_NAMESPACE.sRackYPosPrefix])
        aPosValues.extend(dRackPos[6])
        aPosLabeles.extend(["%s7"%PARAM_NAMESPACE.sRackXPosPrefix, "%s7"%PARAM_NAMESPACE.sRackYPosPrefix])
        aPosValues.extend(dRackPos[7])
        aPosLabeles.extend(["%s8"%PARAM_NAMESPACE.sRackXPosPrefix, "%s8"%PARAM_NAMESPACE.sRackYPosPrefix])
        aPosValues.extend(dRackPos[8])
        return aPosLabeles, aPosValues, dAcPos, dRackPos, aPosThresh
    
    def __refresh_work_rest_time(self):
        '''
        更新记录空调的工作休息时间
        '''
        for i in range(len(_OnoffFlagArray)):
            tmp = _AcWorkTimeArray[i]
            if 1 == _OnoffFlagArray[i]:
                if tmp > 0:
                    _AcWorkTimeArray[i] = tmp + 1
                else:
                    _AcWorkTimeArray[i] = 1
            elif 0 == _OnoffFlagArray[i]:
                if tmp < 0:
                    _AcWorkTimeArray[i] = tmp -1
                else:
                    _AcWorkTimeArray[i] = -1
            else:
                print("error:__refresh_work_rest_time")

    def __t3(self, dAcPos, dRackPos, aPosThresh):
        '''
        获取归化一的空调位置信息、机柜位置信息及开启的空调id
        '''
        iXMin, iXMax, iYMin, iYMax = aPosThresh
        dNormedRackPos = {}
        for iRackId in dRackPos:
            dNormedRackPos[iRackId] = [(dRackPos[iRackId][0]-iXMin)/(iXMax-iXMin), \
                    (dRackPos[iRackId][1]-iYMin)/(iYMax-iYMin)]
        aOpenedAcIdes = []
        for num in range(self.__iAcNum):
            if 1 == _OnoffFlagArray[num]:
                aOpenedAcIdes.append(num+1)#空调的编号从1开始而非从0开始
        dNormedOpenedAcPos = {}
        for each in aOpenedAcIdes:
            dNormedOpenedAcPos[each] = [(dAcPos[each][0]-iXMin)/(iXMax-iXMin), (dAcPos[each][1]-iYMin)/(iYMax-iYMin)]
        return dNormedRackPos, dNormedOpenedAcPos, aOpenedAcIdes

    def __get_ac_power(self, it_electric_power, iOpenedAcNum):
        '''
        获取空调的制冷量和电功率
        每个空调额定制冷量为：35kw，能效系数 2.5
        每个空调最佳制冷量为：21kw，能效系数 4.0
        每个空调最小制冷量为：12kw，能效系数 3.5
        空调能效系数由制冷功率非线性确定
        当制冷不足，或最小制冷量都有冗余时，使用fAcColdPower进行标记
        
        Return:
        ---------------------------------------------------
            fAcPower:单台空调的耗电量
            fAcColdPower：单台空调的制冷量
        '''
        fAcColdPower = float(it_electric_power) / iOpenedAcNum
        if fAcColdPower < 12:#需求冷量比空调的最小制冷量都小，制冷过剩。
            fAcColdPower = 12
            if 1 == _OutOfControlRecord["flag"]:
                _OutOfControlRecord["num_more"] += 1
            else:
                _OutOfControlRecord["flag"] = 1
                _OutOfControlRecord["num_more"] = 0
            if _OutOfControlRecord["num_less"] > 0:
                _OutOfControlRecord["num_less"] -= 1
        elif fAcColdPower > 35:#需求冷量比空调的最大制冷量都大，制冷不足。
            fAcColdPower = 35
            if -1 == _OutOfControlRecord["flag"]:
                _OutOfControlRecord["num_less"] += 1
            else:
                _OutOfControlRecord["flag"] = -1
                _OutOfControlRecord["num_less"] = 0
            if _OutOfControlRecord["num_more"] > 0:
                _OutOfControlRecord["num_more"] -= 1
        else:
            _OutOfControlRecord["flag"] = 0
            if _OutOfControlRecord["num_less"] > 0:
                _OutOfControlRecord["num_less"] -= 1
            if _OutOfControlRecord["num_more"] > 0:
                _OutOfControlRecord["num_more"] -= 1
        
        if fAcColdPower >= 21:
            fCoef = (fAcColdPower-21) * (2.5-4) / (35-21) + 4
        else:
            fCoef = (fAcColdPower-12) * (4-3.5) / (21-12) + 3.5
        fAcPower = fAcColdPower / fCoef
        return fAcPower, fAcColdPower
        
    def __get_ac_return_temp(self, fAcColdPower, aOpenedAcIdes):
        '''
        空调回风温度由设定温度和制冷功率线性确定
        具体计算：
        1、每21kw的冷量会导致12度的送回风温差。
        '''
        fDelieverReturnDiff = fAcColdPower * 12 / 21
        aAcReturnTemp = []
        for num in range(self.__iAcNum):
            iAcId = num + 1
            if iAcId not in aOpenedAcIdes:
                aAcReturnTemp.append(-1)#关闭的空调，回风温度无效
            else:
                aAcReturnTemp.append(_SettingTempArray[num] + fDelieverReturnDiff)
        return aAcReturnTemp

    def __get_rack_temp(self, dNormedRackPos, dNormedOpenedAcPos, aOpenedAcIdes, env_it_power):
        '''
        机柜冷通道温度由空调的送风温度、与各空调的距离、负载功率确定
        具体计算：
        1、初始：机柜冷通道温度 = 送风温度 + 3（3是我根据所有空调数除2取的）
        2、每台空调最多使初始机柜冷通道温度下降0.5度，
            实际下降量根据机柜与空调的x，y距离确定，x最大为0.8*0.5，y最大为0.2*0.5。坐标归一到[0,1]
        3、根据机柜功率再提升第2步得到的机柜冷通道温度。按每10kw提升1度计算。
        注：第2步计算时我引入了权重，x，y坐标不对等。因为开对侧的空调对机柜温度影响更大。
        
        机柜热通道由机柜冷通道温度和负载功率线性确定。机柜冷通道温度按负载功率加上一个值
        具体计算：
        1、每1.5kw增加一度
        '''
        fAcDeliveryTemp = np.mean(_SettingTempArray)
        aRackColdTempes = []
        for num in range(self.__iRackNum):
            iRackId = num + 1
            fRackColdTemp = fAcDeliveryTemp + 3
            aRackPos = dNormedRackPos[iRackId]
            for iAcId in aOpenedAcIdes:
                aAcPos = dNormedOpenedAcPos[iAcId]
                fRackColdTemp = fRackColdTemp - (0.8*0.5) * (1 - abs(aRackPos[0] - aAcPos[0]))
                fRackColdTemp = fRackColdTemp - (0.2*0.5) * abs(aRackPos[1] - aAcPos[1])
#                 fRackColdTemp = fRackColdTemp - 0.3 * (1 - abs(aRackPos[0] - aAcPos[0]))
#                 fRackColdTemp = fRackColdTemp - 0.7 * (1 - abs(aRackPos[1] - aAcPos[1]))
            fRackColdTemp = fRackColdTemp + env_it_power[num] / 10.
            
            #冷量过剩的状态每持续5分钟使冷通道温度下降0.1度
            #冷量不足的状态每持续5分钟使冷通道温度上升0.1度
            fRackColdTemp = fRackColdTemp - _OutOfControlRecord["num_more"] * 0.5 + _OutOfControlRecord["num_less"] * 0.5
            
            aRackColdTempes.append(fRackColdTemp)
        
        aRackHotTempes = []
        for num in range(self.__iRackNum):
            aRackHotTempes.append(aRackColdTempes[num] + env_it_power[num] / 1.5)
        return aRackColdTempes, aRackHotTempes
    
    def run(self):
        aPosLabeles, aPosValues, dAcPos, dRackPos, aPosThresh = self.__get_pos_info()
        env_it_power = _get_env_it_power()
        it_electric_power = float(np.sum(env_it_power))
        self.__refresh_work_rest_time()
        dNormedRackPos, dNormedOpenedAcPos, aOpenedAcIdes = self.__t3(dAcPos, dRackPos, aPosThresh)
        iOpenedAcNum = len(aOpenedAcIdes)
        fAcPower, fAcColdPower = self.__get_ac_power(it_electric_power, iOpenedAcNum)
        aAcReturnTemp = self.__get_ac_return_temp(fAcColdPower, aOpenedAcIdes)
        aRackColdTempes, aRackHotTempes = self.__get_rack_temp(\
                dNormedRackPos, dNormedOpenedAcPos, aOpenedAcIdes, env_it_power)
        fColdTempMean = float(np.mean(aRackColdTempes))
        if fColdTempMean > 26:
            fAcElectricPower = fAcPower * iOpenedAcNum
        else:#引入设置温度越低，pue越大的特性。
            fAcElectricPower = fAcPower * iOpenedAcNum * (1 + 0.01*abs(26-fColdTempMean))
        
        _EnvState[PARAM_NAMESPACE.sTimestamp] = "2019-09-01 09:02"#写死不用
        for num in range(len(aPosLabeles)):
            _EnvState[aPosLabeles[num]] = aPosValues[num]
        for num in range(self.__iAcNum):
            _EnvState["%s%d"%(PARAM_NAMESPACE.sAcSetTempPrefix, num+1)] = _SettingTempArray[num]
            _EnvState["%s%d"%(PARAM_NAMESPACE.sAcOnoffStatePrefix, num+1)] = _OnoffFlagArray[num]
            _EnvState["%s%d"%(PARAM_NAMESPACE.sAcDeliveryTempPrefix, num+1)] = _SettingTempArray[num]
            _EnvState["%s%d"%(PARAM_NAMESPACE.sAcWorkTimePrefix, num+1)] = _AcWorkTimeArray[num]
            _EnvState["%s%d"%(PARAM_NAMESPACE.sAcReturnTempPrefix, num+1)] = aAcReturnTemp[num]
        _EnvState[PARAM_NAMESPACE.sItElectricPower] = it_electric_power
        _EnvState[PARAM_NAMESPACE.sItElectricTotalWork] = it_electric_power * (5 / 60)
        _EnvState[PARAM_NAMESPACE.sTotalElectricPower] = fAcElectricPower + it_electric_power
        _EnvState[PARAM_NAMESPACE.sTotalElectricTotalWork] = (fAcElectricPower + it_electric_power) * (5 / 60)
        _EnvState[PARAM_NAMESPACE.sInstantPue] = (fAcElectricPower + it_electric_power) / it_electric_power
        for num in range(self.__iRackNum):
            _EnvState["%s%d"%(PARAM_NAMESPACE.sRackColdTempPrefix, num+1)] = aRackColdTempes[num]
            _EnvState["%s%d"%(PARAM_NAMESPACE.sRackHotTempPrefix, num+1)] = aRackHotTempes[num]
        _EnvState[PARAM_NAMESPACE.sColdTempMean] = fColdTempMean
        _EnvState[PARAM_NAMESPACE.sHotTempMean] = float(np.mean(aRackHotTempes))
        return _EnvState
def get_env_state__():
    return EnvGen().run()

def get_env_state():
    jEnvState = get_env_state__()
    
    jEnvState2 = {}
    aParamLabeles = []
    aParamDataes = []
    for sLabel in jEnvState:
        aParamLabeles.append(sLabel)
        aParamDataes.append(jEnvState[sLabel])
    jEnvState2[PARAM_NAMESPACE.sParamLabel] = aParamLabeles
    jEnvState2[PARAM_NAMESPACE.sParamData] = aParamDataes
    import logging
    logging.info(_OutOfControlRecord)
    return jEnvState, jEnvState2

def set_setting_temp(temp_array):
    for i in range(len(_SettingTempArray)):
        if temp_array[i] < 15 or temp_array[i] > 25:
            print("The setting temp invalid")
            break
    for i in range(len(_SettingTempArray)):
        _SettingTempArray[i] = temp_array[i]

def set_onoff(onoff_array):
    for i in range(len(_OnoffFlagArray)):
        _OnoffFlagArray[i] = onoff_array[i]
    

__all__ = ["get_env_state",
        "set_setting_temp",
        "set_onoff"]

def main3():
    '''
    打印无群控、空调设置采用默认值情况下，一天的冷通道温度和pue走势
    '''
    import matplotlib.pyplot as plt
    aPues = []
    aColdTempes = []
    for num in range(288):
        jEnvState, jEnvState2 = get_env_state()
        aPues.append(jEnvState[PARAM_NAMESPACE.sInstantPue])
        aColdTempes.append(jEnvState[PARAM_NAMESPACE.sColdTempMean])
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_xlabel("time")
    ax1.set_ylabel("pue")
    ax1.plot(aPues)
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_xlabel("time")
    ax2.set_ylabel("cold temp")
    ax2.plot(aColdTempes)
    plt.show()

def main2():
    '''
    打印无群控、空调设置采用默认值情况下，一天的虚拟环境信息
    '''
    for num in range(288):
        jEnvState, jEnvState2 = get_env_state()
        print(jEnvState)
        #print(jEnvState2)

def main1():
    import matplotlib.pyplot as plt
    def sub_main1(ac_num=5):
        set_setting_temp([23] * 10)
        set_onoff([1 for i in range(ac_num)] + [0 for j in range(10-ac_num)])
        aPues = []
        for num in range(288):
            print(".", end="")
            jEnvState, jInData = get_env_state()
            aPues.append(jEnvState["pue"])
        print()
        fMeanPue = np.mean(aPues)
        length = len(aPues)
        plt.plot(range(length), aPues)
        plt.plot(range(length), [fMeanPue]*length)
        plt.show()
    for ac_num in range(1, 10):
        print(ac_num)
        sub_main1(ac_num)

if "__main__" == __name__:
#     main1()
#     main2()
    main2()
