# coding:utf-8
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import re
import joblib

EnvState = {'2020/6/9 17': [31.7, 67], '2020/6/9 18': [30.9, 70], '2020/6/9 19': [30.4, 70], '2020/6/9 20': [29.8, 76],
            '2020/6/9 21': [29.4, 79], '2020/6/9 22': [29.4, 79], '2020/6/9 23': [29.2, 81], '2020/6/10 0': [29.1, 83],
            '2020/6/10 1': [29.0, 84], '2020/6/10 2': [28.9, 83], '2020/6/10 3': [28.9, 82], '2020/6/10 4': [28.7, 83],
            '2020/6/10 5': [28.6, 83], '2020/6/10 6': [28.6, 84], '2020/6/10 7': [28.9, 84], '2020/6/10 8': [29.4, 81],
            '2020/6/10 9': [29.8, 80],
            '2020/6/10 10': [30.1, 79], '2020/6/10 11': [30.6, 74], '2020/6/10 12': [31.6, 71],
            '2020/6/10 13': [31.9, 67],
            '2020/6/10 14': [31.5, 67], '2020/6/10 15': [31.3, 68], '2020/6/10 16': [31.0, 70],
            '2020/6/10 17': [30.5, 74],
            '2020/6/10 18': [30.2, 75], '2020/6/10 19': [30.1, 74], '2020/6/10 20': [29.6, 79],
            '2020/6/10 21': [29.5, 79], '2020/6/10 22': [29.4, 79], '2020/6/10 23': [29.3, 79],
            '2020/6/11 0': [29.2, 80],
            '2020/6/11 1': [29.1, 81], '2020/6/11 2': [29.0, 82], '2020/6/11 3': [29.0, 82], '2020/6/11 4': [28.7, 83],
            '2020/6/11 5': [28.7, 83], '2020/6/11 6': [28.7, 83], '2020/6/11 7': [28.9, 81], '2020/6/11 8': [29.2, 78],
            '2020/6/11 9': [30.2, 73],
            '2020/6/11 10': [30.8, 68], '2020/6/11 11': [30.1, 67], '2020/6/11 12': [31.7, 63],
            '2020/6/11 13': [31.7, 62], '2020/6/11 14': [31.2, 67],
            '2020/6/11 15': [31.6, 64], '2020/6/11 16': [30.9, 68], '2020/6/11 17': [31.1, 67],
            '2020/6/11 18': [30.5, 72],
            '2020/6/11 19': [30.1, 74], '2020/6/11 20': [29.8, 77], '2020/6/11 21': [29.9, 75],
            '2020/6/11 22': [29.9, 76],
            '2020/6/11 23': [29.7, 78], '2020/6/12 0': [28.4, 79], '2020/6/12 1': [28.3, 79], '2020/6/12 2': [28.2, 79],
            '2020/6/12 3': [28.1, 80],
            '2020/6/12 4': [28, 81], '2020/6/12 5': [27.4, 86], '2020/6/12 6': [27.6, 86], '2020/6/12 7': [28.9, 83],
            '2020/6/12 8': [29.3, 75], '2020/6/12 9': [30.1, 70], '2020/6/12 10': [30.7, 65],
            '2020/6/12 11': [30.9, 65],
            '2020/6/12 12': [31.6, 63], '2020/6/12 13': [31.8, 60], '2020/6/12 14': [32.2, 58],
            '2020/6/12 15': [32.3, 57],
            '2020/6/12 16': [31.7, 58], '2020/6/12 17': [31.5, 60], '2020/6/15 10': [30.1, 67],
            '2020/6/15 11': [30.8, 63],
            '2020/6/15 12': [31.3, 60], '2020/6/15 13': [30.8, 63], '2020/6/15 14': [31.4, 62],
            '2020/6/15 15': [31.8, 58],
            '2020/6/15 16': [30.5, 67], '2020/6/15 17': [30.4, 70], '2020/6/15 18': [29.7, 72],
            '2020/6/15 19': [28.1, 84],
            '2020/6/15 20': [27.8, 81], '2020/6/15 21': [27.8, 82], '2020/6/15 22': [27.8, 81],
            '2020/6/15 23': [27.3, 85], '2020/6/16 0': [28.4, 82],
            '2020/6/16 1': [28.7, 75], '2020/6/16 2': [28.5, 77], '2020/6/16 3': [28.4, 78], '2020/6/16 4': [28.4, 79],
            '2020/6/16 5': [28.3, 82], '2020/6/16 6': [28.1, 84], '2020/6/16 11': [30.7, 70],
            '2020/6/16 12': [31.7, 67],
            '2020/6/16 13': [31.5, 67], '2020/6/16 14': [31.1, 64], '2020/6/16 15': [31.3, 62],
            '2020/6/16 16': [31.1, 65],
            '2020/6/16 17': [31, 67], '2020/6/17 9': [30.1, 70], '2020/6/17 10': [30.1, 67], '2020/6/17 11': [30.8, 63],
            '2020/6/17 12': [31.3, 60], '2020/6/17 13': [30.8, 63], '2020/6/17 14': [31.4, 62],
            '2020/6/17 15': [31.8, 58], '2020/6/17 16': [30.5, 67],
            '2020/6/17 17': [30.4, 70], '2020/6/17 18': [29.7, 72], '2020/6/18 10': [30.7, 74],
            '2020/6/18 11': [30.7, 70], '2020/6/18 12': [31.7, 67],
            '2020/6/18 13': [31.5, 67], '2020/6/18 14': [31.1, 64], '2020/6/18 15': [31.3, 62],
            '2020/6/18 16': [31.1, 65],
            '2020/6/18 17': [31, 67], '2020/6/18 18': [30.6, 71], '2020/6/18 19': [29.9, 77],
            '2020/6/18 20': [29.6, 78], '2020/6/18 21': [29.6, 79],
            '2020/6/18 22': [29.3, 82], '2020/6/18 23': [29.4, 78], '2020/6/19 0': [28.4, 79],
            '2020/6/19 1': [28.3, 79], '2020/6/19 2': [28.2, 79],
            '2020/6/19 3': [28.1, 80], '2020/6/19 4': [28, 81], '2020/6/19 5': [27.4, 86], '2020/6/19 6': [27.6, 86],
            '2020/6/19 7': [28.9, 83], '2020/6/19 8': [29.3, 75], '2020/6/19 9': [30.1, 70], '2020/6/19 10': [30.7, 65],
            '2020/6/19 11': [30.9, 65], '2020/6/19 12': [31.6, 63], '2020/6/19 13': [31.8, 60],
            '2020/6/19 14': [32.2, 58],
            '2020/6/19 15': [32.3, 57], '2020/6/19 16': [31.7, 58], '2020/6/19 17': [31.5, 60],
            '2020/7/1 11': [31.5, 68],
            '2020/7/1 12': [29.1, 78], '2020/7/1 13': [28.5, 82], '2020/7/1 14': [30.1, 77], '2020/7/1 15': [30.5, 78],
            '2020/7/1 16': [28.9, 83], '2020/7/1 17': [30.1, 77], '2020/7/1 18': [29.2, 83], '2020/7/1 19': [29.4, 84],
            '2020/7/1 20': [29.0, 82], '2020/7/1 21': [28.9, 83], '2020/7/1 22': [28.7, 85], '2020/7/1 23': [28.9, 83],
            '2020/7/2 0': [28.6, 84], '2020/7/2 1': [28.1, 87], '2020/7/2 2': [27.8, 89], '2020/7/2 3': [27.2, 92],
            '2020/7/2 4': [27.2, 91], '2020/7/2 5': [27.4, 90], '2020/7/2 6': [27.6, 89], '2020/7/2 7': [26.9, 96],
            '2020/7/2 8': [28.6, 88], '2020/7/2 9': [30.2, 79], '2020/7/2 10': [31.2, 74], '2020/7/2 11': [30.2, 77],
            '2020/6/22 10': [30.9, 73], '2020/6/22 11': [31.4, 70], '2020/6/22 12': [32.2, 66],
            '2020/6/22 13': [32.1, 65],
            '2020/6/22 14': [32.4, 65], '2020/6/22 15': [32.4, 64], '2020/6/22 16': [32.2, 64],
            '2020/6/22 17': [31.6, 67],
            '2020/6/22 18': [31.1, 67], '2020/6/22 19': [30.3, 72], '2020/6/22 20': [29.8, 77],
            '2020/6/22 21': [29.4, 79],
            '2020/6/22 22': [29.6, 79], '2020/6/22 23': [29.5, 80], '2020/6/23 0': [29.3, 81], '2020/6/23 1': [29, 83],
            '2020/6/23 2': [28.8, 84], '2020/6/23 3': [28.7, 85], '2020/6/23 4': [28.7, 85], '2020/6/23 5': [28.5, 85],
            '2020/6/23 6': [28.7, 85], '2020/6/23 7': [28.3, 88], '2020/6/23 8': [28.6, 87], '2020/6/23 9': [29.8, 79],
            '2020/6/23 10': [31, 72], '2020/6/23 11': [31.6, 66], '2020/6/23 12': [31.7, 68],
            '2020/6/23 13': [32.1, 65],
            '2020/6/23 14': [32.9, 64], '2020/6/23 15': [32.1, 69], '2020/6/23 16': [32.4, 65],
            '2020/6/23 17': [32.7, 67],
            '2020/6/23 18': [30.9, 70], '2020/6/23 19': [30.4, 70], '2020/6/23 20': [29.8, 76],
            '2020/6/23 21': [29.4, 79],
            '2020/6/23 22': [29.4, 79], '2020/6/23 23': [29.2, 81], '2020/6/24 0': [29.1, 83],
            '2020/6/24 1': [29.0, 84],
            '2020/6/24 2': [28.9, 83], '2020/6/24 3': [28.9, 82], '2020/6/24 4': [28.7, 83],
            '2020/6/24 5': [28.6, 83], '2020/6/24 6': [28.6, 84], '2020/6/24 7': [28.9, 84], '2020/6/24 8': [29.4, 81],
            '2020/6/24 9': [29.8, 81],
            '2020/6/28 14': [32.6, 60], '2020/6/28 15': [33.2, 57], '2020/6/28 16': [32.5, 56],
            '2020/6/28 17': [32.0, 61],
            '2020/6/28 18': [31.7, 58], '2020/6/28 19': [30.6, 65], '2020/6/28 20': [30.0, 73],
            '2020/6/28 21': [29.3, 77],
            '2020/6/28 22': [29.0, 78], '2020/6/28 23': [29.1, 78], '2020/6/29 0': [29.0, 75],
            '2020/6/29 1': [28.8, 76],
            '2020/6/29 2': [28.4, 77], '2020/6/29 3': [27.2, 86], '2020/6/29 4': [26.8, 88], '2020/6/29 5': [26.9, 89],
            '2020/6/29 6': [26.9, 91], '2020/6/29 7': [28.9, 82], '2020/6/29 8': [30.1, 70], '2020/6/29 9': [31.7, 67],
            '2020/6/29 10': [32.1, 64], '2020/6/29 11': [32.8, 61], '2020/6/29 12': [33.2, 59],
            '2020/6/29 13': [33.0, 54],
            '2020/6/29 14': [33.2, 56], '2020/6/30 13': [33.2, 64], '2020/6/30 14': [33.3, 59],
            '2020/6/30 15': [32.3, 61],
            '2020/6/30 16': [31.8, 65], '2020/6/30 17': [31.5, 67], '2020/6/30 18': [31.1, 71],
            '2020/6/30 19': [30.6, 69],
            '2020/6/30 20': [30.0, 70], '2020/6/30 21': [29.6, 75], '2020/6/30 22': [29.3, 77],
            '2020/6/30 23': [28.9, 81],
            '2020/7/1 0': [28.0, 87], '2020/7/1 1': [27.6, 89], '2020/7/1 2': [27.5, 89], '2020/7/1 3': [27.6, 89],
            '2020/7/1 4': [27.1, 90], '2020/7/1 5': [27.1, 91], '2020/7/1 6': [26.9, 93], '2020/7/1 7': [29.4, 82],
            '2020/7/1 8': [30.2, 75], '2020/7/1 9': [30.0, 74], '2020/7/1 10': [29.5, 79]
            }

def time_stamp(t):  # 时间处理，便于得到环境温度和湿度
    line = re.findall(r"\d+\.?\d*", t)
    # print(line)
    year = line[0]
    month = line[1]
    day = line[2]
    if month == '8':
        month = '6'
        day = '23'
    hour = line[3]
    minute = line[4]
    if int(minute) < 30:  # 对时间进行四舍五入
        return year + '/' + month + '/' + day + ' ' + hour  # 小于半小时，则舍
    else:
        hour = str(int(hour) + 1)
        if int(hour) == 24:
            hour = '0'
            day = str(int(day) + 1)
            if int(day) == 31:
                day = '1'
                month = str(int(month) + 1)
        return year + '/' + month + '/' + day + ' ' + hour

def get_data(path):
    data = pd.read_csv(path, header=0)
    # 得到 x
    ac_setting = np.array(data[['AcOnoffState_24', 'AcOnoffState_25', 'AcOnoffState_27',
                                'AcOnoffState_28', 'AcOnoffState_43', ]])
    device_temp = np.array(data[['AcSetTemp_24', 'AcSetTemp_25', 'AcSetTemp_27',
                                 'AcSetTemp_28', 'AcSetTemp_43']])
    load = np.array(data[['ItElectricPower']])
    time_list = data['Timestamp']
    env_temp = []
    env_hum = []
    # 得到 y

    ColdTempMean = np.array(data[['ColdTempMean']])

    HotTempMean = np.array(data[['HotTempMean']])

    InstantPue = np.array(data[['InstantPue']])

    AcDeliveryTemp = np.array(data[['AcDeliveryTemp_24', 'AcDeliveryTemp_25', 'AcDeliveryTemp_27',
                                    'AcDeliveryTemp_28', 'AcDeliveryTemp_43']])

    AcReturnTemp = np.array(data[['AcReturnTemp_24', 'AcReturnTemp_25', 'AcReturnTemp_27',
                                  'AcReturnTemp_28', 'AcReturnTemp_43']])

    # Test=np.array(data[[AcDeliveryTemp,AcReturnTemp]])

    RackColdTemp = np.array(data[['RackColdTemp_12', 'RackColdTemp_13', 'RackColdTemp_14', 'RackColdTemp_15',
                                  'RackColdTemp_22', 'RackColdTemp_23', 'RackColdTemp_24', 'RackColdTemp_25',
                                  'RackColdTemp_27', 'RackColdTemp_29', 'RackColdTemp_3', 'RackColdTemp_30',
                                  'RackColdTemp_31', 'RackColdTemp_34', 'RackColdTemp_35', 'RackColdTemp_4',
                                  'RackColdTemp_7', 'RackColdTemp_8', 'RackColdTemp_9']])

    RackHotTemp = np.array(data[['RackHotTemp_12', 'RackHotTemp_13', 'RackHotTemp_14', 'RackHotTemp_15',
                                 'RackHotTemp_22', 'RackHotTemp_23', 'RackHotTemp_24', 'RackHotTemp_25',
                                 'RackHotTemp_27', 'RackHotTemp_29', 'RackHotTemp_3', 'RackHotTemp_30',
                                 'RackHotTemp_31', 'RackHotTemp_34', 'RackHotTemp_35', 'RackHotTemp_4',
                                 'RackHotTemp_7', 'RackHotTemp_8', 'RackHotTemp_9']])
    for j in range(len(time_list)):
        time_index = time_stamp(time_list[j])
        env_state = EnvState[time_index]
        env_temp.append(env_state[0])
        env_hum.append(env_state[1])

    t = np.array(env_temp)
    t = np.reshape(t, (-1, 1))
    h = np.array(env_hum)
    h = np.reshape(h, (-1, 1))
    x = np.concatenate((t, h, load, device_temp, ac_setting), axis=1)
    y = np.concatenate((AcDeliveryTemp, AcReturnTemp, RackColdTemp, RackHotTemp, InstantPue), axis=1)
    # y=Test
    # print(state[:,:8].shape)
    # print(h)
    return x, y


# 归一化处理
def data_normalization(data_train):
    # 将数据集合转换为dataframe
    # print(len(data_train[0]))
    columns = [[0] * len(data_train[0])]
    index = range(0, len(data_train))
    df = pd.DataFrame(data_train, index, columns)
    # print(df)
    values = df.values
    # print(values)
    # 归一化处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # print(scaled)
    return scaled, scaler


def model_train(x_train, x_test, y_train, y_test):  # ,scaler
    # model = XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=400, silent=False, objective='reg:gamma')
    other_params = {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 400, 'silent': False}
    multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:gamma', **other_params)).fit(x_train,
                                                                                                             y_train)
    joblib.dump(multioutputregressor, filename='./model/model.pkl')
    # inv_yhat=multioutputregressor.predict(x_test)
    model = joblib.load('./model/model.pkl')
    inv_yhat = model.predict(x_test)
    # print("###",inv_yhat)
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat.round(2)
    print('predict:\n', inv_yhat.shape)
    np.savetxt('./data/inv_yhat.csv', inv_yhat, delimiter=',')
    # inv_y=scaler.inverse_transform(y_test)
    inv_y = y_test
    np.savetxt('./data/inv_y.csv', inv_y, delimiter=',')
    loss = abs(inv_y - inv_yhat)
    np.savetxt('./data/loss.csv', loss, delimiter=',')
    print('label:\n', inv_y.shape)
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print(rmse)
    print(loss.max())
    pos = np.unravel_index(np.argmax(loss), loss.shape)
    print(pos)


def train():
    Path = './data/all_data.csv'
    data_x, data_y = get_data(Path)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1)  # , random_state=0
    # 进行归一化处理
    env_state, scaler_x = data_normalization(x_train[:, :8])
    joblib.dump(scaler_x, filename='./model/scaler_x.pkl')
    x_train[:, :8] = env_state
    x_train = x_train.round(2)
    # print(x_train)
    # y_train,scaler_y=data_normalization(y_train)
    y_train = y_train.round(2)
    # joblib.dump(scaler_y, filename='./model/scaler_y.pkl')
    # print(y_train)
    Scaler_x = joblib.load('./model/scaler_x.pkl')
    x_test[:, :8] = Scaler_x.transform(x_test[:, :8])
    x_test = x_test.round(2)
    # print(x_test)
    # Scaler_y=joblib.load('./model/scaler_y.pkl')
    # y_test=Scaler_y.transform(y_test)
    y_test = y_test.round(2)
    # print(y_test)
    # model_train(x_train,x_test,y_train,y_test,scaler_y)
    model_train(x_train, x_test, y_train, y_test)


def main():
    train()


if __name__ == '__main__':
    main()
