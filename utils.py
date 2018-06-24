import pandas as pd
import config
import os
from sklearn.preprocessing import StandardScaler
import pickle

datafile = os.path.join(config.DATADIR,'application_train.csv')
# data = pd.read_csv(datafile)
#
# print ( set(list(data['OCCUPATION_TYPE'])) )
# print (data['OCCUPATION_TYPE'].dtype)
#
# a = data.values
# print(type(a))

def process_data(datafile):
    data = pd.read_csv(datafile)
    columns_name = list(data.columns)
    print(len(columns_name))
    filter_names = []
    for name in columns_name:
        if (data[name].dtype == 'float32' or data[name].dtype == 'float64') and name != 'TARGET':
            filter_names.append(name)

    filter_data = data[filter_names]
    filter_data = filter_data.fillna(0)

    return filter_data.values

def normlize_data_train(data,save_file):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    with open(save_file,'wb') as f:
        pickle.dump(scaler,f)
    return scaled_data

def normlize_data_test(data,save_file):
    with open(save_file,'rb') as f:
        scaler = pickle.load(f)
        scaled_data = scaler.transform(data)

    return scaled_data



data = process_data(datafile)

scaled_data = normlize_data_test(data,'./scaler.p')

print(data.shape)
print(scaled_data.shape)

print(data[:,0])
print(scaled_data[:,0])
