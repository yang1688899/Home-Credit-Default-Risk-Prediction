import pandas as pd
import config
import os
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.utils import shuffle


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

    labels_data = data['TARGET']
    return filter_data.values,labels_data.values

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

def get_data_batch(features,labels,batch_size=128):
    num_samples = len(features)
    features,labels = shuffle(features,labels)

    while True:
        for offset in range(0,num_samples,batch_size):
            features,labels = shuffle(features,labels)
            batch_features = features[offset,offset+batch_size]
            batch_labels = labels[offset,offset+batch_size]

            yield batch_features,batch_labels



