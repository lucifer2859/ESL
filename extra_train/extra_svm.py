import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# public leaderboard
# poly      0.71733

train_path = '/home/dchen/dataset/ESL/quickdraw-data/train/'
extra_train_path = '/home/dchen/dataset/ESL/extra_training_data/data/'
test_path = '/home/dchen/dataset/ESL/quickdraw-data/released_test/'

category_mapping = {'airplane': 0, 'ant': 1, 'bear': 2, 'bird': 3, 'bridge': 4,
     'bus'     : 5, 'calendar': 6, 'car': 7, 'chair': 8, 'dog': 9,
     'dolphin' : 10, 'door': 11, 'flower': 12, 'fork': 13, 'truck': 14}

id_list = []
scaler = StandardScaler()

def load_data(data_path, train=True):
    data_pairs =[]

    # Get Train Data
    if (train):
        labels = os.listdir(data_path)
        for label in labels:    
            filepath = data_path + label
            filename  = os.listdir(filepath)
            for fname in filename:
                ffpath = filepath + "/" + fname
                data_pair = [ffpath, category_mapping[label]]
                data_pairs.append(data_pair)

        labels = os.listdir(extra_train_path)
        for label in labels:    
            filepath = extra_train_path + label
            filename  = os.listdir(filepath)
            for fname in filename:
                ffpath = filepath + "/" + fname
                data_pair = [ffpath, category_mapping[label]]
                data_pairs.append(data_pair)
    
        data_cnt = len(data_pairs)
        data_x = np.empty((data_cnt, 784), dtype="float32")
        data_y = []

        i = 0
        for data_pair in data_pairs:
            img = cv2.imread(data_pair[0], 0)
            img = cv2.resize(img, (28, 28))
            arr = np.asarray(img, dtype="float32").flatten()
            data_x[i, :] = arr
            data_y.append(data_pair[1])
            i += 1
        
        data_x = scaler.fit_transform(data_x)
        data_y = np.asarray(data_y)

        return data_x, data_y

    # Get Test Data
    filename = os.listdir(data_path)
    
    for fname in filename:
        ffpath = data_path + fname
        data_pairs.append(ffpath)
        id_list.append(fname.split('.')[0])  
    
    data_cnt = len(data_pairs)
    data_x = np.empty((data_cnt, 784), dtype="float32")
    data_y = []

    i = 0
    for path in data_pairs:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (28, 28))
        arr = np.asarray(img, dtype="float32").flatten()
        data_x[i, :] = arr  
        data_y.append(i)
        i += 1

    data_x = scaler.transform(data_x)
    data_y = np.asarray(data_y)
     
    return data_x, data_y


if __name__ == "__main__":
    X_train, Y_train = load_data(train_path, train=True)
    X_test, _ = load_data(test_path, train=False)

    svc = SVC(kernel='poly', class_weight='balanced',)

    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)

    param_grid = [{'kernel': ['poly'], 'C': c_range, 'gamma': gamma_range}]

    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    clf = grid.fit(X_train, Y_train)
    
    Y_pred = grid.predict(X_test)

    print('id,categories')

    for i in range(Y_pred.shape[0]):
        print('%s,%d' % (id_list[i], Y_pred[i]))
