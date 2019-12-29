import os
import cv2
import numpy as np

# public leaderboard
# k   score
# 1   0.62718
# 3   0.64014
# 5   0.64081
# 7   0.64022
# 9   0.63733

train_path = '/home/dchen/dataset/ESL/quickdraw-data/train/'
test_path = '/home/dchen/dataset/ESL/quickdraw-data/released_test/'

category_mapping = {'airplane': 0, 'ant': 1, 'bear': 2, 'bird': 3, 'bridge': 4,
     'bus'     : 5, 'calendar': 6, 'car': 7, 'chair': 8, 'dog': 9,
     'dolphin' : 10, 'door': 11, 'flower': 12, 'fork': 13, 'truck': 14}

id_list = []
 
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
                
        data_x = data_x / 255
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
                
    data_x = data_x / 255
    data_y = np.asarray(data_y)
     
    return data_x, data_y

def distance(X_test, X_train):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    distances = np.zeros((num_test, num_train))
    dist1 = np.multiply(np.dot(X_test, X_train.T), -2)
    dist2 = np.sum(np.square(X_test), axis=1, keepdims=True)
    dist3 = np.sum(np.square(X_train.T), axis=0, keepdims=True)
    distances = np.sqrt(dist1 + dist2 + dist3)

    return distances

def predict(X_test, X_train, Y_train, k = 1):
    distances = distance(X_test, X_train)
    num_test = X_test.shape[0]
    Y_pred = np.zeros(num_test)
    for i in range(num_test):
        dists_min_k = np.argsort(distances[i])[:k]  
        y_labels_k = Y_train[dists_min_k] 
        Y_pred[i] = np.argmax(np.bincount(y_labels_k))

    return Y_pred


if __name__ == "__main__":
    X_train, Y_train = load_data(train_path, train=True)
    X_test, _ = load_data(test_path, train=False)

    Y_pred = predict(X_test, X_train, Y_train, k=9)

    print('id,categories')

    for i in range(Y_pred.shape[0]):
        print('%s,%d' % (id_list[i], Y_pred[i]))
