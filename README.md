# ESL
统计学习课程作业

## CNN:
#### cnn1.py: LeNet+ReLU;
#### cnn2.py: 在cnn1的基础上加宽全连接层;
#### cnn3.py: 在cnn2的基础上修改卷积核;
#### cnn4.py: 在cnn3的基础上修改卷积核;
#### cnn5.py: 在cnn4的基础上加宽全连接层;
#### cnn6.py: 在cnn3的基础上加宽全连接层;
#### cnn7.py: 在cnn6的基础上加宽全连接层;
#### cnn8.py: 在cnn6的基础上加入Dropout层;
#### cnn9.py: 在cnn7的基础上加入Dropout层;
#### cnn10.py: 在cnn6的基础上修改卷积核;
#### cnn11.py: 在cnn10的基础上修改卷积核;
#### cnn12.py: 改用ResNet (4种不同的实现方式);
#### cnn13.py: 在cnn12的基础上加深ResNet;
#### cnn14.py: 在cnn13的基础上加深ResNet;
#### cnn15.py: 在cnn12的基础上加入数据增强 (3种不同的实现方式);

## KNN:
#### knn.py: 标准KNN,k=1,3,5,7,9;

## NN:
#### nn1.py: 784-800-15 (修改激活函数);
#### nn2.py: 784-2500-2000-1500-1000-500-15 (修改激活函数);
#### nn3.py: 在nn2的基础上修改数据预处理方式;

## SVM：
#### svm.py: 核函数(linear,rbf,poly,sigmoid);

## extra_train: 
#### extra_cnn.py: cnn15.py (data_aug_mode=1, ResBlock: mode=1) + extra train data(v0/v1)
#### extra_knn.py: knn.py (k=5) + extra train data
#### extra_nn.py: nn2.py (ReLU) + extra train data
#### extra_svm.py: svm.py (poly) + extra train data (无法得到预测结果)


## 运行指南：
#### CNN: python xxx.py [train / test] 默认为train; e.g. python cnn1.py | python cnn1.py train | python cnn1.py test
#### KNN: python knn.py (可在程序中设定K值)
#### NN:  python xxx.py [train / test] 默认为train; e.g. python nn1.py | python nn1.py train | python nn1.py test
#### SVM: python svm.py (可在程序中设定核函数)


## 最终选择模型：
#### CNN: extra_cnn.py 0.88214
#### KNN: knn.py (k=5) 0.64081
#### NN:  extra_nn.py 0.70340
#### SVM: svm.py (poly) 0.71733
