import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading

'''
iris 데이터셋 불러오기 
현재 불러온 데이터셋은 iris 데이터에 대한 label 정보를 포함하고 있지는 않음
'''
iris = datasets.load_iris()
print("Total Records: ", iris.data.shape)

''' random number를 생성하고 생성한 random number를 각 데이터 행마다 부여함'''
rng = np.random.RandomState(50120057)


''' 모델 정의 '''
''' 인접한 5개 데이터를 확인해서 knn 기반으로 LabelSpreading 진행, 30회 반복 '''
label_prop_model = LabelSpreading(kernel = 'knn'
                                  , n_neighbors = 5
                                  , alpha = 0.2
                                  , max_iter = 30
                                  , tol = 0.001)
''' 일부 데이터에 대한 label 강제 미부여, 30% 데이터에 대해서 label 삭제처리 하려함 '''
random_unlabeled_points = rng.rand(iris.target.shape[0]) <= 0.3 


''' unlabeled data처리하려는 데이터 표기 '''
unlabeled = np.copy(iris.target)
unlabeled[random_unlabeled_points] = -1


''' Label Spreading Model Fitting '''
label_prop_model.fit(iris.data, unlabeled)


''' unlabeled 처리된 데이터에 대한 Label Spreading 모델 기반 예측 '''
pred_lb = label_prop_model.predict(iris.data)


''' 모델로 인해 새롭게 부여된 label에 대한 정확도 검증 '''
print("Accuracy of Label Spreading: ",'{:.2%}'.format(label_prop_model.score(iris.data,pred_lb)))