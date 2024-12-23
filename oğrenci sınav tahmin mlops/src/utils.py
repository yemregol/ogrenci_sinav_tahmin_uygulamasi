import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(0,len(list(models))): # 0-7 döngü başlatır

            model = list(models.values())[i]  # i == 0 iken RandomForestRegressor() çalışır.
            para = param[list(models.keys())[i]]
            rc = RandomizedSearchCV(model,para,cv=3)

            rc.fit(X_train, y_train) # Bu search algoritmasını Train datalar üzerinden çalıştırdım
            model.set_params(**rc.best_params_) # Yukarıda çalışan search algoritmasının bulduğu en iyi parametreleri aldım
            model.fit(X_train, y_train) # En iyi parametrelerle eğitim yaptık

            y_test_pred = model.predict(X_test) # Tahmin değerlerini aldık

            test_model_score = r2_score(y_test,y_test_pred)


            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    