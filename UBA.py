import train_ai as ta
import os
# info, warning 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
import pandas as pd
import datetime
import json
import requests
import joblib

class AI:
    # user(user's name), type(mouse or resource)
    def __init__(self, model, user, type):
        self.model = model
        self.user = user
        self.type = type
    
    # change me 
    def load_model(self):
        model_dir = os.environ.get('MODEL_DIR', '')
        model_name = model_dir+self.user+'_model'
        if(self.type == 'mouse'):
            model_name += '_m.h5'
        else:
            model_name += '_r.h5'
        # model load
        self.model = tf.keras.models.load_model(model_name)
    
    # model 사용을 위한 pattern dataframe 전처리
    # pattern_df(extracted pattern, scaling 대상)
    def scale_pattern(self, pattern_df):
        model_dir = os.environ.get('MODEL_DIR', '')
        scaler_name = model_dir+self.user+'_scaler'
        if(self.type == 'mouse'):
            scaler_name += '_m.gz'
        else:
            scaler_name += '_r.gz'
        pattern_df = pattern_df.drop(['filename', 'label', 'time'], axis=1)
        pattern_df = pattern_df.fillna(0)
        scaler = joblib.load(scaler_name)
        sc_data = scaler.transform(pattern_df)
        return sc_data

    # AI를 통해 pattern owner 예측 확률 반환
    def predict(self, pattern_df):
        # pattern data scaling
        sc_data = self.scale_pattern(pattern_df)
        # model pred
        pred = self.model.predict(sc_data)
        return pred

    # 뉴비 모델 생성
    def train(self):
        ta.train(self.user, self.type)
    