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
    
    # 기존 모델 고려하지 않고 새롭게 재학습 진행
    def retrain_all(self, modify_files, modify_labels):
        # modify wrong label
        control = Control()
        for i in range(len(modify_files)):
            file = modify_files[i]
            label = modify_labels[i]
            control.modify_label(file, self.type, label)
        # train
        ta.train(self.user, self.type)
    
    # 지금으로부터 과거(days+1)일간 데이터로 기존 모델 활용하여 재학습 진행
    def retrain_part(self):
        self.load_model()
        ta.retrain(self.user, self.type, self.model, days=2)

class Control:
    # feature file labeling 수정
    def modify_label(filename, type, label):
        if(type == 'mouse'):
            feature_file = os.environ.get('M_FEATURE_FILE', '')
        else:
            feature_file = os.environ.get('R_FEATURE_FILE', '')
        df = pd.read_csv(feature_file)
        df.loc[df['filename']==filename, 'label'] = label
        df.to_csv(feature_file, index=False)
    
    # request로 보낼 데이터 만들기
    def make_sendData(issue, user, label, pred_m, pred_r, file_m, file_r):
        time = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S.%f")
        sendData = {'user':user, 
        'time':time, 
        'mouse_prediction': str(round(pred_m,5)),
        'resource_prediction': str(round(pred_r,5)), 
        'type':issue, 
        'label':label, 
        'mouse_file': file_m, 
        'resource_file': file_r
        }
        return sendData

    # CERT 팀에게 경고 알리는 함수
    def alert_to_CERT(data):
        url = os.environ.get('CERT_URL','')
        data = json.dumps(data)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        res = requests.post(url, data=data, verify=False , headers=headers)    # verify는 SSL인증서 체크 관련 내용
        return res

    # save user's pattern data
    # @@ 동작 확인 필요
    def recv(user, type):
        if(type == 'mouse'):
            url = os.environ.get('M_PATTERN_URL','')
            extract_df = pd.read_json(url,orient='records')
            if(extract_df.shape == (1,3)):  # idle인 경우
                return extract_df
            feature_file = os.environ.get('M_FEATURE_FILE', '')
        else:
            url = os.environ.get('R_PATTERN_URL','')
            extract_df = pd.read_json(url,orient='records')
            feature_file = os.environ.get('R_FEATURE_FILE', '')
        
        if(os.path.isfile(feature_file)):
            extract_df.to_csv(feature_file, mode='a', index=False, header=None)
        else:
            extract_df.to_csv(feature_file, mode='w', index=False)
        return extract_df
