import tensorflow as tf
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import numpy as np
from kobert_transformers import get_tokenizer
from tkinter import *
import csv
import datetime
import time
import sys
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

tokenizer = get_tokenizer()

path = resource_path('/Users/minjunjung/Desktop/최종제출/P_OR_N_MODEL')
model = BertForSequenceClassification.from_pretrained(path)



def model_call(mode):
    device = torch.device("cpu")
    name = "CPU(DEFAULT)"
    if mode == 0:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            name = "MPS(APPLE SILICON)"
        else:
            device = torch.device("cpu")
            name = "MPS 사용 불가. CPU로 설정됨."
    elif mode == 1:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = "CUDA(NVIDIA)"
        else:
            device = torch.device("cpu")
            name = "CUDA 사용 불가. CPU로 설정됨."

    else:
        device = torch.device("cpu")
        name = "CPU(DEFAULT)"
    return [device,name]

# 시그모이드
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# 테스트셋 데이터 전처리
def convert_input_data(sentences):
    sentences = ["[CLS] " + str(sentences) + " [SEP]"]

    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 150

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks

# 모델 사용
def Emotion_Binary_Classification(sentences,device):
    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    # 시그모이드 함수, 확률 판단
    pred = sigmoid(logits)

    # 결과 판단
    result = np.argmax(pred)

    return [result, pred]


def start(s,g,m):
    a,p = Emotion_Binary_Classification(s,g)

    if m == 0:
        if a == 0:
            ans = s + " 는(은) 부정문입니다."
        else:
            ans = s + " 는(은) 긍정문입니다."

        return ans
    elif m == 1:
        return a
    else:
        return [a,p]


# 프론트엔드 GUI

window = Tk()

window.title("Korean Emotional Binary Classification")
window.geometry("480x600")
window.resizable(False, False)
g = [0,'cpu']
def device_mode(a):
    g[0],g[1] = model_call(a)
    model.to(g[0])
    print(g[1])
    label_dname.configure(text="Device: " + g[1])

label_dname = Label(window)
label_dname.place(x=110, y=100, width=240)

f = [1]

def function_mode(a):
    if a == 0:
        def btncmd():
            sentence = e.get()
            r_ans = start(sentence, g[0],0)
            label3 = Label(window)
            label3.place(x=10, y=380)
            label3.configure(text=str(r_ans))
            e.delete(0, END)

        label1 = Label(window)
        label1.configure(text="감정을 분류할 문장을 입력 하세요")
        label1.place(x=115, y=250, width=240)

        label2 = Label(window)
        label2.place(x=10, y=350)
        label2.configure(text="                                               ")

        e = Entry(window, width=50)
        e.place(x=5, y=280)
        e.insert(0, '')

        btn = Button(window, text="클릭", command=btncmd)
        btn.place(x=200, y=330)

        label3 = Label(window)
        label3.place(x=10, y=380)
        label3.configure(text="                                                                                                          ")

        label4 = Label(window)
        label4.place(x=10, y=430)
        label4.configure(text="                                                                        ")

        label5 = Label(window)
        label5.place(x=10, y=470)
        label5.configure(text="                                                                                 ")
    elif a == 1:
        def btncmd():
            label2.configure(text="분류 완료(./result.csv)                   ")

            df = pd.DataFrame([],columns=['Sentences','Emotions'])

            loc = e.get()
            csv_test = pd.read_csv(loc)
            for i in range(len(csv_test)):
                ans = start(csv_test['Sentences'][i], g[0],1)
                df.loc[i] = [csv_test['Sentences'][i],int(ans)]
            e.delete(0, END)
            df.to_csv('./result.csv')

        label1 = Label(window)
        label1.configure(text="분류할 csv 파일의 경로를 입력하세요")
        label1.place(x=115, y=250, width=240)

        label2 = Label(window)
        label2.place(x=10, y=350)
        label2.configure(text="                                                    ")

        e = Entry(window, width=50)
        e.place(x=5, y=280)
        e.insert(0, '')

        btn = Button(window, text="클릭", command=btncmd)
        btn.place(x=200, y=330)

        label3 = Label(window)
        label3.place(x=10, y=380)
        label3.configure(text="                                                                                       ")

        label4 = Label(window)
        label4.place(x=10, y=430)
        label4.configure(text="                                                      ")

        label5 = Label(window)
        label5.place(x=10, y=470)
        label5.configure(text="                                                                                         ")
    elif a == 2:
        # 이진 분류 결과(0 or 1)이 담길 정답 리스트
        test_ans = []

        # 이진 분류 결과의 확률([0~1,0~1])이 담길 정답 리스트
        test_prob = []
        # 정확도 계산 함수
        def flat_accuracy(preds, labels):

            pred_flat = np.argmax(preds, axis=1).flatten()
            labels_flat = labels.flatten()

            return np.sum(pred_flat == labels_flat) / len(labels_flat)

        # 시간 표시 함수
        def format_time(elapsed):

            # 반올림
            elapsed_rounded = int(round((elapsed)))

            # hh:mm:ss으로 형태 변경
            return str(datetime.timedelta(seconds=elapsed_rounded))

        def btncmd():
            label2 = Label(window)
            label2.place(x=10, y=350)

            loc = e.get()
            csv_eval = pd.read_csv(loc)
            labels = csv_eval.Emotions

            t0 = time.time()
            for i in range(len(csv_eval)):
                a,p = start(csv_eval['Sentences'][i], g[0],2)
                test_ans.append(a)  # 결과 값은 test_ans에 저장
                test_prob.append(p)  # 확률 값은 test_prob에 저장
            label2.configure(text="분류에 소요된 시간: : " + str(format_time(time.time() - t0)))
            e.delete(0, END)

            t_prob = []
            cnt = 0
            for i in test_prob:
                t_prob.append(i[0][1])
            for i in range(len(test_ans)):
                if test_ans[i] == labels[i]:
                    cnt += 1
            auc_score = roc_auc_score(labels, t_prob)
            label3.configure(text="정확도(ACC): " + str(round(cnt/len(test_ans), 6)))
            label4 = Label(window)
            label4.place(x=10, y=430)
            label4.configure(text="AUC Score: " + str(round(auc_score, 6)))

            # ROC CURVE PLOT
            fpr, tpr, thresholds = roc_curve(labels, t_prob)

            roc = pd.DataFrame({'FPR(Fall-out)': fpr, 'TPRate(Recall)': tpr, 'Threshold': thresholds})

            plt.scatter(fpr, tpr)
            plt.title('model ROC curve')
            plt.xlabel('FPR(Fall-out)')
            plt.ylabel('TPR(Recall)');
            plt.plot(fpr, tpr, 'r--')
            plt.savefig('./plot.png')
            label5 = Label(window)
            label5.place(x=10, y=470)
            label5.configure(text="플롯 저장 완료(./plot.png)")

        label1 = Label(window)
        label1.configure(text="모델 평가 데이터(csv)의 경로를 입력하세요")
        label1.place(x=115, y=250, width=240)

        label3 = Label(window)
        label3.place(x=10, y=380)
        label3.configure(text="                                                                    ")

        e = Entry(window, width=50)
        e.place(x=5, y=280)
        e.insert(0, '')

        btn = Button(window, text="클릭", command=btncmd)
        btn.place(x=200, y=330)


label_s2 = Label(window)
label_s2.configure(text="사용할 Device를 고르세요")

btn1 = Button(window, text="MPS(APPLE SILICON)", command=lambda: device_mode(0))
btn2 = Button(window, text="CUDA(NVIDIA)", command=lambda: device_mode(1))
btn3 = Button(window, text="CPU(DEFAULT)", command=lambda: device_mode(2))
label_s2.pack()

btn1.place(x=10,y=40)
btn2.place(x=195,y=40)
btn3.place(x=335,y=40)

f_btn1 = Button(window, text="문장 입력", command=lambda: function_mode(0))
f_btn2 = Button(window, text="파일 처리(csv)", command=lambda: function_mode(1))
f_btn3 = Button(window, text="모델 평가(csv)", command=lambda: function_mode(2))

f_btn1.place(x=30,y=150)
f_btn2.place(x=170,y=150)
f_btn3.place(x=340,y=150)

f_mode = f[0]

window.mainloop()