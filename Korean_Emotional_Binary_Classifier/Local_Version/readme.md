# 한국어 긍정, 부정 이진 분류기
## 모델파일(P_OR_N_MODEL)과 실행파일(Kor_Emo_Classifier(local).py)이 같은폴더에 존재해야함.


<img width="770" alt="스크린샷 2022-11-20 오후 6 36 01" src="https://user-images.githubusercontent.com/93313445/202895114-89c168df-f680-421c-9658-847d82025382.png">

## 초기화면

### 본인의 컴퓨터 환경에 맞게 설정 필요, 기본값은 CPU<br>
<img width="496" alt="스크린샷 2022-11-20 오후 6 37 04" src="https://user-images.githubusercontent.com/93313445/202895135-5aa85dd5-15ac-4225-8162-d2ea75127e2b.png">

## 문장입력

### 빈칸에 문장을 입력학 클릭을 누르면 분류 결과가 출력됨.<br>
<img width="563" alt="스크린샷 2022-11-20 오후 6 38 31" src="https://user-images.githubusercontent.com/93313445/202895188-193341dc-a37d-4cf4-8a1d-b3ca4ecf6670.png">

## 파일처리

### csv 형태로 된 데이터 셋의 경로(절대경로)를 입력받고 데이터셋을 읽고 판별. 결과를 실행 파일이 존재하는 폴더에 result.csv로 저장한다.<br>
<img width="711" alt="스크린샷 2022-11-20 오후 6 41 48" src="https://user-images.githubusercontent.com/93313445/202895297-db51ccb0-6e4c-40d3-8cec-6312082eb54e.png">

## 모델 평가

### csv 형태로 된 데이터 셋의 경로(절대경로)를 입력받고 데이터셋을 읽고 판별 및 평가. 정확도, AUC SCORE을 출력하고 ROC CURVE를 실행 파일이 존재하는 폴더에 plot.png로 저장한다.<br>

<img width="549" alt="스크린샷 2022-11-20 오후 6 50 35" src="https://user-images.githubusercontent.com/93313445/202895629-7efd4cbc-2fa1-4a02-9fd7-9f5a307edc60.png">
