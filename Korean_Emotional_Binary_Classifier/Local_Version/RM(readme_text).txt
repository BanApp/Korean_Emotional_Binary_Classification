사용한 언어 및 라이브러리는 아래와 같습니다. 


<언어> 

1. python3
Version: 3.7.14 (default, Sep 8 2022, 00:06:44)
GCC 7.5.0

===============================<라이브러리>===============================

1. transformers
Version: 4.23.1
License: Apache

2. torch
Version: 1.12.1+cu113
License: BSD-3

3. tensorflow)
Version: 2.9.2
License: Apache 2.0

4. keras
Version: 2.9.0
License: Apache 2.0

5. scikit-learn
Version: 1.0.2
License: new BSD

6. pandas
Version: 1.3.5
License: BSD-3-Clause

7. numpy
Version: 1.21.6
License: BSD

8. matplotlib
Version: 3.2.2
License: PSF
==========================================================================



==================================<사용법>==================================


tkinter를 통해서 구현한 GUI기반 한국어 감정 이진 분류기
Kor_Emo_Classifier(local).py(로컬 전용)

1. 파이썬이 설치되고, 위의 라이브러리가 모두 존재하는 가상환경 혹은 PC에서 실행 가능하다.

2. 터미널에서 패턴인식_프로젝트/소스코드_및_프로그램/Kor_Emo_Classifier/Local_Version 폴더로 이동한다.

3. python Kor_Emo_Classifier.py 명령어를 통해서 실행한다.

4. IDE 에디터를 통해서 직접 실행도 가능하다.

5. 모델(P_OR_N_MODEL) 과 실행파일(Kor_Emo_Classifier.py)는 같은 폴더에 존재해야 한다.

6. csv데이터의 경로를 입력할때는 절대 경로로 입력한다.

7. 로컬 버전 실행이 불가능시 패턴인식_프로젝트/소스코드_및_프로그램/Kor_Emo_Classifier/Colab_Version 폴더에 존재하는 Kor_Emo_Classifier(colab).ipynb 와 현재 폴더에 존재하는 모델(P_OR_N_MODEL)을 이용.

==========================================================================



=================================<데이터 셋>=================================

1. 패턴인식_프로젝트/소스코드_및_프로그램/데이터셋/test_data.csv = 평가를 위해 따로 분리해둔 데이터 셋

2. 패턴인식_프로젝트/소스코드_및_프로그램/데이터셋/train_data.csv = 모델 학습에 사용된 데이터 셋

3. 패턴인식_프로젝트/소스코드_및_프로그램/데이터셋/원본_데이터.csv = test_data.csv + train_data.csv

===========================================================================
