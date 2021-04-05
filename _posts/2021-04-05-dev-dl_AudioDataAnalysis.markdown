---
layout: post
title: '[자료] DL을 이용한 오디오 데이터 분석'
subtitle: Audio Data Analysis
categories: dev
tags: dl audio signal python
comments: true
published: true
---
DL을 이용해 오디오 데이터를 분석하는 것에 대한 자료이며, Nagesh Singh Chauhan이 작성한 [Audio Data Analysis Using Deep Learning with Python (Part 1)](https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html){:target="_blank"}의 내용을 기반으로 작성된 것이다.

**1. 오디오 데이터를 다루는 주요 Python 라이브러리**

1) librosa --> 오디오 신호를 다루는 대표적인 라이브러리

2) IPython.display.Audio --> 주피터 노트북에서 음악 실행 등을 할 때 사용

3) 예제 코드 (Colab에서 실행 --> [Audio_Data_Analysis_Ex_1.ipynb](https://colab.research.google.com/github/AIWithDaddy/AIWithDaddy.github.io/blob/master/code/Audio_Data_Analysis_Ex_1.ipynb){:target="_blank"})

> import os, shutil<br>
> from google.colab import drive<br>
> drive.mount('/content/gdrive')<br>
> os.chdir('gdrive/My Drive')<br>
> %cd Test/Audio/<br>
><br>
> #librosa --> wav 파일에서 오디오 데이터를 읽어들임 (numpy 배열과 샘플링 레이트 값을 리턴)<br>
> import librosa<br>
> audio_data = 'rain.wav'<br>
> x , sr = librosa.load(audio_data, sr=44100)<br>
> print(type(x), type(sr))<br>
><br>
> #IPython.display.Audio<br>
> import IPython.display as ipd<br>
> ipd.Audio(audio_data)<br>
><br>
> #오디오 시각화 예<br>
> %matplotlib inline<br>
> import matplotlib.pyplot as plt<br>
> import librosa.display<br>
> plt.figure(figsize=(14, 5))<br>
> librosa.display.waveplot(x, sr=sr)<br>
