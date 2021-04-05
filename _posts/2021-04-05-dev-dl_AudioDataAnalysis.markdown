---
published: false
---
---
layout: post
title: '[자료] DL을 이용한 오디오 데이터 분석'
subtitle: Audio Data Analysis #1
categories: dev
tags: dl audio signal python
comments: true
---
DL을 이용해 오디오 데이터를 분석하는 것에 대한 자료이며, Nagesh Singh Chauhan이 작성한 [Audio Data Analysis Using Deep Learning with Python (Part 1)](https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html){:target="_blank"}의 내용을 기반으로 작성된 것이다.

**예제 코드** Colab에서 실행 --> [Audio_Data_Analysis_Ex_1.ipynb](https://colab.research.google.com/github/AIWithDaddy/AIWithDaddy.github.io/blob/master/code/Audio_Data_Analysis_Ex_1.ipynb){:target="_blank"}

> import os, shutil
> from google.colab import drive
> drive.mount('/content/gdrive')
> os.chdir('gdrive/My Drive')
> %cd Test/Audio/
>
> #librosa --> 오디오 신호를 다루는 대표적인 라이브러리
> import librosa
> audio_data = 'rain.wav'
> x , sr = librosa.load(audio_data, sr=44100)
> print(type(x), type(sr))
>
> #IPython.display.Audio --> 주피터 노트북에서 음악 실행 등을 할 때 사용
> import IPython.display as ipd
> ipd.Audio(audio_data)
>
> #오디오 시각화 예
> %matplotlib inline
> import matplotlib.pyplot as plt
> import librosa.display
> plt.figure(figsize=(14, 5))
> librosa.display.waveplot(x, sr=sr)


