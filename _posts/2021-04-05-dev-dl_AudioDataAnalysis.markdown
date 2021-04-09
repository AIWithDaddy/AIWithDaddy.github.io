---
layout: post
title: '[자료] DL을 이용한 오디오 데이터 분석'
subtitle: Audio Data Analysis
categories: dev
tags: dl audio signal python
published: true
---
DL을 이용해 오디오 데이터를 분석하는 것에 대한 자료이며, Nagesh Singh Chauhan이 작성한 [Audio Data Analysis Using Deep Learning with Python (Part 1)](https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html){:target="_blank"}의 내용을 기반으로 작성된 것이다.

**1. 오디오 데이터를 다루는 주요 Python 라이브러리**

1) librosa --> 오디오 신호를 다루는 대표적인 라이브러리

2) IPython.display.Audio --> 주피터 노트북에서 음악 실행 등을 할 때 사용

3) 예제 코드 (Colab에서 실행 --> [Audio_Data_Analysis_Ex_1.ipynb](https://colab.research.google.com/github/AIWithDaddy/AIWithDaddy.github.io/blob/master/code/Audio_Data_Analysis_Ex_1.ipynb){:target="_blank"})

> #librosa<br>
> import librosa<br>
> audio_data = 'rain.wav'<br>
> x , sr = librosa.load(audio_data, sr=44100) --> wav 파일에서 오디오 데이터를 읽어들임 (numpy 배열과 샘플링 레이트 값을 리턴)<br>
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

  ![오디오 시각화](https://AIWithDaddy.github.io/assets/img/dev/dl/2021-04-05-dev-dl-AudioDataAnalysis_1.jpg)

**2. Spectrogram**

1) Spectrogram은 신호의 세기(loudness)를 표현하는 방법 중 하나이다.

2) x축은 시간, y축은 주파수이며, 보통 히트맵(heatmap)으로 표현된다.

3) 아래는 librosa를 이용해 spectogram을 생성하고 보여주는 예이다 (Colab에서 실행 --> [Audio_Data_Analysis_Ex_2.ipynb](https://colab.research.google.com/github/AIWithDaddy/AIWithDaddy.github.io/blob/master/code/Audio_Data_Analysis_Ex_2.ipynb){:target="_blank"}).

> import librosa, librosa.display<br>
> audio_data = 'rain.wav'<br>
> x , sr = librosa.load(audio_data, sr=44100)<br>
><br>
> import matplotlib.pyplot as plt<br>
> X = librosa.stft(x)<br>
> Xdb = librosa.amplitude_to_db(abs(X))<br>
> plt.figure(figsize=(14, 5))<br>
> librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')<br>
> plt.colorbar()<br>

  ![Spectrogram 표시](https://AIWithDaddy.github.io/assets/img/dev/dl/2021-04-05-dev-dl-AudioDataAnalysis_2.jpg)

**3. 임의의 오디오 신호 생성 예**

필요 시 아래 예와 같은 방법으로 임의의 오디오 신호를 생성할 수 있다 (Colab에서 실행 --> [Audio_Data_Analysis_Ex_3.ipynb](https://colab.research.google.com/github/AIWithDaddy/AIWithDaddy.github.io/blob/master/code/Audio_Data_Analysis_Ex_3.ipynb){:target="_blank"}).

> import numpy as np<br>
> sr = 22050 # sample rate<br>
> T = 5.0    # seconds<br>
> t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable<br>
> x = 0.5 * np.sin(2 * np.pi * 220 * t)# pure sine wave at 220 Hz<br>
> <br>
> #Playing the audio<br>
> import IPython.display as ipd<br>
> ipd.Audio(x, rate=sr) # load a NumPy array<br>
> <br>
> #Saving the audio<br>
> import soundfile<br>
> soundfile.write('test_220.wav', x, sr)<br>

**4. 오디오 신호 주요 특성(feature) 추출**

1) 오디오 데이터에 DL을 적용함에 있어 특정 응용이나 성능 향상 등을 위해 데이터 전처리나 특성 추출이 선행되는 경우가 많다.

**2) 분광 특성들(Spectral Features)**

분광 특성은 주파수 기반 특성으로 시간 기반의 원(raw) 신호 데이터를 푸리에 변환(Fourier Transfrom)을 이용해 주파수 기반의 데이터로 변경한 후, 특정 필요(해석/분석/판단)에 사용될 수 있는 다양한 특성 값을 산출할 수 있다. 예를 들면 기본 주파수(fundamental freqeuncy), 주파수 요소(frequency components), 분광 중심(spectral centroid), 분광 플럭스(spectral flux), 분광 밀도(spectral density), 분광 롤오프(spectral roll-off) 등이다.
  
- **Spectral Centroid**

- Sub 스팩트럼 에너지의 중심이 어떤 주파수에 위치해 있는지는 보여준다. 
- Sub Spectral Centroid는 소리의 밝기(brightness)와 관계가 있다. 
- Sub 이를 구하는 예는 다음과 같다 (Colab에서 실행 --> [Audio_Data_Analysis_Ex_4.ipynb](https://colab.research.google.com/github/AIWithDaddy/AIWithDaddy.github.io/blob/master/code/Audio_Data_Analysis_Ex_4.ipynb){:target="_blank"})

> import librosa, librosa.display<br>
> audio_data = 'rain.wav'<br>
> x , sr = librosa.load(audio_data, sr=44100)<br>
> <br>
> import sklearn<br>
> spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]<br>
> spectral_centroids.shape<br>
> <br>
> #Computing the time variable for visualization<br>
> import matplotlib.pyplot as plt<br>
> plt.figure(figsize=(12, 4))<br>
> frames = range(len(spectral_centroids))<br>
> t = librosa.frames_to_time(frames)<br>
> <br>
> #Normalising the spectral centroid for visualisation<br>
> def normalize(x, axis=0):<br>
>     return sklearn.preprocessing.minmax_scale(x, axis=axis)<br>
> <br>
> #Plotting the Spectral Centroid along the waveform<br>
> librosa.display.waveplot(x, sr=sr, alpha=0.4)<br>
> plt.plot(t, normalize(spectral_centroids), color='b')<br>

  ![Spectral Centrold](https://AIWithDaddy.github.io/assets/img/dev/dl/2021-04-05-dev-dl-AudioDataAnalysis_3.jpg)

- **Spectral Rolloff**
