import matplotlib.pyplot as plt
import numpy as np
import wave
import scipy.io.wavfile as wav
from audiolazy.lazy_lpc import levinson_durbin
from pip._vendor.distlib.compat import raw_input
from scipy import signal
import scipy as sk
from audiolazy import *
import audiolazy.lazy_lpc
from audiolazy import lpc
from sklearn import preprocessing
import scipy.signal as sig
import scipy.linalg as linalg
import joblib# for saving the GMMs model
from sklearn.mixture import GaussianMixture #scikit-learn


def readWavFile(wav):
    # given a path from the keyboard to read a .wav file
    # wav = raw_input('Give me the path of the .wav file you want to read: ')
    inputWav = 'data' + wav
    return inputWav


# reading the .wav file (signal file) and extract the information we need
def initialize(inputWav):
    rate, signal = wav.read(readWavFile(inputWav))  # returns a wave_read object , rate: sampling frequency
    sig = wave.open(readWavFile(inputWav))
    # signal is the numpy 2D array with the date of the .wav file
    # len(signal) number of samples
    sampwidth = sig.getsampwidth()
    print('The sample rate of the audio is: ', rate)
    print('Sampwidth: ', sampwidth)
    return signal, rate


# implementation of the low-pass filter
def lowPassFilter(signal, coeff=0.97):
    return np.append(signal[0],
                     signal[1:] - coeff * signal[:-1])  # y[n] = x[n] - a*x[n-1] , a = 0.97 , a>0 for low-pass filters


def preEmphasis(wav):
    # taking the signal
    signal, rate = initialize(wav)
    # Pre-emphasis Stage
    preEmphasis = 0.97
    emphasizedSignal = lowPassFilter(signal)
    Time = np.linspace(0, len(signal) / rate, num=len(signal))
    EmphasizedTime = np.linspace(0, len(emphasizedSignal) / rate, num=len(emphasizedSignal))
    # plots using matplotlib
    '''plt.figure(figsize=(9, 7)) 
    plt.subplot(211, facecolor='darkslategray')
    plt.title('Signal wave')
    plt.ylim(-50000, 50000)
    plt.ylabel('Amplitude', fontsize=16)
    plt.plot(Time,signal,'C1')
    plt.subplot(212, facecolor='darkslategray')
    plt.title('Pre-emphasis')
    plt.ylim(-50000, 50000)
    plt.xlabel('time(s)', fontsize=10)
    plt.ylabel('Amplitude', fontsize=16)
    plt.plot(EmphasizedTime,emphasizedSignal,'C1')
    plt.show()'''
    return emphasizedSignal, signal, rate


def visualize(rate, signal):
    # taking the signal's time
    Time = np.linspace(0, len(signal) / rate, num=len(signal))
    # plots using matplotlib
    plt.figure(figsize=(10, 6))
    plt.subplot(facecolor='darkslategray')
    plt.title('Signal wave')
    plt.ylim(-40000, 40000)
    plt.ylabel('Amplitude', fontsize=16)
    plt.xlabel('Time(s)', fontsize=8)
    plt.plot(Time, signal, 'C1')
    plt.draw()


# plt.show()

def framing(fs, signal):
    # split the signal into frames
    windowSize = 0.025  # 25ms
    windowStep = 0.01  # 10ms
    overlap = int(fs * windowStep)
    frameSize = int(fs * windowSize)  # int() because the numpy array can take integer as an argument in the initiation
    numberOfframes = int(np.ceil(float(np.abs(len(signal) - frameSize)) / overlap))
    print('Overlap is: ', overlap)
    print('Frame size is: ', frameSize)
    print('Number of frames: ', numberOfframes)
    frames = np.ndarray(
        (numberOfframes, frameSize))  # initiate a 2D array with numberOfframes rows and frame size columns
    # assing samples into the frames (framing)
    for k in range(0, numberOfframes):
        for i in range(0, frameSize):
            if ((k * overlap + i) < len(signal)):
                frames[k][i] = signal[k * overlap + i]
            else:
                frames[k][i] = 0
    return frames, frameSize


def hamming(frames, frameSize):
    # Windowing with Hamming
    # Hamming implementation : W[n] = 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frameSize - 1))
    # y[n] = s[n] (signal in a specific sample) * w[n] (the window function Hamming)
    frames *= np.hamming(frameSize)
    '''plt.figure(figsize=(10, 6)) 
    plt.subplot(facecolor='darkslategray')
    plt.title('Hamming window')
    plt.ylim(-40000, 40000)
    plt.ylabel('Amplitude', fontsize=16)
    plt.xlabel('Time(ms)', fontsize=8)
    plt.plot(frames,'C1')
    plt.show()'''
    return frames


def autocorrelation(hammingFrames):
    correlateFrames = []
    for k in range(len(hammingFrames)):
        correlateFrames.append(np.correlate(hammingFrames[k], hammingFrames[k], mode='full'))
    # print 'Each frame after windowing and autocorrelation: \n',correlateFrames
    yolo = correlateFrames[len(correlateFrames) / 2:]
    return yolo


def levinsonDurbin(correlateFrames):
    # normalizedCF = preprocessing.normalize(correlateFrames, norm='l2')
    filt1 = levinson_durbin(correlateFrames, 13)
    print(filt1.numerator[1:])


def lpc_train():
    #folder = raw_input('Give the name of the folder that you want to read data: ')
    #amount = raw_input('Give the number of samples in the specific folder: ')
    for x in range(1, 10 + 1):
        wav = '/data_raw/'+'notepad_'+str(x)+'.wav'
        print(wav)
        emphasizedSignal, signal, rate = preEmphasis(wav)
        filt = lpc(emphasizedSignal, order=16)
        lpc_features = filt.numerator[1:]
        print('panjang data = ',len(lpc_features))
        print('LPC Feature ke -',x,' = ',lpc_features)
        np.save('data//data_raw//feature_' + str(x) + '.npy', lpc_features)
        print('LPC Feature di save pada feature_' + str(x) + '.npy')
    return lpc_features

def lpc_uji():
    #folder = raw_input('Give the name of the folder that you want to read data: ')
    #amount = raw_input('Give the number of samples in the specific folder: ')
    for x in range(1, 5 + 1):
        wav = '/data_uji/notepad_'+str(x)+'.wav'
        print(wav)
        emphasizedSignal, signal, rate = preEmphasis(wav)
        filt = lpc(emphasizedSignal, order=16)
        lpc_features = filt.numerator[1:]
        print('panjang data = ',len(lpc_features))
        print('LPC Feature ke -',x,' = ',lpc_features)
        np.save('data//data_uji//feature_' + str(x) + '.npy', lpc_features)
        print('LPC Feature di save pada feature_'+str(x)+'.npy')
    return lpc_features

# Defining a function which takes the MFCCs as a parameter(input) and returns the GMM(output)
def model_construct(data, n_components=1):
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='diag',
                          tol=0.001,
                          reg_covar=1e-06,
                          max_iter=100,
                          n_init=1,
                          init_params='kmeans',
                          warm_start=False,
                          verbose=0,
                          verbose_interval=10)
    gmm.fit(X=data)
    return gmm

def create_model():
    #folder = raw_input('Give the name of the folder that you want to read data: ')
    #amount = raw_input('Give the number of samples in the specific folder: ')
    for x in range(1, 10 + 1):
        feature = 'data/data_raw/feature_'+str(x)+'.npy'
        fture = np.load(feature) #load feature
        # Initializing gmm_<word> to the output of the gmm_construct function
        k = 8
        reshape_feature = np.reshape(fture, (-1, 2)) #reshape 1D array to 2D array
        model_data = model_construct(reshape_feature, n_components=k) #create model
        # Saving the model to disk
        joblib.dump(model_data, 'data//data_raw//model'+str(x)+'.pkl')
        print("model - ",str(x)," have been constructed and saved to disk")

def score_gmm(data, gmm):
    log_likelihood = gmm.score(X=data)
    return log_likelihood

def match():
    notepad_model1 = joblib.load('data//data_raw//model1.pkl')
    notepad_model2 = joblib.load('data//data_raw//model2.pkl')
    notepad_model3 = joblib.load('data//data_raw//model3.pkl')
    notepad_model4 = joblib.load('data//data_raw//model4.pkl')
    notepad_model5 = joblib.load('data//data_raw//model5.pkl')
    notepad_model6 = joblib.load('data//data_raw//model6.pkl')
    notepad_model7 = joblib.load('data//data_raw//model7.pkl')
    notepad_model8 = joblib.load('data//data_raw//model8.pkl')
    notepad_model9 = joblib.load('data//data_raw//model9.pkl')
    notepad_model10 = joblib.load('data//data_raw//model10.pkl')
    for x in range(1, 5 + 1):
        feature = 'data/data_uji/feature_'+str(x)+'.npy'
        mfccs = np.load(feature)
        reshape_feature = np.reshape(mfccs, (-1, 2))
        # score the MFCCs under each GMM

        scores = [notepad_model1.score(reshape_feature),notepad_model2.score(reshape_feature),notepad_model3.score(reshape_feature),notepad_model4.score(reshape_feature),notepad_model5.score(reshape_feature),notepad_model6.score(reshape_feature),notepad_model7.score(reshape_feature),notepad_model8.score(reshape_feature),notepad_model9.score(reshape_feature),notepad_model10.score(reshape_feature)]

        # if score 0 - 9 maka ke deteksi
        if (scores.index(max(scores)) > -1 and scores.index(max(scores)) < 10) :
            print('notepad ke - ' + str(x)+' memiliki score '+str(scores.index(max(scores)))+' terdeteksi notepad')
        else:
          print('notepad ke - ' + str(x)+' memiliki score '+str(scores.index(max(scores)))+' tidak terdeteksi')


def play():
    lpc_train()
    create_model()
    lpc_uji()
    match()
    # mylpc()


play()
