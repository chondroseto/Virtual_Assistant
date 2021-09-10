import numpy as np
import wave
import scipy.io.wavfile as wav
import pickle
from audiolazy import lpc
from hmmlearn.hmm import GaussianHMM as hmm
#import os
#from hmmlearn import base
#from hmmlearn import _hmmc
#from scipy.special import logsumexp


def initialize(inputWav):
    rate, signal = wav.read(inputWav)  # returns a wave_read object , rate: sampling frequency
    return signal, rate

def lowPassFilter(signal, coeff=0.95):
    return np.append(signal[0],
                     signal[1:] - coeff * signal[:-1])  # y[n] = x[n] - a*x[n-1] , a = 0.97 , a>0 for low-pass filters

def preEmphasis(wav):
    signal, rate = initialize(wav)
    emphasizedSignal = lowPassFilter(signal)
    return emphasizedSignal, signal, rate


def lpc_hmm_train():
    print('========================================train========================================')
    for i in range(540):
        i = i + 1
        audio = 'data/data_wav/_ (' + str(i) + ').wav'

        emphasizedSignal, signal, rate= preEmphasis(audio)
        filt = lpc(emphasizedSignal, order=16)
        lpc_features = filt.numerator[1:]

        lpc_refeatures = np.reshape(lpc_features, (-1, 1))  # reshape to matrix
        #print('LPC Reshape Feature ke -', i, ' = ', lpc_refeatures)
        model = hmm(n_iter=10).fit(lpc_refeatures)  #hmm default

        with open("data/model_hmm/model_"+ str(i) + ".pkl", "wb") as file: pickle.dump(model, file)
        print('Create model and save model as model_',str(i))
    return lpc_features

def lpc_hmm_uji_all():
    whatsapp_match = 0
    linkedin_match = 0
    tokopedia_match = 0
    gmail_match = 0
    powerpoint_match = 0
    word_match = 0
    data_uji_per_class = 20
    print('========================================test all========================================')
    for i in range(120):
        i = i + 1

        #audio = 'data/data_uji_all/' +str(i)+ '.wav'
        audio = 'data/data_outdoor/_ (' + str(i) + ').wav'
        #audio = 'data/data_uji_orang_lain/_ (' + str(i) + ').wav'
        print(audio)

        if (i >= 1) and (i <= 20):
            label_actual = 'whatsapp'
        elif (i >= 21) and (i <= 40):
            label_actual = 'linkedin'
        elif (i >= 41) and (i <= 60):
            label_actual = 'tokopedia'
        elif (i >= 61) and (i <= 80):
            label_actual = 'gmail'
        elif (i >= 81) and (i <= 100):
            label_actual = 'powerpoint'
        elif (i >= 101) and (i <= 120):
            label_actual = 'word'

        print('actual label = ',label_actual)

        emphasizedSignal, signal, rate= preEmphasis(audio)
        filt = lpc(emphasizedSignal, order=16)
        lpc_features = filt.numerator[1:]
        lpc_refeatures = np.reshape(lpc_features, (-1, 1))  # reshape to matrix

        max_score = -float("inf")
        max_label = 0

        for j in range(540):
            j = j + 1

            model = pickle.load(open("data/model_hmm/model_" + str(j) + ".pkl", 'rb'))

            scr = model.score(lpc_refeatures) #method score menggunakan algorithm="forward"

            if scr > max_score:
                max_score = scr
                max_label = j

        if (max_label>=1)and(max_label<=90):
            label_predict = 'whatsapp'
        elif (max_label>=91)and(max_label<=180):
            label_predict = 'linkedin'
        elif (max_label>=181)and(max_label<=270):
            label_predict = 'tokopedia'
        elif (max_label>=271)and(max_label<=360):
            label_predict = 'gmail'
        elif (max_label>=361)and(max_label<=450):
            label_predict = 'powerpoint'
        elif (max_label>=451)and(max_label<=540):
            label_predict = 'word'

        if label_actual==label_predict:
            status = 'detected'
            if label_predict == 'whatsapp':
                whatsapp_match=whatsapp_match+1
            elif label_predict == 'linkedin':
                linkedin_match = linkedin_match + 1
            elif label_predict == 'tokopedia':
                tokopedia_match = tokopedia_match + 1
            elif label_predict == 'gmail':
                gmail_match = gmail_match + 1
            elif label_predict == 'powerpoint':
                powerpoint_match = powerpoint_match + 1
            elif label_predict == 'word':
                word_match = word_match + 1
        else:
            status = 'undetected'

        print("predicted data -", str(max_label), " label = ", label_predict," status = ",status)
        result = 'Detection Persentase rate= '+str((whatsapp_match+linkedin_match+tokopedia_match+gmail_match+powerpoint_match+word_match)/(data_uji_per_class*6)*100)+'% \n whatsapp = '+str((whatsapp_match/data_uji_per_class)*100)+'%\n linkedin = '+str((linkedin_match/data_uji_per_class)*100)+' %\n Tokopedia = '+str((tokopedia_match/data_uji_per_class)*100)+'%\n Gmail = '+str((gmail_match/data_uji_per_class)*100)+'%\n PowerPoint = '+str((powerpoint_match/data_uji_per_class)*100)+'%\n Word = '+str((word_match/data_uji_per_class)*100)+'%'
    return result

def lpc_hmm_uji_one(fname):
    print('========================================test========================================')
    for i in range(1):
        i = i + 1

        print(fname)

        if fname.find('whatsapp')>-1:
            label_actual = 'whatsapp'
        elif fname.find('linkedin') > -1:
            label_actual = 'linkedin'
        elif fname.find('tokopedia') > -1:
            label_actual = 'tokopedia'
        elif fname.find('gmail') > -1:
            label_actual = 'gmail'
        elif fname.find('powerpoint') > -1:
            label_actual = 'powerpoint'
        elif fname.find('word') > -1:
            label_actual = 'word'

        print('actual label = ',label_actual)

        emphasizedSignal, signal, rate= preEmphasis(fname)
        filt = lpc(emphasizedSignal, order=16)
        lpc_features = filt.numerator[1:]
        lpc_refeatures = np.reshape(lpc_features, (-1, 1))  # reshape reshape to matrix

        max_score = -float("inf")
        max_label = 0

        for j in range(540):
            j = j + 1

            model = pickle.load(open("data/model_hmm/model_"+ str(j) + ".pkl", 'rb'))

            scr = model.score(lpc_refeatures) #method score menggunakan algorithm="forward"

            if scr > max_score:
                max_score = scr
                max_label = j

        if (max_label>=1)and(max_label<=90):
            label_predict = 'whatsapp'
        elif (max_label>=91)and(max_label<=180):
            label_predict = 'linkedin'
        elif (max_label>=181)and(max_label<=270):
            label_predict = 'tokopedia'
        elif (max_label>=271)and(max_label<=360):
            label_predict = 'gmail'
        elif (max_label>=361)and(max_label<=450):
            label_predict = 'powerpoint'
        elif (max_label>=451)and(max_label<=540):
            label_predict = 'word'

        if label_actual==label_predict:
            status = 'detected'
        else:
            status = 'undetected'

        print("predicted data -",str(max_label)," label = ", label_predict," status = ",status)
        result = "predicted data -"+str(max_label)+" label = "+ label_predict+" status = "+status

        #if status == 'undetected':
        #    label_predict = ''

    return result,label_actual,label_predict,status

