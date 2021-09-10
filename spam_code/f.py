import librosa
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.io import wavfile
from scipy.fft import *
import data_processing


def tes1():
    np.set_printoptions(threshold=sys.maxsize)

    filename = "tes.wav"
    print("load success")
    fs = 8000
    clip, sample_rate = librosa.load(filename, sr=fs)
    n_fft = 1024
    start = 0
    hop_length = 512
    print("all parameter completed")
    # ccode to display spectogram
    X = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)


    f_hertz = np.fft.rfftfreq(n_fft, 1 / fs)
    print("proses")

    print("Frequency (Hz) : ", f_hertz)


def tes2(file, start_time, end_time):
    # Open the file and convert to mono
    sr, data = wavfile.read(file)
    if data.ndim > 1:
        data = data[:, 0]
    else:
        pass

    # Return a slice of the data from start_time to end_time
    dataToRead = data[int(start_time * sr / 1000): int(end_time * sr / 1000) + 1]

    # Fourier Transform
    N = len(dataToRead)
    yf = rfft(dataToRead)
    xf = rfftfreq(N, 1 / sr)

    # Uncomment these to see the frequency spectrum as a plot
    # plt.plot(xf, np.abs(yf))
    # plt.show()

    # Get the most dominant frequency and return it
    idx = np.argmax(np.abs(yf))
    freq = xf[idx]
    return freq,sr

def spectral_statistics(y: np.ndarray, fs: int) -> float:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()

    #print("Frequency (Hz) : ", freq)
    #print("amp (Hz) : ", amp)
    print("mean (Hz) : ", mean)

def spectral_properties(y: np.ndarray, fs: int) -> dict:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4

    result_d = {
        'mean': mean,
        #'sd': sd,
        #'median': median,
        #'mode': mode,
        #'Q25': Q25,
        #'Q75': Q75,
        #'IQR': IQR,
        #'skew': skew,
        #'kurt': kurt
    }

    return result_d


#tes,sample_rate = tes2("tes3.wav", 0, 36000)
#print(sample_rate)
#print(tes)
data_processing.LPCode("_ (49).wav")
#sr, data = wavfile.read("_ (288).wav")
#datas = data.flatten()
#properties=spectral_properties(y=datas,fs=sr)
#spectral_statistics(y=datas,fs=sr)
#print(properties)




