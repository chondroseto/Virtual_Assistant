import numpy as np
import scipy.io.wavfile as wav
from audiolazy import lpc
import matplotlib.pyplot as plt
import librosa as lr

def initialize(inputWav):
    rate, signal = wav.read(inputWav)  # returns a wave_read object , rate: sampling frequency
    #sig = wave.open(inputWav)
    #print('The sample rate of the audio is: ', rate)

    return signal, rate

def lowPassFilter(signal, coeff=0.95):
    return np.append(signal[0],
                     signal[1:] - coeff * signal[:-1])  # y[n] = x[n] - a*x[n-1] , a = 0.97 , a>0 for low-pass filters

def preemphasis(wav):
    signal, rate = initialize(wav)
    emphasized = lowPassFilter(signal)
    print('sample rate = ', rate)
    print('panjang signal = ', len(signal))
    print('signal = ',signal)
    #print('panjang emp',len(emphasized))
    print('emphasized',emphasized)
    signals = signal.flatten() #change 2D to 1D array
    #print('panjang signal = ', len(signals))
    #print('signal = ', signals)
    #print('The 1000 data signal of the audio is: ', signal[0:1000])
    #print('The 1000 data emp of the audio is: ', emphasized[0:1000])
    #print('The 1000 data signal of the audio is: ', signal[1000:2000])
    #print('The 1000 data signal of the audio is: ', signal[2000:3000])
    #print('The 1000 data signal of the audio is: ', signal[3000:4000])
    #print('The 1000 data signal of the audio is: ', signal[4000:5000])
    #print('The 1000 data signal of the audio is: ', signal[5000:6000])
    #print('The 1000 data signal of the audio is: ', signal[6000:7000])
    #print('The 1000 data signal of the audio is: ', signal[7000:8000])
    #print('The 1000 data signal of the audio is: ', signal[8000:9000])
    #print('The 1000 data signal of the audio is: ', signal[9000:10000])
    #print('The 1000 data signal of the audio is: ', signal[11000:12000])
    #print('The 1000 data signal of the audio is: ', signal[12000:13000])
    #print('The 1000 data signal of the audio is: ', signal[14000:15000])
    #print('The 1000 data signal of the audio is: ', signal[15000:16000])
    #print('The 1000 data signal of the audio is: ', signal[16000:17000])
    #print('The 1000 data signal of the audio is: ', signal[17000:18000])
    #print('The 1000 data signal of the audio is: ', signal[18000:19000])
    #print('The 1000 data signal of the audio is: ', signal[19000:20000])
    #print('preemphasis = ',emphasized)
    #print('The 1000 data emphasized signal of the audio is: ', emphasizedSignal[0:1000])
    # edit n_components=1 #hidden state, covariance_type='diag' #diag = each state uses a diagonal covariance matrix, n_iter=10 #iteration, tol=1e-2 #Convergence threshold.
    #proses ini tidak bekerja jika variabel berisi array 2D hanya bekerja untuk 1D

    x = 0
    for i in signals:
        new = emphasized[x] + i
        emphasized[x] = new
        x = x + 1
    return emphasized,signal,rate

def spectral_statistics(y: np.ndarray, fs: int) -> float:
    z=y.flatten()
    spec = np.abs(np.fft.rfft(z))
    freq = np.fft.rfftfreq(len(z), d=1 / fs)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    jumlah = freq.sum()

    #print("Frequency (Hz) : ", freq)
    #print("amp (Hz) : ", amp)
    return mean,freq

def dbfft(x, fs, win=None, ref=32768):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
        win: vector containing window samples (same length as x).
             If not provided, then rectangular window is used by default.
        ref: reference value used for dBFS scale. 32768 for int16 and 1 for float

    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """

    N = len(x)  # Length of input sequence

    if win is None:
        win = np.ones(1, N)
    if len(x) != len(win):
            raise ValueError('Signal and window must be of the same length')
    x = x * win

    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)

    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / np.sum(win)

    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag/ref)

    return freq, s_dbfs

def LPCode(fname):
    emphasized,signal,rate = preemphasis(fname)
    signals=signal.flatten()
    s_properties=spectral_statistics(signals,rate)
    e_properties=spectral_statistics(emphasized, rate)
    print('preemphasis = ',emphasized)
    #print('The 1000 data emphasized signal of the audio is: ', emphasized[0:1000])
    filt = lpc(emphasized, order=16)
    lpc_features = filt.numerator[1:]
    print('panjang data = ', len(lpc_features))
    #print('rata-rata sinyal freq(Hz) = ', s_properties)
    #print('rata-rata emp freq(Hz) = ', e_properties)
    print('LPC Feature ke = ', lpc_features)

    return lpc_features


def Visual_waktu(fname):
    audio, sfreq = lr.load(fname)

    time = np.arange(0, len(audio)) / sfreq
    fig, ax = plt.subplots()
    ax.plot(time, audio,color='black')
    ax.set(xlabel="Time(s)", ylabel="Sound Amplitude")
    plt.title("Audio")
    plt.show()

    emphasized = lowPassFilter(audio)

    #time = np.arange(0, len(emphasized)) / sfreq
    #fig, ax = plt.subplots()
    #ax.plot(time, emphasized)
    #ax.set(xlabel="Time(s)", ylabel="Sound Amplitude")
    #plt.title("Audio After Preemphasis")
    #plt.show()
    x = 0
    for i in audio:
        new = emphasized[x] + i
        emphasized[x] = new
        x = x + 1

    # print('The 1000 data emphasized signal of the audio is: ', emphasizedSignal[0:1000])
    time = np.arange(0, len(emphasized)) / sfreq
    fig, ax = plt.subplots()
    ax.plot(time, emphasized,color='black')
    ax.set(xlabel="Time(s)", ylabel="Sound Amplitude")
    plt.title("preemphasis")
    plt.show()

    filt = lpc(emphasized, order=16)
    lpc_features = filt.numerator[1:]

    time = np.arange(0, len(lpc_features)) / sfreq
    fig, ax = plt.subplots()
    ax.plot(time, lpc_features,color='black')
    ax.set(xlabel="Time(s)", ylabel="Sound Amplitude")
    plt.title("LPC Feature")
    plt.show()

    return lpc_features


