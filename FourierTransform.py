from __future__ import print_function
import scipy.io.wavfile as wavfile
from scipy.io.wavfile import write
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt

FILE = 'test.wav'
FS_RATE, SIGNAL = wavfile.read(FILE) # Загружаем частоту дискретизации и сами точки
THRESHOLD = 800000
SIZE = len(SIGNAL)
N = SIGNAL.shape[0]
DENSITY = SIZE/N

def DFT(file):
    print ("Частота дискретизации: ", FS_RATE, "Гц ")
    print ("Количество сэмплов: ", N)
    secs = N / float(FS_RATE)
    print ("Длина аудиодорожки: ", secs, "с. ")
    Ts = 1.0/FS_RATE # Вычисляем период
    print ("Период: ", round(Ts, 2), "с. ")
    t = np.arange(0, secs, Ts) # Составляем массив с координатами точек во времени
    FFT = abs(scipy.fft.rfft(SIGNAL))
    FFT_side = FFT[range(N//2)] # Нам необходима лишь первая половина сигнала
    freqs = scipy.fftpack.fftfreq(SIGNAL.size, t[1]-t[0])
    freqs_side = freqs[range(N//2)] # Соответственно, лишь первая половина частот
    fft_freqs_side = np.array(freqs_side)
    fft_map = [[0, i] for i in range(0, 24001)] # Для удобства, снизим точность, округлив частоты до ближайшего целого и просуммировав
    for i, j in zip(FFT_side, fft_freqs_side):
        fft_map[round(j)][0] += i
    for i, freq in enumerate(sorted(fft_map, reverse=True)[0:10]): #Выведем 10 основных частот
        print(f'Частота {freq[1]} Гц, амплитуда {freq[0]}') 
    plt.subplot(221)
    p1 = plt.plot(t[1:], SIGNAL, "r") # График исходного звука
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    fr1 = plt.subplot(222)
    p2 = plt.plot(freqs_side, abs(FFT_side), "b") # Строим результат преобразования Фурье. Нам нужна только положительная часть, т.к. частота звука не может быть отрицательной, и отрицательная часть будет симметрична положительной.
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    fr1.set_yscale('log')
    plt.show(block=False)
    
    return(FFT, freqs_side, N, t)
def Filter(FFT, freqs_side, N, t):
    global THRESHOLD
    for i in range(len(FFT)):
        if FFT[i]<THRESHOLD: # Если значение ниже порога, уменьшаем его
            FFT[i] = FFT[i]/100
    return(FFT, freqs_side, N, t)

def IDFT(FFT, freqs_side, N, t):
    FFT_side = FFT[range(N//2)]
    fr2=  plt.subplot(224)
    p3 = plt.plot(freqs_side, abs(FFT_side), "b")
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')


    new_SIGNAL = scipy.fft.irfft(FFT)
    plt.subplot(223)
    p4 = plt.plot(t[1:], new_SIGNAL, "g") # График преобразования фурье после фильтрации 
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    fr2.set_yscale('log')
    plt.show(block=False)
    return(new_SIGNAL)

result = Filter(*DFT(FILE))
new_sig = IDFT(*result)
norm_new_sig = np.int32(new_sig * (2**32 / max(new_sig)))
write("clean.wav", 48000, norm_new_sig)

plt.show()