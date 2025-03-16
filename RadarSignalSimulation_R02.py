import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
from scipy.signal import butter, filtfilt, freqz

def ChirpSignal(t,tau,PRI,fi,B,Np=1,delay=0):
    """ 
    La chirp(t,tao,PRI,fc,B,N=0):
    Genera un chirp periódico en un intervalo de tiempo t, ancho de pulso tau, 
    periodo PRI, frecuecia central fc, con crecimiento lineal en frecuencia para un 
    ancho de banda B y con N pulsos en intervalos de pulsos coherentes CPI = N*PRI. 
    N=0 por default te genera infinitos. 
    """
    y   = np.zeros(len(t),dtype=np.complex64)
    K   = B/tau
    ind_PRI = np.where((t >= delay) & (t <= PRI))[0]
    ind_tau = np.where((delay <= t) & (t <= delay+tau))[0]
    t = np.extract((delay<=t) & (t<=delay+tau),t)
    for i in range(Np):
        y[ind_tau[0]+ind_PRI[-1]*i:ind_tau[-1]+1+ind_PRI[-1]*i] = np.exp(1j*(2*np.pi*fi*(t-delay)+2*np.pi*K*((t-delay)**2)/2))

    return y

def ChirpSignal2(t, tau, PRI, fi, B, Np=1, delay=0):
    """ 
    Genera un chirp periódico en un intervalo de tiempo t, con ancho de pulso tau, 
    periodo PRI, frecuencia central fi, con crecimiento lineal en frecuencia para un 
    ancho de banda B y con N pulsos en intervalos de pulsos coherentes (CPI = N*PRI).
    Np=0 por defecto genera infinitos pulsos.
    """
    # Inicialización de la señal
    y = np.zeros(len(t), dtype=np.complex64)
    K = B / tau  # Constante de modulación

    # Índices del PRI y tau
    ind_PRI = np.where((t >= delay) & (t <= PRI))[0]
    ind_tau = np.where((delay <= t) & (t <= delay + tau))[0]
    
    # Solo tomamos el tiempo dentro del rango del chirp
    t_chirp = np.extract((delay <= t) & (t <= delay + tau), t)
    
    for i in range(Np):
        # Calcular los índices y aplicar el chirp a las posiciones correctas
        start_index = ind_tau[0] + ind_PRI[-1] * i
        end_index = ind_tau[-1] + 1 + ind_PRI[-1] * i
        
        # Asegurarse de que no exceda el tamaño de la señal
        if end_index <= len(t):
            y[start_index:end_index] = np.exp(1j * (2 * np.pi * fi * (t_chirp - delay) + 2 * np.pi * K * ((t_chirp - delay) ** 2) / 2))

    # Eliminar ceros al final fuera del intervalo de PRI
    y[len(t_chirp) + len(ind_PRI) * (Np - 1):] = 0

    return y


def ReceivedPower(Pt,Gtx, Grx,lamb,sigma,R):
    Prx = Pt*Gtx*Grx*(lamb**2)*sigma/(((4*np.pi)**3)*(R**4))
    return Prx

def GaussNoiseSignal(t,mu,sigma,Amplitude):
    noise_signal = Amplitude*np.random.normal(mu, sigma, len(t)).astype(np.float32)*(1+1j)   
    return noise_signal

def ButterLowPassFilter(Signal, CutOff, fs, Order=5):
    nyquist = 0.5 * fs
    normal_cutoff = CutOff / nyquist
    b, a = butter(Order, normal_cutoff, btype='low', analog=False)
    Signal = filtfilt(b, a, Signal)
    return Signal

def ADC(x, y, Bits,FS, fs):
    N = 2**Bits
    Q = FS/N
    #y_Q = y/max(y)
    y_Q = np.round(y/Q)*Q
    #Decimation = round(len(y_Q)/((x[-1]-x[0])*fs))
    #y_Q = y_Q[::Decimation]
    x_Q = x#[::Decimation]
    return x_Q, y_Q, N, Q 

def SignalAnalysis(x,y,fs,FigureName, SignalName=None, TimeUnits=[1,'s',1,'V'],FreqUnits=[1,'Hz',1,'dBm']):
    plt.figure(f"Signal {FigureName}", figsize=(12, 2.5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2, hspace=0.5)
    plt.subplot(121)
    plt.plot(x*TimeUnits[0], np.sqrt(max(y)*50*2)*y/max(y)*TimeUnits[2], linewidth = 0.7, label = f"{SignalName}")
    plt.title(f'Time Domain')
    plt.xlabel(f'Time({TimeUnits[1]})')
    plt.ylabel(f'Voltege({TimeUnits[3]})')
    #plt.legend(loc='upper right')
    plt.grid()

    N, y = len(y), np.concatenate((y, np.zeros(int(20*len(y)))))
    y = np.fft.fftshift(np.abs(np.fft.fft(y))**2/(len(y)))
    Y_dB = 10*np.log10(np.sqrt(y)/0.001)
    f = np.linspace(-fs/2,+fs/2,len(Y_dB))
    plt.subplot(122)
    plt.plot(f/FreqUnits[0],Y_dB*FreqUnits[2],linewidth = 0.7, label = f"{SignalName}")
    plt.title(f'Frequency Domain')
    plt.xlabel(f'Frequency({FreqUnits[1]})')
    plt.ylabel(f'Power({FreqUnits[3]})')
    #plt.legend(loc='upper right')
    plt.grid()


def main():
    # --------------------- Constantes Generales ---------------------
    c = 3e8                              # m/s, velocidad de la luz

    # --------------------- Parámetros de Radar ---------------------
    P_tx = 31e-3                         # W, potencia de transmisión
    G_tx = 10**(17/10)                   # Ganancia de transmisión (escala lineal)
    G_rx = 10**(17/10)                   # Ganancia de recepción (escala lineal)
    fi = 7.835e9                         # Hz, frecuencia inicial/central
    ff = 12.817e9                        # Hz, frecuencia final
    fc = (fi + ff) / 2                   # Hz, frecuencia central promedio
    lamb = c / fc                        # m, longitud de onda
    B = ff - fi                          # Hz, ancho de banda
    PRF = 25000                         # Hz, frecuencia de repetición de pulso
    PRI = 1 / PRF                        # s, intervalo de repetición de pulso
    DutyCycle = 100                      # %, ciclo de trabajo
    tau = DutyCycle * PRI / 100          # s, ancho de pulso

    # --------------------- Parámetros del Target y Señal ---------------------
    R_virtual = 387                    # m, rango simulado del objetivo
    delay = 2 * R_virtual / c            # s, retardo asociado a la distancia (ida y vuelta)
    EdgeCorner = 1                    # Parámetro geométrico del objetivo
    RCS = 4 * np.pi * (EdgeCorner**4) / (3 * (lamb**2))  # m², sección eficaz del objetivo
    #RCS = 11000
    # --------------------- Parámetros de Ruido ---------------------
    mu = 0.0001                            # Media para la generación de ruido gaussiano
    sigma = 1                            # Desviación estándar del ruido
    Noise = 0.001 * 10**(-100/10)         # W, potencia del ruido

    # --------------------- Parámetros de Simulación ---------------------
    Np = 1                               # Número de pulsos
    ti = 0                               # s, tiempo inicial de simulación
    tf = Np * PRI                # s, tiempo final de simulación
    fs_tx = 3 * ff                       # muestras/s, frecuencia de muestreo en transmisión
    #N = int(fs_tx * tf)                  # Número total de muestras
    #t = np.linspace(ti, tf, N)
    t = np.arange(ti, tf, 1/fs_tx)           # s, vector de tiempo de la simulación
    N = len(t)
    # --------------------- Parámetros de Procesamiento ---------------------
    G_LNA = 10**(22/10)                  # Ganancia del amplificador de bajo ruido (LNA)
    alpha = B / tau                      # Pendiente del chirp (Hz/s)
    fcutoff = 650e6  # Hz, frecuencia de corte del filtro pasa bajo
    Order = 5                            # Orden del filtro pasa bajo
    G_IFAmp = 10**(42/10)                # Ganancia del amplificador de video (IF)
    bits = 12                            # Resolución (bits) para la digitalización
    FS = 1                              # Escala para ADC (puede representar FullScale en otro contexto)
    fs_adc = 2*fcutoff                # Hz, frecuencia de muestreo para el ADC

        # ------------------------- Impresión de Parámetros y Resultados -------------------------
    print("\n----- Parámetros de Radar -----")
    print("c (m/s):", round(c, 3))
    print("P_tx (mW):", round(P_tx * 1000, 3))
    print("G_tx:", G_tx)
    print("G_rx:", G_rx)
    print("fi (GHz):", round(fi*1e-9, 3))
    print("ff (GHz):", round(ff*1e-9, 3))
    print("fc (GHz):", round(fc*1e-9, 3))
    print("lamb (cm):", round(lamb*100, 3))
    print("B (GHz):", round(B*1e-9, 3))
    print("PRF (kHz):", round(PRF*1e-3, 3))
    print("PRI (us):", round(PRI*1e6, 3))
    print("DutyCycle (%):", round(DutyCycle, 3))
    print("tau (us):", round(tau*1e6, 3))

    print("\n----- Parámetros del Target y Señal -----")
    print("R_virtual (m):", round(R_virtual, 3))
    print("delay (us):", round(delay*1e6, 3))
    print("EdgeCorner:", EdgeCorner)
    print("RCS (m²):", round(RCS, 3))
    print("RCS (dBm²):", round(10 * np.log10(RCS), 3))

    print("\n----- Parámetros de Ruido -----")
    print("mu:", round(mu, 3))
    print("sigma:", round(sigma, 3))
    print("Noise (W):", round(Noise, 3))

    print("\n----- Parámetros de Simulación -----")
    print("Np:", Np)
    print("ti (s):", round(ti, 3))
    print("tf (s):", round(tf, 3))
    print("fs_tx (MPSps):", round(fs_tx/1e6, 3))
    print("N (muestras totales):", N)

    print("\n----- Parámetros de Procesamiento -----")
    print("G_LNA:", round(G_LNA, 3))
    print("alpha (Hz/s):", round(alpha, 3))
    print("fcutoff (MHz):", round(fcutoff/1e6, 3))
    print("fs_adc (MHz):", round(fs_adc/1e6, 3))
    print("Order (filtro):", Order)
    print("G_IFAmp:", round(G_IFAmp, 3))
    print("bits:", bits)

    print("\n----- Resultados del Procesamiento de Señal -----")
    lll = ReceivedPower(P_tx, G_tx, G_rx, lamb, RCS, R_virtual)
    print("P_rx (uW):", lll*1e6)
    print("P_rx (dBm):", 10*np.log10(lll/0.001))
    print("Noise (uW):", round(Noise, 3))
    print("Noise (dBm):", 10*np.log10(Noise/0.001))
    print("N_adc (muestras ADC):", 2**bits)
    print("Q (parámetro adicional ADC) (uV):", round(FS/(2**bits)*1e6, 3))
    print("------------------------------------------------\n")


    # ------------------------- Procesamiento de la Señal -------------------------
    # Generación del chirp transmitido (sin retardo)
    ChirpSignalTx1 = P_tx * ChirpSignal(t, tau, PRI, fi, B, Np, 0)
    # Aplicación de la ganancia de transmisión
    ChirpSignalTx2 = G_tx * ChirpSignalTx1

    # Cálculo de la potencia recibida en función de P_tx, ganancias y RCS
    P_rx = ReceivedPower(P_tx, G_tx, G_rx, lamb, RCS, R_virtual)
    # Generación de la señal recibida con retardo y adición de ruido gaussiano

    ChirpSignalRx4 = (P_rx * ChirpSignal(t, tau, PRI, fi, B, Np, delay) + GaussNoiseSignal(t, mu, sigma, Noise))

    # Aplicación de la ganancia del LNA a la señal recibida
    ChirpSignalRx5 = G_LNA * ChirpSignalRx4

    # Mezcla (multiplicación) entre la señal transmitida y la señal recibida conjugada (beat signal)
    BeatSignal6 = ChirpSignalTx1 * np.conj(ChirpSignalRx5)

    # Amplificación y filtrado de la señal beat
    BeatSignal7 = G_IFAmp * ButterLowPassFilter(BeatSignal6, fcutoff, fs_tx, Order)

    # ------------------------- Digitalización (ADC) -------------------------
    # Escalado de la señal beat para la digitalización
    BeatSignal7_V = np.sqrt(max(BeatSignal7) * 50) * BeatSignal7 / max(BeatSignal7)
    # Proceso de digitalización (ADC)
    t_Q, BeatSignal7_V_Q, N_adc, Q = ADC(t, BeatSignal7_V, bits, FS, fs_adc)

        # ------------------------- Transformada de Fourier y Perfil de Rango -------------------------
    BeatSignal7_P_Q = (max(BeatSignal7_V_Q)**2/50) * BeatSignal7_V_Q/max(BeatSignal7_V_Q)

    N_val, y = len(BeatSignal7_P_Q), np.concatenate((BeatSignal7_P_Q, np.zeros(int(10 * len(BeatSignal7_P_Q)))))
    BeatSignal7_FFT = np.fft.fftshift(np.abs(np.fft.fft(BeatSignal7_P_Q))**2 / len(BeatSignal7_P_Q))
    BeatSignal7_FFT_dB = 10 * np.log10(np.sqrt(BeatSignal7_FFT) / 0.001)
    BeatFrequency = np.linspace(-fs_tx / 2, fs_tx / 2, len(BeatSignal7_FFT_dB))
    RangeProfile = c * BeatFrequency / (2 * alpha)  # Cálculo del rango a partir de la frecuencia de beat

    # ------------------------- Análisis de Señales -------------------------
    SignalAnalysis(t, np.real(ChirpSignalTx1), fs_tx, "Tx Signal", "Tx Signal", TimeUnits=[1e6, 'us', 1, 'V'], FreqUnits=[1e9, 'GHz', 1, 'dBm'])
    SignalAnalysis(t, np.real(ChirpSignalRx4), fs_tx, "Rx Signal", "Rx Signal", TimeUnits=[1e6, 'us', 1e6, 'uV'], FreqUnits=[1e9, 'GHz', 1, 'dBm'])
    SignalAnalysis(t, np.real(BeatSignal6), fs_tx, "Mixed Signals", "Beat Signal", TimeUnits=[1e6, 'us', 1e3, 'mV'], FreqUnits=[1e6, 'MHz', 1, 'dBm'])
    SignalAnalysis(t, np.real(BeatSignal7), fs_tx, "Beat Signal Filtered", "Beat Signal Filtered", TimeUnits=[1e6, 'us', 1e3, 'mV'], FreqUnits=[1e6, 'MHz', 1, 'dBm'])


    plt.figure('Quantification Signal')#, figsize=(12, 2.5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2, hspace=0.5)
    plt.step(t_Q*1e6, np.real(BeatSignal7_V_Q)*1000, where='post', label="Señal escalonada")
    plt.title(f'Time Domain')
    plt.xlabel(f'Time (µs)')
    plt.ylabel(f'Quantized Voltage (mV)')
    plt.legend(loc='upper right', labels=['Bits = 12, FS = 1V, N = 4096, Q = 244.141 µV'])  # Solo una leyenda
    plt.grid()

    
    # ------------------------- Ploteo -------------------------
    plt.figure('Range Profile')
    plt.plot(RangeProfile, np.real(BeatSignal7_FFT_dB), linewidth=0.7)
    plt.title('Range Profile')
    plt.xlabel('Range (m)')
    plt.ylabel('Power (dBm)')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    exit()

if __name__ == '__main__':
    main()
