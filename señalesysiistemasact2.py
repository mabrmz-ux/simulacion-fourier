import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------
#CONFIGURACIÓN INICIAL DE LA SIMULACIÓN
#--------------------------------------

fs = 1000    #Frecuencia de muestreo (Hz)
dt = 1/fs    #Intervalo de tiempo entre semana (s)
t_max = 2.0  #Duración total de la señal (s)
t = np.arange(-t_max/2, t_max/2, dt)   #Vector de de tiempo centrado
N = len(t)   #Número total de muestras

#Vector de frecuncias para la FFT
freqs = np.fft.fftfreq(N, dt)
freqs_shifted = np.fft.fftshift(freqs) #Centrar frecuencias en 0 Hz.

def compute_fft(signal):
    """Calcula la FFT centrada, su magnitud y fase"""
    X = np.fft.fft(signal)
    X_shifted = np.fft.fftshift(X)
    magnitude = np.abs(X_shifted) / N 
    phase = np.angle(X_shifted) 
    #Limpiar ruido de fase en magnitudes despreciables
    phase[magnitude<1e-4] = 0
    return X_shifted, magnitude, phase

#------------------------------------------------------------------
#1. SEÑALES ELEMENTALES Y SU TRANSFORMADA
#-------------------------------------------------------------------

#A) Pulso Rectangular (ancho T0 = 0.04s)
T0 = 0.4
rect = np.where(np.abs(t)<= T0/2, 1.0, 0.0)
_, mag_rect, phase_rect = compute_fft(rect)

#B) Función Escalón Continuo / Pulso Largo
step = np.where(t>= 0, 1.0, 0.0)
_, mag_step, phase_step = compute_fft(step)

#C) FUNCION SENOIDAL (f0 = 10Hz)
f0 = 10
sine = np.sin(2*np.pi*f0*t)
_, mag_sine, phase_sine = compute_fft(sine)

#GRAFICAR SEÑALES ELEMENTALES
fig, axs = plt.subplots(3, 3, figsize=(14, 9))
fig.suptitle("Señales Elementales: Tiempo vs Magnitud y Fase en Frecuencia", fontsize=14, fontweight="bold")

signals = [
    ("Pulso Rectangular", rect, mag_rect, phase_rect),
    ("Paso / Escalón", step, mag_step, phase_step),
    (f"Senoidal ({f0} Hz)", sine, mag_sine, phase_sine)
]

for i, (name, sig, mag, phase) in enumerate(signals):
    #DOMINIO DEL TIEMPO
    axs[i,0].plot(t, sig, "b")
    axs[i,0].set_title(f"{name}-Tiempo")
    axs[i,0].set_xlabel("Tiempo(s)")
    axs[i,0].set_ylabel("Amplitud")
    axs[i,0].grid(True)
    axs[i,0].set_xlim([-1,1])

    #MAGNITUD DE FRECUENCIA
    axs[i,1].plot(freqs_shifted, mag, "r")
    axs[i,1].set_title(f"{name}-Espectro de Magnitud")
    axs[i,1].set_xlabel("Frecuencia (Hz)")
    axs[i,1].set_ylabel("Magnitud")
    axs[i,1].grid(True)
    axs[i,1].set_xlim([-30,30])

    #FASE EN FRECUENCIA
    axs[i,2].plot(freqs_shifted, phase, "g")
    axs[i,2].set_title(f"{name}-Espectro de Fase")
    axs[i,2].set_xlabel("Frecuencia (Hz)")
    axs[i,2].set_ylabel("Fase (rad)")
    axs[i,2].grid(True)
    axs[i,2].set_xlim([-30,30])

plt.tight_layout()
plt.show()

#--------------------------------------------
#2. VERIFICACIÓN DE PROPIEDADES DE LA TF
#--------------------------------------------

#Propiedad 1: Linealidad -> TF{a*x1+b*x2}=a*X1+b*X2
a, b = 2.0, 1.5
f1, f2 = 5, 15
x1 = np.sin(2*np.pi*f1*t)
x2 = np.cos(2*np.pi*f2*t)
x_comb=a*x1+b*x2

_, mag_comb, _ = compute_fft(x_comb)
_, mag1, _ = compute_fft(x1)
_, mag2, _ = compute_fft(x2)
mag_comb_teorica = a*mag1 + b*mag2

#PROPIEDAD 2: DESPLAZAMIENTO EN EL TIEMPO -> x(t-t0) -> |X(f)| se mantiene, cambia la fase
t0=0.2
rect_shifted=np.where(np.abs(t-t0)<-T0/2,1.0,0.0)
_, mag_rect_shift, phase_rect_shift = compute_fft(rect_shifted)

#PROPIEDAD 3: ESCALAMIENTO EN FRECUENCIA / TIEMPO -> x(a*t) -> COMPRENSION EN TIEMPO = EXPANSIÓN EN FRECUENCIA
a_scale=2.0 #Pulso más angosto en tiempo
rect_scaled=np.where(np.abs(a_scale*t)<-T0/2,1.0,0.0)
_, mag_rect_scale, _ = compute_fft(rect_scaled)

#GRAFICAR VERIFICACION DE PROPIEDADES
fig_prop, axs_p=plt.subplots(3,2,figsize=(12,9))
fig_prop.subtitle("Demostración práctica de las Propiedades de la Transformada de Fourier", fontsize=14,fontweight="bold")

#LINEALIDAD
axs_p[0, 0].plot(t, x_comb, 'purple')
axs_p[0, 0].set_title('Superposición de Señales: 2·sin(2π·5t) + 1.5·cos(2π·15t)')
axs_p[0, 0].set_xlim([-0.5, 0.5])
axs_p[0, 0].grid(True)

axs_p[0, 1].plot(freqs_shifted, mag_comb, 'r', label='FFT Directa')
axs_p[0, 1].plot(freqs_shifted, mag_comb_teorica, 'k--', label='Suma de FFTs Individuales')
axs_p[0, 1].set_title('Propiedad de Linealidad (Espectro)')
axs_p[0, 1].set_xlim([-25, 25])
axs_p[0, 1].legend()
axs_p[0, 1].grid(True)

# DESPLAZAMIENTO    
axs_p[1, 0].plot(freqs_shifted, mag_rect, 'r', label='Original (t0=0)')
axs_p[1, 0].plot(freqs_shifted, mag_rect_shift, 'b--', label=f'Desplazada (t0={t0}s)')
axs_p[1, 0].set_title('Desplazamiento en Tiempo: Magnitud (Invariante)')
axs_p[1, 0].set_xlim([-20, 20])
axs_p[1, 0].legend()
axs_p[1, 0].grid(True)

axs_p[1, 1].plot(freqs_shifted, phase_rect_shift, 'g')
axs_p[1, 1].set_title('Desplazamiento en Tiempo: Cambio Lineal en Fase')
axs_p[1, 1].set_xlim([-20, 20])
axs_p[1, 1].grid(True)

# ESCALAMIENTO
axs_p[2, 0].plot(t, rect, 'r', label='Original (Ancho T0)')
axs_p[2, 0].plot(t, rect_scaled, 'b', label='Escalada (Ancho T0/2)')
axs_p[2, 0].set_title('Escalamiento Temporal (Compresión)')
axs_p[2, 0].set_xlim([-0.5, 0.5])
axs_p[2, 0].legend()
axs_p[2, 0].grid(True)

axs_p[2, 1].plot(freqs_shifted, mag_rect, 'r', label='Espectro Original')
axs_p[2, 1].plot(freqs_shifted, mag_rect_scale, 'b', label='Espectro Expandido')
axs_p[2, 1].set_title('Escalamiento en Frecuencia (Expansión Espectral)')
axs_p[2, 1].set_xlim([-20, 20])
axs_p[2, 1].legend()
axs_p[2, 1].grid(True)

plt.tight_layout()
plt.show()