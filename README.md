# Actividad Formativa 2: Simulación y Análisis de Señales con la Transformada de Fourier

Este proyecto realiza la simulación y análisis de señales en el dominio del tiempo y de la frecuencia utilizando **Python** y el algoritmo de la **Transformada Rápida de Fourier (FFT)**.

## 📋 Descripción del Proyecto

El objetivo es analizar cómo se representan las señales elementales en ambos dominios y verificar experimentalmente las propiedades fundamentales de la Transformada de Fourier:

1. **Señales Elementales Analizadas:**
   - **Pulso Rectangular:** Visualización del espectro en forma de función sinc.
   - **Función Escalón / Paso:** Análisis de la componente de corriente directa (DC).
   - **Función Senoidal ($10\text{ Hz}$):** Identificación de deltas de Dirac espectrales en $\pm 10\text{ Hz}$.

2. **Propiedades Verificadas:**
   - **Linealidad:** La FFT de la combinación lineal equivale a la suma de las FFTs individuales.
   - **Desplazamiento en el Tiempo:** El retardo en tiempo modifica la fase linealmente sin alterar la magnitud del espectro.
   - **Escalamiento Temporal:** La compresión en tiempo genera una expansión en el dominio de la frecuencia.

---

## 🛠️ Requisitos e Instalación

Asegúrate de contar con Python 3 instalado. Puedes instalar las librerías necesarias ejecutando en tu terminal:

```bash
pip install numpy matplotlib scipy