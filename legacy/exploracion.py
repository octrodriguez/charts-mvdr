import os
from glob import glob
from charts_utils.core.io import load_gpu_corr_frame, gpu_corr_to_visibilities
from charts_utils.cal.algorithms import utvec2mat
import numpy as np
from scipy.linalg import inv
from charts_utils.cal.algorithms import utvec2mat #[cite: 7]
from charts_utils.core.kotekan import (
    KOTEKAN_GPU_N_FREQ,
    KOTEKAN_GPU_N_RECEIVERS,
    KOTEKAN_GPU_N_POLARIZATIONS
)

# Buscar automáticamente el primer archivo .bin en tu carpeta de datos
directorio_datos = "../corr_24h_2608"
archivos = glob(os.path.join(directorio_datos, "*.bin"))

if not archivos:
    print(f"No se encontraron archivos .bin en {directorio_datos}")
    exit()

archivo_prueba = archivos[0]
print(f"Analizando archivo: {archivo_prueba}")
print("-" * 50)

# Cargar el frame crudo
corr = load_gpu_corr_frame(
    path=archivo_prueba,
    n_freq=KOTEKAN_GPU_N_FREQ,
    n_recv=KOTEKAN_GPU_N_RECEIVERS,
    n_pol=KOTEKAN_GPU_N_POLARIZATIONS
)
print(f"Dimensiones del tensor crudo de Kotekan: {corr.shape}")
print(f"Tipo de dato: {corr.dtype}")

# Convertir al formato matemático de visibilidades triangulares
visibilidades = gpu_corr_to_visibilities(corr, n_pol=KOTEKAN_GPU_N_POLARIZATIONS)
print(f"\nDimensiones de las visibilidades (triángulo superior): {visibilidades.shape}")
print(f"Tipo de dato: {visibilidades.dtype}")

# Inspeccionar un dato al azar
canal = 300
baseline = 0
print(f"\nValor de la visibilidad en el canal {canal}, línea de base {baseline}:")
print(visibilidades[canal, baseline])

# 5. Reconstruir la Matriz de Covarianza para el canal 300
n_ant = 32
vector_triangular = visibilidades[canal, :] # Extraemos las 528 visibilidades del canal 300

# utvec2mat convierte el vector de 528 elementos en una matriz hermitiana de 32x32[cite: 7]
R_obs = utvec2mat(n_ant, vector_triangular)

print(f"\nDimensiones de la matriz de covarianza R_obs: {R_obs.shape}")

# Verificamos la propiedad hermitiana comprobando la diagonal (debería ser puramente real)
print(f"Autocorrelación Antena 0 (elemento [0,0]): {R_obs[0, 0]}")
print(f"Autocorrelación Antena 1 (elemento [1,1]): {R_obs[1, 1]}")

# 1. Reconstruir la matriz de covarianza
n_ant = 32
vector_triangular = visibilidades[canal, :]
R_obs = utvec2mat(n_ant, vector_triangular) #[cite: 7]

print("\n--- PRUEBA DE ESTABILIDAD Y DIAGONAL LOADING ---")

# 2. Evaluar la salud de la matriz original
# El número de condición indica qué tan cerca está la matriz de ser singular (no invertible).
cond_original = np.linalg.cond(R_obs)
print(f"Número de condición original: {cond_original:.2e}")

# 3. Calcular el factor de Diagonal Loading (delta)
# Una buena práctica es usar una fracción diminuta (ej. 1e-5 o 1e-6) 
# de la potencia media del arreglo. np.trace suma la potencia de las 32 antenas.
potencia_media = np.real(np.trace(R_obs)) / n_ant
delta = 1e-5 * potencia_media

# 4. Aplicar Diagonal Loading
R_reg = R_obs + delta * np.eye(n_ant)

# 5. Evaluar la salud de la matriz regularizada
cond_reg = np.linalg.cond(R_reg)
print(f"Número de condición con Diagonal Loading: {cond_reg:.2e}")

# 6. Intentar la inversión matemática
try:
    R_inv = inv(R_reg)
    print("\n¡Éxito! La matriz se ha invertido y es numéricamente estable.")
    print(f"Dimensiones de la matriz inversa R_inv: {R_inv.shape}")
    
    # Comprobación de cordura algebraica: R * R^-1 debe ser muy cercana a la matriz Identidad
    identidad_aprox = R_reg @ R_inv
    error_identidad = np.linalg.norm(identidad_aprox - np.eye(n_ant))
    print(f"Error residual respecto a la matriz Identidad: {error_identidad:.2e}")
    
except np.linalg.LinAlgError as e:
    print(f"\nError crítico al invertir la matriz: {e}")