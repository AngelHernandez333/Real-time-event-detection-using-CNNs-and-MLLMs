import numpy as np

# Cargar el archivo .npz
frames = np.load("../Database/NWPU_IITB/GT/gt.npz")
frames_dict = dict(frames)  # Convertir a diccionario
# print(len(frames_dict.keys()))  # Imprimir las claves del diccionario
# 310 videos
# +134
# ----------
# 444
import os

rute = "/home/ubuntu/Database/CHAD DATABASE/CHAD_Meta/anomaly_labels"
files = os.listdir(rute)  # Listar archivos en el directorio
print(len(files))  # Imprimir el número de archivos
actual_files = []
for i in range(len(files)):
    if files[i].endswith("1.npy"):
        actual_files.append(files[i])  # Filtrar archivos .npy

for i in range(len(actual_files)):
    array = np.load(file=rute + "/" + actual_files[i])  # Cargar cada archivo .npy
    name = actual_files[i].split(".")[0]  # Obtener el nombre del archivo sin extensión
    frames_dict[name] = array

np.savez("../Database/NWPU_IITB/GT/gt_ALL.npz", **frames_dict)  # Guardar
