import numpy as np
import os
import csv

def get_directories(data_dir):
    directories = []
    
    if os.path.exists(data_dir):
        #Buscamos todos los directorios de la ruta
        for d in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, d)):
                directories.append(d)
                
    else:
        print("path doesn't exists")
    return directories

root_path = os.getcwd()
data_path = os.path.join(root_path, "Datasets/ger/train/Images")
directorios = get_directories(data_path)

for d in directorios:
    label_path = os.path.join(data_path, d)
    archivos = os.listdir(label_path)
    
    #Recuperamos el subset de imagenes
    for a in archivos[:len(archivos)-1]:
        aux = a[6:11]
        n = int(aux)
        if n<15:
            os.remove(data_path + '/' + d + '/' + a)
    
    #Modificamos el groundt-truth
    csvfile = data_path + '/' + d + '/' + archivos[-1]
    res = []
    with open(csvfile, 'r') as f:
        lineas = f.read().splitlines()
        lineas.pop(0)
        for linea in lineas:
            n = int(linea[6:11])
            if n>=15:
                res.append(linea)

    os.remove(csvfile)

    res[:0] = ['Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId']

    with open(data_path + '/' + d + '/' + archivos[-1], 'w') as f:
        writer = csv.writer(f)
        for r in res:
            writer.writerow([r])
                
data_path = os.path.join(root_path, "Datasets/ger/test")  
dicIds = {}

with open(data_path + '/GT-final_test.csv', 'r') as f:
    lineas = f.read().splitlines()
    lineas.pop(0)
    for linea in lineas:
        l = linea.split(';')
        ident = l[-1]
        if ident not in dicIds:
            dicIds[ident] = 1
        else:
            dicIds[ident] += 1

for d in dicIds:
    dicIds[d] = int(dicIds[d]/10)

countDicIds = dicIds.copy()

for d in countDicIds:
    countDicIds[d] = 0

res = []
imgsSeleccionadas = []
with open(data_path + '/GT-final_test.csv', 'r') as f:
    lineas = f.read().splitlines()
    lineas.pop(0)
    for linea in lineas:
        l = linea.split(';')
        ident = l[-1]
        if countDicIds[ident] != dicIds[ident]:
            imgsSeleccionadas.append(l[0])
            res.append(linea)
            countDicIds[ident] += 1

res[:0] = ['Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId']

with open(data_path + '/fGT-final_test.csv', 'w') as f:
    writer = csv.writer(f)
    for r in res:
        writer.writerow([r])

data_path = os.path.join(data_path, "Images")
archivos = os.listdir(data_path)

for a in archivos:
    if a not in imgsSeleccionadas:
        os.remove(data_path + '/' + a)