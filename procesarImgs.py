import numpy as np
import os
#from cv2 import resize, imread
from PIL import Image
import csv
import scipy

## Creamos los métodos auxiliares ##

# Cargamos una imagen
def load_image(path, size):
    img = Image.open(path)
    return np.asarray(img.resize(size), dtype = np.float32)

# Devuelve una lista con los datos del fichero csv.
def read_csv(path, delimiter):
    file = open(path)
    title_csv = csv.reader(file, delimiter = delimiter)
    return list(title_csv)

# Devuelve una colección de directorios de la ruta especificada
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

# Devuelve las imagenes y etiquetas de la carpeta especificada
def read_images_labels_from_dir(directory, images, labels, shape, get_label_from_dir):
    
    #Buscamos fotos en el directorio
    for f in os.listdir(directory):
        # Cargamos archivos con extension .ppm 
        # Obviamos el color y lo cargamos como escala de grises y normalizamos el tamaño
        if f.endswith(".ppm"):
            image = load_image(os.path.join(directory, f), shape)
            images.append(image)
            if get_label_from_dir:
                labels.append(int(os.path.basename(directory)))
    
    return images, labels

# Devuelve una colección con las imágenes y los labels de la ruta
def readDataset(data_dir, shape, get_label_from_dir):
    images = []
    labels = []
    directories = get_directories(data_dir)
    
    images, labels = read_images_labels_from_dir(data_dir, images, labels, shape, get_label_from_dir)
    
    for d in directories:
                
        #Buscamos fotos en el directorio
        images_dir = os.path.join(data_dir, d)
        images, labels = read_images_labels_from_dir(images_dir, images, labels, shape, get_label_from_dir)
       
    return images, labels

# Imprime los tamaños de las colecciones
def print_size_dataset(images, labels, np_images, np_labels, environment):
    print("Total de imágenes (" + environment + "): ", len(images))
    print("Total de etiquetas (" + environment + "): ", len(set(labels)))
    print("Tamaño de imágenes: ", np_images.shape)
    print("Tamaño de etiquetas: ", np_labels.shape)

# Devuelve una lista con las categorías de las imágenes de prueba leídas del fichero csv.
def get_class_id_array(csv, class_column, first_is_header = True):
    labels = []
    for row in csv:
        if not first_is_header:
            labels.append(int(row[class_column]))
        else:
            first_is_header = False
    
    return labels

# Imprime los atributos de las imágenes de una etiqueta especifica.
def print_signals_attributes(label, images, source, titles):
    start = 0
    end = 0
    try:        
        start = source.index(label)
        end = start + source.count(label)
    except:
        print("label doesn't exist")
    
    if start < end:
        print("Signal: ", titles[label][1])
        for image in images[start:end]:
            print("shape: ", image.shape, "\tmin:", image.min(), "\tmax: ", image.max())

## Fijamos la dimensión de las imágenes ##
IMG_SHAPE = (32, 32)
print("Tamaño de las imágenes de entrada: ", IMG_SHAPE)
IMG_SHAPE_LEN = IMG_SHAPE[0] * IMG_SHAPE[1]
print("Vectorizando la entrada, sería de un tamaño: ", IMG_SHAPE_LEN)

## Obtenemos los directorios ##
root_path = os.getcwd()
labels_path = os.path.join(root_path, "Datasets/ger/labels.csv")
train_path = os.path.join(root_path, "Datasets/ger/train/Images")
test_info_path = os.path.join(root_path, "Datasets/ger/test/GT-final_test.csv")
test_path = os.path.join(root_path, "Datasets/ger/test/Images")

## Cargamos las imágenes de entrenamiento ##
images_train, labels_train = readDataset(train_path, IMG_SHAPE, True)

## Convertimos las listas a array numpy de float32 ##
np_images_train = np.asarray(images_train, dtype = np.float32)
np_labels_train = np.asarray(labels_train, dtype = np.int8)

np.save("x_train.npy", np_images_train)
np.save("y_train.npy", np_labels_train)

## Recuperamos los diferentes tipo de señales de tráfico que se van a clasificar ##
titles = read_csv(labels_path, ",")

## Se imprime información de los datos cargados ##
print_size_dataset(images_train, labels_train, np_images_train, np_labels_train, "train")
print("Total de categorías: ", len(titles))

## Cargamos las imágenes de evaluación ##
images_test, labels_test = readDataset(test_path, IMG_SHAPE, False)

test_info  = read_csv(test_info_path, ";")
labels_test = get_class_id_array(test_info, 7)

## Convertimos las listas a array numpy de float32 ##
np_images_test = np.asarray(images_test, dtype = np.float32)
np_labels_test = np.asarray(labels_test, dtype = np.int8)

np.save("x_test.npy", np_images_test)
np.save("y_test.npy", np_labels_test)

## Imprimimos información de los datos cargados ##
print_size_dataset(images_test, labels_test, np_images_test, np_labels_test, "test")
print_signals_attributes(10, images_train, labels_train, titles)