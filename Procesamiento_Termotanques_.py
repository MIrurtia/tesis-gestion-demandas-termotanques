##########################################################################################################

########################################## SCRIPT TERMOTANQUES ###########################################

##########################################################################################################

### Importar bibliotecas usadas

import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
import shutil
from sklearn.cluster import KMeans
from scipy import stats
import numpy as np
import math

##########################################################################################################

################################### PARTE B.2. PROCESAMIENTO DE DATOS ####################################

##########################################################################################################

### Parte 1 - Colección de archivos Termotanques_min

## Parte 1.1.1

# Ruta de la carpeta original
carpeta_origen = "D:/TT/Termotanques"

# Ruta de la nueva carpeta
carpeta_destino = "D:/TT/Termotanques_1.0"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Listar todos los archivos en la carpeta de origen
archivos_origen = os.listdir(carpeta_origen)

# Iterar sobre cada archivo en la carpeta de origen
for archivo in archivos_origen:
    # Crear la ruta completa del archivo de origen
    ruta_origen = os.path.join(carpeta_origen, archivo)

    # Leer el archivo .csv sin encabezados
    df = pd.read_csv(ruta_origen, sep='\t', header=None, names=['ID_cliente', 'Imei', 'Fecha_unix', 'Potencia'])

    # Convertir la columna 'Fecha_unix' a tipo datetime
    df['Fecha_unix'] = pd.to_datetime(df['Fecha_unix'], unit='s')

    # Redondear al minuto más cercano
    df['Fecha_redondeada'] = df['Fecha_unix'].dt.round('T')

    # Crear una nueva columna con la fecha formateada
    df['Fecha_formateada'] = df['Fecha_redondeada'].dt.strftime('%d/%m/%Y %H:%M:%S')

    # Seleccionar solo las columnas necesarias
    df_resultado = df[['Fecha_formateada', 'Potencia']]

    # Crear la ruta completa del archivo de destino
    ruta_destino = os.path.join(carpeta_destino, archivo)

    # Guardar el nuevo archivo .csv en la carpeta de destino
    df_resultado.to_csv(ruta_destino, index=False, header=False, sep='\t')

## Parte 1.1.2

# Ruta de la carpeta original
carpeta_origen = "D:/TT/Termotanques_1.0"

# Ruta de la nueva carpeta
carpeta_destino = "D:/TT/Termotanques_1"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Listar todos los archivos en la carpeta de origen
archivos_origen = os.listdir(carpeta_origen)

# Definir la zona horaria original (UTC)
zona_horaria_original = pytz.utc

# Definir la zona horaria deseada (UTC-3)
zona_horaria_deseada = pytz.timezone('America/Argentina/Buenos_Aires')

# Iterar sobre cada archivo en la carpeta de origen
for archivo in archivos_origen:
    # Crear la ruta completa del archivo de origen
    ruta_origen = os.path.join(carpeta_origen, archivo)

    # Leer el archivo .csv sin encabezados
    df = pd.read_csv(ruta_origen, sep='\t', header=None, names=['Fecha_hora', 'Potencia'])

    # Convertir la columna 'Fecha_hora' a tipo datetime
    df['Fecha_hora'] = pd.to_datetime(df['Fecha_hora'], format='%d/%m/%Y %H:%M:%S')

    # Convertir la columna 'Fecha_hora' a la zona horaria original (UTC)
    df['Fecha_hora_utc'] = df['Fecha_hora'].dt.tz_localize(zona_horaria_original)

    # Convertir la columna 'Fecha_hora_utc' a la zona horaria deseada (UTC-3) y formatear sin el huso horario
    df['Fecha_hora_utc-3'] = df['Fecha_hora_utc'].dt.tz_convert(zona_horaria_deseada).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Seleccionar solo las columnas necesarias
    df_resultado = df[['Fecha_hora_utc-3', 'Potencia']]
    
    # Agrupar por 'Fecha_hora_utc-3' y quedarse con el primer valor de 'Potencia' para cada grupo
    df_resultado = df.groupby('Fecha_hora_utc-3').first().reset_index()
    
    # Filtrar las dos columnas que quiero
    df_resultado = df_resultado[['Fecha_hora_utc-3', 'Potencia']]

    # Crear la ruta completa del archivo de destino
    ruta_destino = os.path.join(carpeta_destino, archivo)

    # Guardar el nuevo archivo .csv en la carpeta de destino
    df_resultado.to_csv(ruta_destino, index=False, header=False, sep='\t')

# Eliminar la carpeta Termotanques 1.0 y su contenido
carpeta_eliminar = "D:/TT/Termotanques_1.0"
shutil.rmtree(carpeta_eliminar)

## Parte 1.2

# Ruta de la carpeta original
carpeta_origen = "D:/TT/Termotanques_1"

# Ruta de la nueva carpeta
carpeta_destino = "D:/TT/Termotanques_min"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Listar todos los archivos en la carpeta de origen
archivos_origen = os.listdir(carpeta_origen)

# Iterar sobre cada archivo en la carpeta de origen
for archivo in archivos_origen:
    # Crear la ruta completa del archivo de origen
    ruta_origen = os.path.join(carpeta_origen, archivo)

    # Leer el archivo .csv sin encabezados
    df = pd.read_csv(ruta_origen, sep='\t', header=None, names=['Fecha_hora_utc-3', 'Potencia'])

    # Convertir la columna 'Fecha_hora_utc-3' a tipo datetime
    df['Fecha_hora_utc-3'] = pd.to_datetime(df['Fecha_hora_utc-3'], format='%Y-%m-%d %H:%M:%S')

    # Obtener la fecha mínima y máxima del dataframe
    fecha_minima = df['Fecha_hora_utc-3'].min()
    fecha_maxima = df['Fecha_hora_utc-3'].max()

    # Crear un rango de fechas y horas completas entre las horas mínima y máxima
    rango_completo = pd.date_range(start=fecha_minima, end=fecha_maxima, freq='T')

    # Crear un nuevo dataframe con el rango completo
    df_completo = pd.DataFrame({'Fecha_hora_utc-3': rango_completo})

    # Fusionar los dataframes para rellenar los valores faltantes con NaN
    df_resultado = pd.merge(df_completo, df, on='Fecha_hora_utc-3', how='left')

    # Rellenar los valores de potencia nulos con NaN
    df_resultado['Potencia'] = df_resultado['Potencia'].fillna(pd.NA)

    # Crear la ruta completa del archivo de destino
    ruta_destino = os.path.join(carpeta_destino, archivo)

    # Guardar el nuevo archivo .csv en la carpeta de destino
    df_resultado.to_csv(ruta_destino, index=False, header=False, sep='\t')

print("Parte 1 completada. Se obtuvo la colección de archivos Termotanques_min")

##########################################################################################################

### Parte 2 - Colección de archivos Encendidos

## Parte 2.1

# Ruta de la carpeta con los archivos
carpeta_datos = "D:/TT/Termotanques_min"
carpeta_destino = "D:/TT/Termotanques_E"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta_datos)

# Iterar sobre cada archivo en la carpeta
for archivo in archivos:
    # Crear la ruta completa del archivo de origen
    ruta_origen = os.path.join(carpeta_datos, archivo)

    # Leer el archivo .csv sin encabezados
    df = pd.read_csv(ruta_origen, sep='\t', header=None, names=['Fecha_hora_utc-3', 'Potencia'])

    # Convertir la columna 'Fecha_hora_utc-3' a tipo datetime
    df['Fecha_hora_utc-3'] = pd.to_datetime(df['Fecha_hora_utc-3'], format='%Y-%m-%d %H:%M:%S')

    # Identificar los encendidos
    encendidos = (df['Potencia'] > 0).astype(int)
    cambios_encendido = encendidos.diff()

    # Iniciar variables para seguimiento del encendido actual
    en_encendido = False
    inicio_encendido = None
    fin_encendido = None

    # Listas para almacenar datos de encendidos
    encendidos_numero = []
    duracion_encendidos = []
    inicio_encendidos = []
    fin_encendidos = []

    # Iterar sobre los cambios de estado
    for i, cambio in enumerate(cambios_encendido):
        if cambio == 1:
            # Comienzo de un encendido
            en_encendido = True
            inicio_encendido = df['Fecha_hora_utc-3'][i]
        elif cambio == -1:
            # Fin de un encendido
            en_encendido = False
            fin_encendido = df['Fecha_hora_utc-3'][i - 1]
            
            # Verificar si inicio_encendido es None antes de calcular la duración
            if inicio_encendido is not None:
                duracion_encendido = (fin_encendido - inicio_encendido).total_seconds() / 60

                # Almacenar datos del encendido
                encendidos_numero.append(len(encendidos_numero) + 1)
                duracion_encendidos.append(duracion_encendido)
                inicio_encendidos.append(inicio_encendido)
                fin_encendidos.append(fin_encendido)
            else:
                print(f"Advertencia: Inicio de encendido es None para el archivo {archivo}")

    # Crear DataFrame con los resultados
    df_encendidos = pd.DataFrame({
        'Encendido N': encendidos_numero,
        'Duracion en minutos': duracion_encendidos,
        'Hora de inicio': inicio_encendidos,
        'Hora de fin': fin_encendidos
    })

    # Sumar 1 a cada valor de la columna 'Duracion en minutos'
    df_encendidos['Duracion en minutos'] += 1

    # Guardar el nuevo archivo .csv en la carpeta de destino
    ruta_destino = os.path.join(carpeta_destino, archivo.replace(".csv", "_E.csv"))
    df_encendidos.to_csv(ruta_destino, index=False)

## Parte 2.2

# Ruta de la carpeta con los archivos
carpeta_origen = "D:/TT/Termotanques_E"
carpeta_destino = "D:/TT/Encendidos"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta_origen)

# Iterar sobre cada archivo en la carpeta
for archivo in archivos:
    # Crear la ruta completa del archivo de origen
    ruta_origen = os.path.join(carpeta_origen, archivo)

    # Leer el archivo .csv
    df = pd.read_csv(ruta_origen, parse_dates=['Hora de inicio', 'Hora de fin'])

    # Filtrar las filas donde la duración en minutos sea menor a 4 minutos
    df_filtrado = df[(df['Duracion en minutos'] > 4)]

    # Crear la ruta del archivo de destino
    ruta_destino = os.path.join(carpeta_destino, archivo)

    # Guardar el DataFrame filtrado como un nuevo archivo .csv
    df_filtrado.to_csv(ruta_destino, index=False)

print("Parte 2 completada. Se obtuvo la colección de archivos Encendidos")

##########################################################################################################

### Parte 3 - Colección de archivos Termotanques_min_encendidos_uso

## Parte 3.1

# Ruta de la carpeta con los archivos de encendidos
carpeta_encendidos = "D:/TT/Encendidos"

# Crear una lista para almacenar los resultados de los tiempos de recuperación
resultados_1 = []

# Obtener la lista de archivos en la carpeta
archivos_encendidos = os.listdir(carpeta_encendidos)

# Iterar sobre cada archivo en el directorio
for archivo_encendidos in archivos_encendidos:
    # Crear la ruta completa del archivo de encendidos
    ruta_encendidos = os.path.join(carpeta_encendidos, archivo_encendidos)

    # Leer el archivo .csv con los encendidos
    df_encendidos = pd.read_csv(ruta_encendidos)
        
    # Extraer la columna 'Duracion en minutos'
    duraciones = df_encendidos['Duracion en minutos'].values
        
    # Realizar el clustering con k-means para dos clusters
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(duraciones.reshape(-1, 1))
        
    # Asignar los clusters a cada duración
    df_encendidos['Cluster'] = kmeans.labels_
    
    # Valores que corresponden a cada cluster
    duraciones_cluster_1 = (df_encendidos[df_encendidos['Cluster'] == 0]['Duracion en minutos']).values
    duraciones_cluster_2 = (df_encendidos[df_encendidos['Cluster'] == 1]['Duracion en minutos']).values
    
    # Calcular la moda de cada cluster
    moda_cluster_1 = stats.mode(duraciones_cluster_1).mode
    moda_cluster_2 = stats.mode(duraciones_cluster_2).mode
        
    # Ordenar las modas para que Cluster 1 sea el menor y Cluster 2 el mayor
    if moda_cluster_1 > moda_cluster_2:
        moda_cluster_1, moda_cluster_2 = moda_cluster_2, moda_cluster_1

    # Calcular el mínimo de cada cluster
    min_cluster_1 = duraciones_cluster_1.min()
    min_cluster_2 = duraciones_cluster_2.min()

    # Añadir el resultado a la lista
    resultados_1.append([archivo_encendidos, moda_cluster_1])

# Crear dos DataFrame con el resultado
df_resultados_1 = pd.DataFrame(resultados_1, columns=['Termotanque', 'Tiempo de recuperacion'])

# Eliminar el sufijo '_E' de la columna 'Termotanque'
df_resultados_1['Termotanque'] = df_resultados_1['Termotanque'].str.replace('_E.csv', '')

# Guardar los DataFrame de resultados en dos archivos .csv
df_resultados_1.to_csv("tiempos_de_recuperacion.csv", index=False)

## Parte 3.2

# Ruta de la carpeta con los archivos de consumo
carpeta_consumo = "D:/TT/Termotanques_min"

# Ruta del archivo con los tiempos de recuperación por termotanque
ruta_tiempos_recuperacion = "D:/TT/tiempos_de_recuperacion.csv"

# Leer la tabla de tiempos de recuperación
df_tiempos_recuperacion = pd.read_csv(ruta_tiempos_recuperacion)

# Crear una carpeta para almacenar los resultados
carpeta_resultados = "D:/TT/Termotanques_min_encendidos_uso"
if not os.path.exists(carpeta_resultados):
    os.makedirs(carpeta_resultados)

# Obtener la lista de archivos en la carpeta de consumo
archivos_consumo = os.listdir(carpeta_consumo)

# Iterar sobre cada archivo en la carpeta de consumo
for archivo_consumo in archivos_consumo:
    # Crear la ruta completa del archivo de consumo
    ruta_consumo = os.path.join(carpeta_consumo, archivo_consumo)

    # Obtener el nombre del termotanque
    nombre_termotanque = archivo_consumo.replace(".csv", "")

    # Verificar si hay datos para el termotanque actual en la tabla de tiempos de recuperación
    if nombre_termotanque in df_tiempos_recuperacion['Termotanque'].values:
        # Obtener el tiempo mínimo de uso para el termotanque actual
        tiempo_recuperacion = df_tiempos_recuperacion[df_tiempos_recuperacion['Termotanque'] == nombre_termotanque]['Tiempo de recuperacion'].values[0]

        # Leer el archivo .csv con los datos de consumo
        df_consumo = pd.read_csv(ruta_consumo, sep='\t', header=None, names=['Fecha_hora', 'Potencia'])

        # Convertir la columna 'Fecha_hora' a tipo datetime
        df_consumo['Fecha_hora'] = pd.to_datetime(df_consumo['Fecha_hora'], format='%Y-%m-%d %H:%M:%S')

        # Crear una columna 'Uso' (1 si el termotanque está encendido, 0 si está apagado)
        df_consumo['Uso'] = (df_consumo['Potencia'] > 0).astype(int)

        # Crear una columna 'Encendido' que marca el inicio de cada encendido
        df_consumo['Encendido'] = (df_consumo['Uso'] > df_consumo['Uso'].shift(fill_value=0)).astype(int)

        # Etiquetar cada encendido con un número de encendido
        df_consumo['Encendido_Numero'] = df_consumo['Encendido'].cumsum()

        # Calcular la duración de cada encendido
        df_duraciones = df_consumo[df_consumo['Uso'] == 1].groupby('Encendido_Numero')['Fecha_hora'].agg(lambda x: (x.max() - x.min()).total_seconds() / 60)

        # Filtrar encendidos de recuperación (duración menor a 1.2 veces el tiempo de recuperación calculado)
        encendidos_recuperacion = df_duraciones[df_duraciones < 1.2 * tiempo_recuperacion].index
        
        # Asignar valores de potencia nulos a los encendidos de recuperación
        df_consumo.loc[df_consumo['Encendido_Numero'].isin(encendidos_recuperacion), 'Potencia'] = 0

        # Eliminar columnas temporales creadas para el procesamiento
        df_consumo.drop(['Uso', 'Encendido', 'Encendido_Numero'], axis=1, inplace=True)

        # Crear la ruta para guardar el nuevo archivo modificado
        ruta_resultado = os.path.join(carpeta_resultados, f"{nombre_termotanque}_3.csv")

        # Guardar el DataFrame modificado como un nuevo archivo .csv
        df_consumo.to_csv(ruta_resultado, sep='\t', index=False)
    else:
        print(f"No hay datos para el termotanque {nombre_termotanque} en la tabla de tiempos de recuperacion.")

print("Parte 3 completada. Se obtuvo la colección de archivos Termotanques_min_encendidos_uso")

##########################################################################################################

### Parte 4 - Colección de archivos Termotanques_min_encendidos_uso_1

# Ruta de la carpeta con los archivos
carpeta_originales = "D:/TT/Termotanques_min_encendidos_uso"
carpeta_nueva = "D:/TT/Termotanques_min_encendidos_uso_1"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_nueva):
    os.makedirs(carpeta_nueva)

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta_originales)

# Iterar sobre cada archivo en la carpeta
for archivo in archivos:
    # Leer el archivo .csv
    ruta_archivo = os.path.join(carpeta_originales, archivo)
    df = pd.read_csv(ruta_archivo, sep='\t')
    
    # Convertir la columna 'Fecha_hora' a tipo datetime
    df['Fecha_hora'] = pd.to_datetime(df['Fecha_hora'], format='%Y-%m-%d %H:%M:%S')
    
    # Crear una columna auxiliar para horas y minutos
    df['hora_minuto'] = df['Fecha_hora'].dt.hour + df['Fecha_hora'].dt.minute / 60
    
    # Convertir a 0 aquellos valores de 'Potencia' que sean mayores a 0 y que estén fuera del horario especificado
    df.loc[(df['Potencia'] > 0) & ((df['hora_minuto'] < 18) | (df['hora_minuto'] >= 23)), 'Potencia'] = 0
    
    # Eliminar la columna auxiliar
    df.drop(columns=['hora_minuto'], inplace=True)
    
    # Guardar el archivo modificado en la carpeta nueva
    nueva_ruta_archivo = os.path.join(carpeta_nueva, archivo)
    df.to_csv(nueva_ruta_archivo, index=False)

print("Parte 4 completada. Se obtuvo la colección de archivos Termotanques_min_encendidos_uso_1")

##########################################################################################################

### Parte 5 - Colección de archivos Encendidos_uso_1

# Ruta de la carpeta con los archivos
carpeta_datos = "D:/TT/Termotanques_min_encendidos_uso_1"
carpeta_destino = "D:/TT/Encendidos_uso_1"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta_datos)

# Iterar sobre cada archivo en la carpeta
for archivo in archivos:
    # Crear la ruta completa del archivo de origen
    ruta_origen = os.path.join(carpeta_datos, archivo)

    # Leer el archivo .csv con encabezados y parseo de fechas
    df = pd.read_csv(ruta_origen, sep=',')
    df['Fecha_hora'] = pd.to_datetime(df['Fecha_hora'], format='%Y-%m-%d %H:%M:%S')

    # Identificar los encendidos
    encendidos = (df['Potencia'] > 0).astype(int)
    cambios_encendido = encendidos.diff()

    # Iniciar variables para seguimiento del encendido actual
    en_encendido = False
    inicio_encendido = None
    fin_encendido = None

    # Listas para almacenar datos de encendidos
    encendidos_numero = []
    duracion_encendidos = []
    inicio_encendidos = []
    fin_encendidos = []

    # Iterar sobre los cambios de estado
    for i, cambio in enumerate(cambios_encendido):
        if cambio == 1:
            # Comienzo de un encendido
            en_encendido = True
            inicio_encendido = df['Fecha_hora'][i]
        elif cambio == -1:
            # Fin de un encendido
            en_encendido = False
            fin_encendido = df['Fecha_hora'][i - 1]
            
            # Verificar si inicio_encendido es None antes de calcular la duración
            if inicio_encendido is not None:
                duracion_encendido = (fin_encendido - inicio_encendido).total_seconds() / 60

                # Almacenar datos del encendido
                encendidos_numero.append(len(encendidos_numero) + 1)
                duracion_encendidos.append(duracion_encendido)
                inicio_encendidos.append(inicio_encendido)
                fin_encendidos.append(fin_encendido)
            else:
                print(f"Advertencia: Inicio de encendido es None para el archivo {archivo}")

    # Crear DataFrame con los resultados
    df_encendidos = pd.DataFrame({
        'Encendido N': encendidos_numero,
        'Duracion en minutos': duracion_encendidos,
        'Hora de inicio': inicio_encendidos,
        'Hora de fin': fin_encendidos
    })

    # Sumar 1 a cada valor de la columna 'Duracion en minutos'
    df_encendidos['Duracion en minutos'] += 1

    # Guardar el nuevo archivo .csv en la carpeta de destino
    ruta_destino = os.path.join(carpeta_destino, archivo.replace("_3.csv", "_EU.csv"))
    df_encendidos.to_csv(ruta_destino, index=False)

print("Parte 5 completada. Se obtuvo la colección de archivos Encendidos_uso_1")

##########################################################################################################

### Parte 6 - Colección de archivos Encendidos_uso_potencialmente_interrumplibles_1

# Definir la hora de comienzo de la ventana de interrupción (VI)
hora_inicio_VI = 18

# Ruta de la carpeta original
carpeta_origen = "D:/TT/Encendidos_uso_1"

# Ruta de la nueva carpeta
carpeta_destino = "D:/TT/Encendidos_uso_potencialmente_interrumplibles_1"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Listar todos los archivos en la carpeta de origen
archivos_origen = os.listdir(carpeta_origen)

for archivo in archivos_origen:
    ruta_origen = os.path.join(carpeta_origen, archivo)
    df = pd.read_csv(ruta_origen)

    # Convertir columnas a datetime
    df['Hora de inicio'] = pd.to_datetime(df['Hora de inicio'])
    df['Hora de fin'] = pd.to_datetime(df['Hora de fin'])

    # Crear columna de fecha (sin hora) para agrupar por día
    df['Fecha'] = df['Hora de inicio'].dt.date

    encendidos_finales = []

    # Agrupar por día
    for fecha, grupo in df.groupby('Fecha'):
        inicio_intervalo = datetime.combine(fecha, datetime.min.time()) + timedelta(hours=hora_inicio_VI)
        fin_intervalo = datetime.combine(fecha + timedelta(days=1), datetime.min.time())

        # Filtrar encendidos que comienzan dentro del intervalo 1
        grupo_intervalo = grupo[(grupo['Hora de inicio'] >= inicio_intervalo) & (grupo['Hora de inicio'] < fin_intervalo)]

        if not grupo_intervalo.empty:
            # Verificar si hay encendido de uso en la ventana de recuperación (VR)
            inicio_vr = datetime.combine(fecha + timedelta(days=1), datetime.min.time())
            fin_vr = inicio_vr + timedelta(hours=6)  # Ajustar si la VR termina en otro horario

            encendido_vr = grupo[(grupo['Hora de inicio'] >= inicio_vr) & (grupo['Hora de inicio'] < fin_vr)]

            if encendido_vr.empty:
                # Si no hay encendido en VR, se puede guardar el del intervalo 1
                ultimo_encendido = grupo_intervalo.loc[grupo_intervalo['Hora de inicio'].idxmax()]
                encendidos_finales.append(ultimo_encendido)

    # Crear nuevo DataFrame con los encendidos seleccionados
    df_resultado = pd.DataFrame(encendidos_finales)

    # Guardar el archivo resultante
    ruta_destino = os.path.join(carpeta_destino, archivo)
    df_resultado.to_csv(ruta_destino, index=False)

print("Parte 6 completada. Se obtuvo la colección de archivos Encendidos_uso_potencialmente_interrumplibles_1")

##########################################################################################################

### Parte 7 - Colección de archivos Encendidos_uso_potencialmente_interrumplibles

# Definir la hora de finalización de la ventana de interrupción (VI)
hora_fin_VI = 23

# Ruta de la carpeta original
carpeta_origen = "D:/TT/Encendidos_uso_potencialmente_interrumplibles_1"

# Ruta de la nueva carpeta
carpeta_destino = "D:/TT/Encendidos_uso_potencialmente_interrumplibles"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Listar todos los archivos en la carpeta de origen
archivos_origen = os.listdir(carpeta_origen)

for archivo in archivos_origen:
    ruta_origen = os.path.join(carpeta_origen, archivo)
    df = pd.read_csv(ruta_origen)

    # Convertir columnas a datetime
    df['Hora de inicio'] = pd.to_datetime(df['Hora de inicio'])
    df['Hora de fin'] = pd.to_datetime(df['Hora de fin'])

    # Crear columna de fecha (sin hora) para agrupar por día
    df['Fecha'] = df['Hora de inicio'].dt.date

    encendidos_finales = []

    # Agrupar por día
    for fecha, grupo in df.groupby('Fecha'):
        inicio_intervalo = datetime.combine(fecha, datetime.min.time()) + timedelta(hours=hora_inicio_VI)
        fin_intervalo = datetime.combine(fecha, datetime.min.time()) + timedelta(hours=hora_fin_VI)

        # Filtrar encendidos que comienzan dentro del intervalo 1
        grupo_intervalo = grupo[(grupo['Hora de inicio'] >= inicio_intervalo) & (grupo['Hora de inicio'] < fin_intervalo)]

        if not grupo_intervalo.empty:
            # Seleccionar el último encendido del intervalo 1 (mayor Hora de inicio)
            ultimo_encendido = grupo_intervalo.loc[grupo_intervalo['Hora de inicio'].idxmax()]
            encendidos_finales.append(ultimo_encendido)

    # Crear nuevo DataFrame con los encendidos seleccionados
    df_resultado = pd.DataFrame(encendidos_finales)

    # Guardar el archivo resultante
    ruta_destino = os.path.join(carpeta_destino, archivo)
    df_resultado.to_csv(ruta_destino, index=False)

print("Parte 7 completada. Se obtuvo la colección de archivos Encendidos_uso_potencialmente_interrumplibles")

##########################################################################################################

### Parte 8 - Colección de archivos Termotanques_min_encendidos_potencialmente_interrumpibles

## Parte 8.1

# Directorios de entrada y salida
dir_consumo = "D:/TT/Termotanques_min_encendidos_uso_1"
dir_encendidos = "D:/TT/Encendidos_uso_potencialmente_interrumplibles"
dir_salida = "D:/TT/Termotanques_min_encendidos_potencialmente_interrumpibles"

# Obtener la lista de archivos de encendidos
archivos_encendidos = os.listdir(dir_encendidos)

# Crear la carpeta de destino si no existe
if not os.path.exists(dir_salida):
    os.makedirs(dir_salida)

# Iterar sobre cada archivo de encendidos
for archivo_encendidos in archivos_encendidos:
    if archivo_encendidos.endswith("_EU.csv"):
        # Leer el archivo de encendidos
        df_encendidos = pd.read_csv(os.path.join(dir_encendidos, archivo_encendidos))
        
        # Obtener el ID del termotanque
        termotanque_id = archivo_encendidos.split("_")[0]
        
        # Leer el archivo de consumo del termotanque
        ruta_consumo = os.path.join(dir_consumo, f"{termotanque_id}_3.csv")
        df_consumo = pd.read_csv(ruta_consumo, sep=',')
        
        # Convertir la columna 'Fecha_hora' a tipo datetime
        df_consumo['Fecha_hora'] = pd.to_datetime(df_consumo['Fecha_hora'], format='%Y-%m-%d %H:%M:%S')
        
        # Guardar la columna 'Potencia' del dir_consumo de forma auxiliar
        df_aux = df_consumo['Potencia']
        
        # Inicializar la columna 'Potencia' con 0
        df_consumo['Potencia'] = 0
        
        # Iterar sobre cada encendido principal y asignar el valor de potencia correspondiente
        for index, row in df_encendidos.iterrows():
            inicio = pd.to_datetime(row['Hora de inicio'])
            fin = pd.to_datetime(row['Hora de fin'])
            duracion = (fin - inicio).total_seconds() / 60
            
            # Seleccionar los valores de potencia relevantes para el rango de fechas del encendido
            potencia_encendido = df_aux[(df_consumo['Fecha_hora'] >= inicio) & (df_consumo['Fecha_hora'] <= fin)]
            
            # Asignar los valores de potencia del encendido al rango de fechas correspondiente
            df_consumo.loc[(df_consumo['Fecha_hora'] >= inicio) & (df_consumo['Fecha_hora'] <= fin), 'Potencia'] = potencia_encendido
                                 
        # Guardar el archivo resultante en el directorio de salida
        ruta_salida = os.path.join(dir_salida, f"{termotanque_id}_4.csv")
        df_consumo.to_csv(ruta_salida, index=False, sep='\t')

## Parte 8.2

# Directorio de entrada (y salida)
dir_entrada = "D:/TT/Termotanques_min_encendidos_potencialmente_interrumpibles"

# Obtener la lista de archivos de consumo horario
archivos = os.listdir(dir_entrada)

# Iterar sobre cada archivo de consumo horario
for archivo in archivos:
    if archivo.endswith(".csv"):
        # Leer el archivo .csv
        ruta_archivo = os.path.join(dir_entrada, archivo)
        df = pd.read_csv(ruta_archivo, sep='\t')

        # Convertir la columna 'Fecha_hora' a tipo datetime
        df['Fecha_hora'] = pd.to_datetime(df['Fecha_hora'], format='%Y-%m-%d %H:%M:%S')

        # Calcular la diferencia de tiempo entre filas consecutivas
        diff_tiempo = df['Fecha_hora'].diff().fillna(pd.Timedelta(seconds=0))

        # Encontrar los índices donde la diferencia de tiempo es mayor a 3 días
        indices_intervalos = diff_tiempo.loc[diff_tiempo > pd.Timedelta(days=3)].index

        # Iterar sobre los índices de los intervalos y reemplazar los valores de 'Potencia' con NaN donde sea apropiado
        for idx in indices_intervalos:
            df.loc[idx:, 'Potencia'] = np.where(df.loc[idx:, 'Potencia'] == 0, np.nan, df.loc[idx:, 'Potencia'])

        # Sobrescribir el archivo original con los cambios
        df.to_csv(ruta_archivo, index=False, sep='\t')

print("Parte 8 completada. Se obtuvo la colección de archivos Termotanques_min_encendidos_potencialmente_interrumpibles.")

##########################################################################################################

### Parte 9 - Colección de archivos Termotanques_min_enc_pot_interr_recortados

## Parte 9.1

# Ruta del archivo con los tiempos de recuperación
ruta_tiempos_recuperacion = "D:/TT/tiempos_de_recuperacion.csv"

# Crear listas vacías para almacenar los tiempos de interrupción
tiempos_interrupcion = []

# Leer el archivo .csv con los encendidos
df_tiempos_recuperacion = pd.read_csv(ruta_tiempos_recuperacion)

# Extraer la columna 'Tiempo de recuperacion'
tiempos_de_recuperacion = df_tiempos_recuperacion['Tiempo de recuperacion'].values

# Dividir los tiempos de recuperación por el coeficiente 0.34
tiempos_de_interrupcion = tiempos_de_recuperacion / 0.34

# Redondear al entero superior más cercano
tiempos_de_interrupcion = [math.ceil(tiempo) for tiempo in tiempos_de_interrupcion]

# Crear DataFrame para los tiempos de interrupción
df_tiempos_interrupcion = pd.DataFrame({
    'Termotanque': df_tiempos_recuperacion['Termotanque'],
    'Tiempo de interrupcion': tiempos_de_interrupcion
})

# Guardar el DataFrame de resultados en un archivos .csv
df_tiempos_interrupcion.to_csv("tiempos_de_interrupcion.csv", index=False)

### Parte 9.2

# Directorios de entrada y salida
carpeta_entrada = "D:/TT/Termotanques_min_encendidos_potencialmente_interrumpibles"
carpeta_salida = "D:/TT/Termotanques_min_enc_pot_interr_recortados"
archivo_interrupciones = "D:/TT/tiempos_de_interrupcion.csv"

# Leer datos de interrupciones
df_interrupciones = pd.read_csv(archivo_interrupciones)

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# Iterar sobre archivos de entrada
for archivo in os.listdir(carpeta_entrada):
    if archivo.endswith(".csv"):
        # Leer el archivo de entrada
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        df = pd.read_csv(ruta_entrada, sep='\t')

        # Obtener el nombre del termotanque
        nombre_termotanque = archivo.replace("_4.csv", "")

        # Verificar si hay datos para el termotanque actual en las interrupciones
        if nombre_termotanque in df_interrupciones['Termotanque'].values:
            # Obtener el tiempo de interrupción para el termotanque actual
            tiempo_interrupcion = df_interrupciones[df_interrupciones['Termotanque'] == nombre_termotanque]['Tiempo de interrupcion'].values[0]

            # Convertir la columna 'Fecha_hora' a tipo datetime
            df['Fecha_hora'] = pd.to_datetime(df['Fecha_hora'], format='%Y-%m-%d %H:%M:%S')

            # Crear una columna 'Uso' (1 si el termotanque está encendido, 0 si está apagado)
            df['Uso'] = (df['Potencia'] > 0).astype(int)

            # Crear una columna 'Encendido' que marca el inicio de cada encendido
            df['Encendido'] = (df['Uso'] > df['Uso'].shift(fill_value=0)).astype(int)

            # Etiquetar cada encendido con un número de encendido
            df['Encendido_Numero'] = df['Encendido'].cumsum()

            # Calcular la duración de cada encendido
            df_duraciones = df[df['Uso'] == 1].groupby('Encendido_Numero')['Fecha_hora'].agg(lambda x: (x.max() - x.min()).total_seconds() / 60)

            # Filtrar encendidos para recortar
            encendidos_recortar = df_duraciones.index

            # Asignar valores de potencia nulos a los últimos minutos de los encendidos a recortar
            for encendido in encendidos_recortar:
                indices_recortar = df[(df['Encendido_Numero'] == encendido) & (df['Uso'] == 1)].index[-int(tiempo_interrupcion):]
                df.loc[indices_recortar, 'Potencia'] = 0

            # Eliminar columnas temporales creadas para el procesamiento
            df.drop(['Uso', 'Encendido', 'Encendido_Numero'], axis=1, inplace=True)

            # Crear la ruta para guardar el nuevo archivo modificado
            ruta_salida = os.path.join(carpeta_salida, f"{nombre_termotanque}_5.csv")

            # Guardar el DataFrame modificado como un nuevo archivo .csv
            df.to_csv(ruta_salida, sep='\t', index=False)
        else:
            print(f"No hay datos de interrupción para el termotanque {nombre_termotanque}.")

print("Parte 9 completada. Se obtuvo la colección de archivos Termotanques_min_enc_pot_interr_recortados.")

##########################################################################################################

### Parte 10 - Colección de archivos Termotanques_min_potenciales_interrupciones

# Directorios de entrada
dir_ep = "D:/TT/Termotanques_min_encendidos_potencialmente_interrumpibles"
dir_ep_r = "D:/TT/Termotanques_min_enc_pot_interr_recortados"

# Directorio de salida
dir_i = "D:/TT/Termotanques_min_potenciales_interrupciones"

# Crear la carpeta de destino si no existe
if not os.path.exists(dir_i):
    os.makedirs(dir_i)

# Obtener la lista de archivos de ep
archivos_ep_r = os.listdir(dir_ep_r)

# Iterar sobre cada archivo de ep
for archivo_ep_r in archivos_ep_r:
    # Obtener el ID del termotanque
    id_termotanque = archivo_ep_r.split('_')[0]

    # Definir los nombres de los archivos de ep e i
    archivo_ep = f"{id_termotanque}_4.csv"
    archivo_i = f"{id_termotanque}_I.csv"

    # Crear las rutas completas de los archivos
    ruta_ep = os.path.join(dir_ep, archivo_ep)
    ruta_ep_r = os.path.join(dir_ep_r, archivo_ep_r)
    ruta_i = os.path.join(dir_i, archivo_i)

    # Verificar si el archivo de ep existe
    if not os.path.exists(ruta_ep):
        print(f"Advertencia: No se encontró el archivo de ep para {id_termotanque}.")
        continue

    # Leer los archivos de ep e i
    df_ep = pd.read_csv(ruta_ep, parse_dates=['Fecha_hora'], sep='\t')
    df_ep_r = pd.read_csv(ruta_ep_r, parse_dates=['Fecha_hora'], sep='\t')

    # Fusionar los dataframes usando la columna 'Fecha_hora' como índice
    df_merged = pd.merge(df_ep, df_ep_r, on='Fecha_hora', how='left', suffixes=('_4', '_5'))

    # Calcular la resta de las potencias y guardar en el archivo I
    df_i = pd.DataFrame({
        'Fecha_hora': df_merged['Fecha_hora'],
        'Potencia_I': df_merged['Potencia_4'] - df_merged['Potencia_5']
    })

    # Guardar el archivo I en el directorio de salida
    df_i.to_csv(ruta_i, index=False)

print("Parte 10 completada. Se obtuvo la colección de archivos Termotanques_min_potenciales_interrupciones.")

##########################################################################################################

### Parte 11 - Colección de archivos Termotanques_horario_potenciales_interrupciones

### Parte 11.1

# Ruta de la carpeta con los archivos
carpeta_archivos = "D:/TT/Termotanques_min_potenciales_interrupciones"
carpeta_resultados = "D:/TT/Termotanques_horario_potenciales_interrupciones_1"

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_resultados):
    os.makedirs(carpeta_resultados)

# Lista para almacenar los DataFrames procesados
resultados = []

# Iterar sobre cada archivo en la carpeta
for archivo in os.listdir(carpeta_archivos):
    ruta_archivo = os.path.join(carpeta_archivos, archivo)
    
    # Verificar si es un archivo .csv
    if archivo.endswith(".csv"):
        # Leer el archivo .csv con tabulador como delimitador
        df = pd.read_csv(ruta_archivo, sep=',')

        # Verificar si 'Fecha_hora' está en el DataFrame antes de convertir
        if 'Fecha_hora' in df.columns:
            # Convertir la columna 'Fecha_hora' a formato datetime
            df['Fecha_hora'] = pd.to_datetime(df['Fecha_hora'], format='%Y-%m-%d %H:%M:%S')

            # Agrupar por hora y calcular el promedio de la columna 'Potencia_I'
            df_result = df.groupby(df['Fecha_hora'].dt.strftime('%Y-%m-%d %H:00:00'))['Potencia_I'].mean().reset_index()

            # Renombrar las columnas
            df_result.columns = ['Fecha', 'Potencia_I']

            # Guardar el resultado en un nuevo archivo .csv en la carpeta de resultados
            ruta_resultado = os.path.join(carpeta_resultados, f"{os.path.splitext(archivo)[0]}_horario.csv")
            df_result.to_csv(ruta_resultado, index=False, date_format='%Y-%m-%d %H:%M:%S')

            # Almacenar el DataFrame procesado en la lista de resultados
            resultados.append(df_result)

# Combinar todos los resultados en un solo DataFrame
df_resultados_totales = pd.concat(resultados, ignore_index=True)

### Parte 11.2

# Directorios de entrada y salida
directorio_entrada = "D:/TT/Termotanques_horario_potenciales_interrupciones_1"
directorio_salida = "D:/TT/Termotanques_horario_potenciales_interrupciones"

# Crear la carpeta de destino si no existe
if not os.path.exists(directorio_salida):
    os.makedirs(directorio_salida)

# Fechas dadas
fecha_mas_antigua = pd.to_datetime("2019-01-01 00:00:00")
fecha_mas_reciente = pd.to_datetime("2023-01-01 00:00:00")

# Iterar sobre cada archivo en el directorio de entrada
for archivo in os.listdir(directorio_entrada):
    # Leer el archivo
    ruta_archivo_entrada = os.path.join(directorio_entrada, archivo)
    df = pd.read_csv(ruta_archivo_entrada, parse_dates=['Fecha'])

    # Extender el horizonte de tiempo
    fechas_faltantes = pd.date_range(start=fecha_mas_antigua, end=fecha_mas_reciente, freq='H').difference(df['Fecha'])
    df_extension = pd.DataFrame({'Fecha': fechas_faltantes, 'Potencia_I': np.nan})

    # Concatenar el DataFrame original con el extendido
    df_resultado = pd.concat([df, df_extension], ignore_index=True)

    # Ordenar por fecha
    df_resultado.sort_values(by='Fecha', inplace=True)

    # Guardar el nuevo archivo en el directorio de salida
    ruta_archivo_salida = os.path.join(directorio_salida, archivo)
    df_resultado.to_csv(ruta_archivo_salida, index=False)

print("Parte 11 completada. Se obtuvo la colección de archivos Termotanques_horario_potenciales_interrupciones.")

##########################################################################################################

######################## PARTE B.3. CALCULO DE LA POTENCIA MÁXIMA DE INTERRUPCIÓN ########################

##########################################################################################################

### Parte 1 - Tabla de interrupciones

# Directorio que contiene los archivos .csv
directorio_datos = "D:/TT/Termotanques_horario_potenciales_interrupciones"

# Obtener la lista de archivos en el directorio
archivos_csv = [archivo for archivo in os.listdir(directorio_datos) if archivo.endswith('.csv')]

# Leer el primer archivo para obtener las fechas
archivo_inicial = pd.read_csv(os.path.join(directorio_datos, archivos_csv[0]))
fechas = archivo_inicial['Fecha']

# Crear un DataFrame para almacenar las potencias de todos los archivos
df_final = pd.DataFrame({'Fecha': fechas})

# Lista para almacenar las columnas de potencia
columnas_potencia = []

# Iterar sobre los archivos y agregar las potencias al DataFrame final
for archivo_csv in archivos_csv:
    nombre_columna = 'Potencia_I_' + archivo_csv.split('.')[0]
    df_temporal = pd.read_csv(os.path.join(directorio_datos, archivo_csv))
    columnas_potencia.append(df_temporal['Potencia_I'])

# Concatenar todas las columnas de potencia en el DataFrame final
df_final = pd.concat([df_final] + columnas_potencia, axis=1)

# Guardar el DataFrame final en un archivo .csv
df_final.to_csv("D:/TT/combined_data.csv", index=False)

print("Parte 1 completada. Se obtuvo la tabla de interrupciones.")

##########################################################################################################

### Parte 2 - Interrupción promedio horaria unitaria

# Leer el archivo combinado
ruta_archivo_combinado = "D:/TT/combined_data.csv"
df_combinado = pd.read_csv(ruta_archivo_combinado, parse_dates=['Fecha'])

# Calcular el promedio de cada fila (excluyendo la columna de fechas)
df_combinado['Promedio'] = df_combinado.iloc[:, 1:].mean(axis=1)

# Quedarme solo con la fecha y el promedio
df_combinado = df_combinado.iloc[:, [0, -1]]

# Guardar el DataFrame actualizado en un nuevo archivo .csv
ruta_archivo_promedio = "D:/TT/combined_data_with_avg.csv"
df_combinado.to_csv(ruta_archivo_promedio, index=False)

print("Parte 2 completada. Se obtuvo la interrupción promedio horaria unitaria.")

##########################################################################################################

### Parte 3 - Generalización al parque de termotanques gestionables

# Ruta del archivo original con promedios
ruta_archivo_original = "D:/TT/combined_data_with_avg.csv"
df_original = pd.read_csv(ruta_archivo_original, parse_dates=['Fecha'])

# Multiplicar la segunda columna por k=1.000.000 y pasar a MW
# Aplicar el término de pérdidas del 7%
k = 1000000
df_original['Promedio'] = df_original['Promedio'] * k * 0.000001 * (1 - 0.07)

# Cambiar los nombres de las columnas
nuevos_nombres = {'Fecha': 'Fecha_hora', 'Promedio': 'Potencia_I_MW'}
df_original.columns = [nuevos_nombres.get(col, col) for col in df_original.columns]

carpeta_resultados = "D:/TT/Resultados_N"
if not os.path.exists(carpeta_resultados):
    os.makedirs(carpeta_resultados)

# Guardar el resultado en un nuevo archivo .csv
ruta_archivo_salida = "D:/TT/Resultados_N/Tira_1M_TT_2022_Interrupcion_18_23.csv"
df_original.to_csv(ruta_archivo_salida, index=False)

##########################################################################################################

##################################### TERMINA EL SCRIPT TERMOTANQUES #####################################

##########################################################################################################