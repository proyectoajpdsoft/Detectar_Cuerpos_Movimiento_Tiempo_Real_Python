import cv2
from os.path import exists

# Escribir texto en frame (imagen)
def escribirTexto(imagen, texto, tamaño, grosor, x, y):
    cv2.putText(imagen, texto, (x, y), cv2.FONT_HERSHEY_PLAIN, tamaño, (0, 255, 0), grosor)

# Capturar imágenes a analizar desde WebCam
capturarWebCam = False

rutaXMLClasificador = r"D:\Mis documentos\ProyectoA\Python\reconocimiento_facial\haarcascade_fullbody.xml"
rutaVideo = r"D:\Mis documentos\ProyectoA\Python\reconocimiento_facial\video2.mp4"

# Iniciamos el clasificador de cuerpo completo
clasificadorCuerpo = cv2.CascadeClassifier(rutaXMLClasificador)

if capturarWebCam:
    # Para capturar vídeo de la webcam del equipo en tiempo real y detectar cuerpos
    capturaVideo = cv2.VideoCapture(0)
else:    
    # Si queremos capturar vídeo en fichero del equipo y detectar cuerpos
    if exists(rutaVideo):
        capturaVideo = cv2.VideoCapture(filename=rutaVideo)
    else:
        print("No se ha encontrado el vídeo indicado.")
        exit()

# Obtenemos el número total de frames del vídeo para ponerlo en la pantalla
totalFrames = int(capturaVideo.get(cv2.CAP_PROP_FRAME_COUNT))

# Mantenemos la ventana de visualización abierta mientras dure el vídeo
cuerpos = 0
while capturaVideo.isOpened():
    # Obtenemos y analizamos cada frame del vídeo
    videoIniciado, imagenCapturada = capturaVideo.read()
    # Si no hay más frames en el vídeo, salimos del bucle
    if not videoIniciado:
        print("No se ha encontrado imagen en la entrada de WebCam o se ha finalizado el fichero.")
        exit() # para vídeo desde fichero
    # Obtenemos el frame actual
    frameActual = int(capturaVideo.get(cv2.CAP_PROP_POS_FRAMES))
    # Convertimos a escala de grises
    frameEscalaGrises = cv2.cvtColor(imagenCapturada, cv2.COLOR_BGR2GRAY)
    # Detectamos los posibles cuerpos en el frame
    cuerposDetectados = clasificadorCuerpo.detectMultiScale(frameEscalaGrises, 1.2, 3)    
    # Recorremos los cuerpos detectados en el frame para identificarlos con un recuadro azul    
    for (x, y, w, h) in cuerposDetectados:
        cuerpos +=1
        # Si queremos mostrar un texto en el recuadro de cada cuerpo detectado
        cv2.rectangle(imagenCapturada, (x, y), (x + w, y + h), (255, 0, 0), 2)
        escribirTexto (imagenCapturada, f"C{cuerpos}", 0.9, 1, x, y - 10)
        # Mostramos el frame actual analizado y el total de frames del vídeo
        escribirTexto(imagenCapturada, f"{frameActual}/{totalFrames}", 1, 2, 10, 30)
        # Mostramos la ventana con el vídeo y los cuerpos detectados        
        cv2.imshow("ProyectoA - Detector de cuerpos completos", imagenCapturada)

    # Para cerrar detección y salir (pulsación de tecla "s")
    if cv2.waitKey(5) & 0xFF == ord('s'):
        break

# Liberamos los recursos
capturaVideo.release()
cv2.destroyAllWindows()