import numpy as np
import cv2
from ultralytics import YOLO
import os
from abc import ABC, abstractmethod

# Definir la interfaz para el modelo YOLO
class YoloModelInterface(ABC):
    @abstractmethod
    def inferencia(self, imagen, conf=0.5, filtro=None, dibujar=False):
        """
        Realiza la inferencia sobre una imagen de entrada utilizando el modelo YOLO.
        :param imagen: Imagen de entrada en formato numpy array.
        :return: Resultados de la inferencia.
        """
        pass

    @abstractmethod
    def guardar_deteciones(self, tstamp):
        pass
        
    @abstractmethod
    def obtener_resultados(self):
        pass


    @abstractmethod
    def obtener_nodos(self):
        pass

    @abstractmethod
    def obtener_nodo_mas_grande(self):
        pass

    @abstractmethod
    def obtener_pos_jugador(self):
        pass


# Implementación de la clase que realiza la inferencia y obtiene las detecciones
class YoloModel(YoloModelInterface):

    class_names_to_id = {
        "accion": 0,
        "jugador": 1,
        "nodo": 2
    }
    
    def __init__(self, ruta, output_dir="./train/detecciones/"):
        """
        Inicializa el modelo YOLO para detección de objetos.
        :param ruta: Ruta del modelo YOLO.
        """
        self.modelo = YOLO(ruta)
        self.resultados = None
        self.nodo_mas_grande = None
        self.deteccion_mas_grande = None
        self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.nodos = []
        self.pos_jugador = None
        self.imagen = None
        self.imagen_resultados = None
        self.imagen_tracker = None
        self.bbox = None
        self.visto = None
        self.output_dir = output_dir

    def inferencia(self, imagen, conf=0.5, filtro=None, dibujar=False):
        """
        Realiza la inferencia sobre una imagen de entrada utilizando el modelo YOLO.
        :param imagen: Imagen de entrada en formato numpy array.
        :return: Resultados de la inferencia.
        """
        self.imagen = imagen
        self.imagen_resultados = imagen.copy()

        if filtro is None:
            self.resultados = self.modelo(imagen, conf=conf)
        else:
            self.resultados = self.modelo(source=imagen, conf=conf, classes=filtro)

        self._calcular_detecciones(dibujar)


    def obtener_resultados(self):
        return self.resultados

    def obtener_nodos(self):
        return self.nodos

    def obtener_nodo_mas_grande(self):
        return self.deteccion_mas_grande

    def obtener_pos_jugador(self):
        return self.pos_jugador

    ######LLEVAR A OTRA CLASE
    def _iniciar_tracker(self, nodo):
        _, x, y, h, w, _ = nodo
        self.bbox = (x, y, w, h)
        self.tracker.init(self.imagen, self.bbox)
        return self.bbox
    
    def _actualizar_tracker(self, imagen):
        self.visto, self.bbox = self.tracker.update(imagen)
        return self.visto, self.bbox


    def _calcular_detecciones(self, dibujar=False):
        """
        Devuelve las detecciones filtradas por confianza y por clase (si se especifica).
        Además, identifica la detección con el área más grande.
        :param confianza: Umbral de confianza para filtrar las detecciones.
        :param filtro: (Opcional) Clase específica para filtrar, si es None devuelve todas las clases.
        :return: Detecciones filtradas como un array con formato (idclase, x, y, altura, ancho, confianza).
        """
        if not self.resultados:
            raise ValueError("Se debe realizar la inferencia primero con el método 'infer'.")

        self.nodos = []
        max_area = 0
        self.deteccion_mas_grande = None
        
        if self.resultados:
            for resultado in self.resultados: 
                for box in resultado.boxes:
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls == 0:  # Filtrar por la clase con identificador 1
                        self.pos_jugador = (cls, x1, y1, x2, y2, conf)
                    else:
                        area = (x2 - x1) * (y2 - y1)  # Calcular el área del cuadro
                        self.nodos.append((cls, x1, y1, x2, y2, conf))
                        # Determinar la detección con el área más grande
                        if area > max_area:
                            max_area = area
                            self.deteccion_mas_grande = (cls, x1, y1, x2, y2, conf)

                        
        if dibujar:
            self._dibujar_cajas()
            self._mostrar_imagen()


    def _dibujar_cajas(self, color=(0, 255, 0), grosor=2):
        """
        Dibuja las cajas delimitadoras y etiquetas sobre la imagen.
        :param imagen: Imagen sobre la que se dibujarán las cajas.
        :param color: Color de las cajas y etiquetas.
        :param grosor: Grosor de las cajas.
        :return: Imagen con las cajas dibujadas.
        """
        
        if self.imagen is not None and self.nodos:
            self.imagen_resultados = self.imagen.copy()
            for nodo in self.nodos:  
                if nodo == self.deteccion_mas_grande:
                    color = (0, 0, 255) 
                else:
                    color = (0, 255, 0)
                cls, x1, y1, x2, y2, conf = nodo
                cv2.rectangle(self.imagen_resultados, (x1, y1), (x2, y2), color, 2)
                class_name = self.modelo.names[cls]
                label_text = f"{class_name} {conf:.2f}"
                self.imagen_resultados = cv2.putText(self.imagen_resultados, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if self.imagen is not None and self.pos_jugador:
            cls, x1, y1, x2, y2, conf = self.pos_jugador
            cv2.rectangle(self.imagen_resultados, (x1, y1), (x2, y2), (255, 0, 0), 2)
            class_name = self.modelo.names[cls]
            label_text = f"{class_name} {conf:.2f}"
            self.imagen_resultados = cv2.putText(self.imagen_resultados, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def guardar_deteciones(self, tstamp):
        """
        Guarda las detecciones en un archivo de texto.
        :param tstamp: Timestamp para el nombre del archivo.
        :param salida: Ruta de salida para el archivo de texto.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        nombre_fichero = tstamp + ".txt"
        salida = os.path.join(self.output_dir, nombre_fichero)

        if self.detecciones:
            with open(salida, 'w') as f:
                for deteccion in self.detecciones:
                    cls, x1, y1, x2, y2, _ = deteccion
                    x1, y1, x2, y2 = self._normalizar_coordenadas(x1, y1, x2, y2)
                    #x1, y1, x2, y2 = self._desnormalizar_coordenadas(x1, y1, x2, y2)
                    d = f"{cls} {x1} {y1} {x2} {y2}"
                    f.write(f"{d}\n")
        else:
            with open(salida, 'w') as f:
                f.write(" ")
    
    def _normalizar_coordenadas(self, x1, y1, x2, y2, image_width=1920, image_height=1080):
        """
        Normaliza las coordenadas de una caja delimitadora a un rango [0, 1] 
        basado en las dimensiones de la imagen de entrada (1920x1080 por defecto).
        
        :param x1: Coordenada x1 (esquina superior izquierda).
        :param y1: Coordenada y1 (esquina superior izquierda).
        :param x2: Coordenada x2 (esquina inferior derecha).
        :param y2: Coordenada y2 (esquina inferior derecha).
        :param image_width: Ancho de la imagen (por defecto 1920).
        :param image_height: Alto de la imagen (por defecto 1080).
        
        :return: Las coordenadas normalizadas (x1, y1, x2, y2).
        """
        x1_normalized = x1 / image_width
        y1_normalized = y1 / image_height
        x2_normalized = x2 / image_width
        y2_normalized = y2 / image_height

        return x1_normalized, y1_normalized, x2_normalized, y2_normalized

    def _desnormalizar_coordenadas(self, x1, y1, x2, y2, image_width=1024, image_height=576):
        """
        Desnormaliza las coordenadas de una caja delimitadora a un rango [0, 1]"""
        x1_desnormalizado = x1 * image_width
        y1_desnormalizado = y1 * image_height
        x2_desnormalizado = x2 * image_width
        y2_desnormalizado = y2 * image_height

        return x1_desnormalizado, y1_desnormalizado, x2_desnormalizado, y2_desnormalizado

    def _dibujar_caja_tracker(self, bbox, color=(0, 0, 255), grosor=2):
        """
        Dibuja las cajas delimitadoras y etiquetas sobre la imagen.
        :param imagen: Imagen sobre la que se dibujarán las cajas.
        :param color: Color de las cajas y etiquetas.
        :param grosor: Grosor de las cajas.
        :return: Imagen con las cajas dibujadas.
        """
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height

            # Dibuja la caja delimitadora
        imagen = cv2.rectangle(imagen, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, grosor)
        
        return imagen

    def _mostrar_imagen(self):
        """
        Muestra la imagen con las detecciones.
        """
        
        if self.imagen_resultados is not None:
            cv2.imshow("Detecciones", self.imagen_resultados)
