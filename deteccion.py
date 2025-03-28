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
    def obtener_resultados(self):
        pass

    @abstractmethod
    def obtener_coord_jugador(self):
        pass

    @abstractmethod
    def obtener_coord_nodos(self):
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

    def inferencia(self, imagen, conf=0.5, filtro=[1], dibujar=False):
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

    def obtener_coord_jugador(self):
        return (957, 670)
    
    def obtener_coord_nodos(self):
        return self.coord_nodos

    def obtener_resultados(self):
        return self.resultados

    def obtener_nodos(self):
        return self.nodos

    def obtener_nodo_mas_grande(self):
        return self.deteccion_mas_grande

    def obtener_pos_jugador(self):
        return self.pos_jugador

    ######LLEVAR A OTRA CLASE
    def iniciar_tracker(self, nodo):
        _, x, y, h, w, _ = nodo
        self.bbox = (x, y, w, h)
        self.tracker.init(self.imagen, self.bbox)
        return self.bbox
    
    def actualizar_tracker(self, imagen):
        self.visto, self.bbox = self.tracker.update(imagen)
        return self.visto, self.bbox

    def _calcular_detecciones(self, dibujar=False):
        if not self.resultados:
            raise ValueError("Se debe realizar la inferencia primero con el método 'infer'.")

        self.nodos = []
        self.coord_nodos = []
        mejor_puntuacion = float('-inf')
        self.deteccion_mas_grande = None
        self.pos_jugador = None

        # Centro de la pantalla (en un array para cálculos más rápidos con NumPy)
        centro_pantalla = np.array([960, 540])

        # Dimensiones de la pantalla (puedes ajustar si cambian)
        ancho_pantalla = 1920
        alto_pantalla = 1080

        # Dividir la pantalla en tercios
        tercio_izquierdo = ancho_pantalla // 3
        tercio_derecho = 2 * ancho_pantalla // 3

        if self.resultados:
            for resultado in self.resultados:
                for box in resultado.boxes:
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    xn = int((x1 + x2) / 2)
                    yn = int((y1 + y2) / 2)
                    if cls == 0:  # Filtrar por la clase con identificador 0 (jugador)
                        self.pos_jugador = (cls, x1, y1, x2, y2, conf)
                        self.coord_jugador = (xn, yn)
                    else:
                        # Calcular el área y el centro del nodo
                        area = (x2 - x1) * (y2 - y1)
                        nodo_centro = np.array([(x1 + x2) // 2, (y1 + y2) // 2])

                        # Calcular la distancia euclidiana desde el centro de la pantalla
                        distancia = np.linalg.norm(nodo_centro - centro_pantalla)

                        # Ponderar la puntuación (más área y menos distancia)
                        puntuacion =  area - distancia * 0.5

                        # Penalizar los nodos que estén en los tercios laterales (izquierda o derecha)
                        if nodo_centro[0] < tercio_izquierdo or nodo_centro[0] > tercio_derecho:
                            puntuacion -= 100  # Penaliza con un valor adecuado

                        self.nodos.append((cls, x1, y1, x2, y2, conf))
                        self.coord_nodos.append((xn, yn))
                        # Establecer un umbral para evitar que el objeto cambie de uno a otro fácilmente
                        umbral_proximidad = 50  # Ajusta este valor según sea necesario

                        # Si la puntuación es mejor y el objeto no está demasiado cerca de otro objeto
                        if puntuacion > mejor_puntuacion and (self.deteccion_mas_grande is None or
                                np.linalg.norm(nodo_centro - np.array([self.deteccion_mas_grande[1], self.deteccion_mas_grande[2]])) > umbral_proximidad):
                            mejor_puntuacion = puntuacion
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
                
        if self.imagen_resultados is not None and self.pos_jugador:
            cls, x1, y1, x2, y2, conf = self.pos_jugador
            cv2.rectangle(self.imagen_resultados, (x1, y1), (x2, y2), (255, 0, 0), 2)
            class_name = self.modelo.names[cls]
            label_text = f"{class_name} {conf:.2f}"
            self.imagen_resultados = cv2.putText(self.imagen_resultados, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
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

    def _corregir_deteccion_fov(self, x1, y1, x2, y2, fov, screen_center_x, screen_center_y):
            # 1. Calcular el centro del bounding box correctamente
        x_center = (x1 + y1) / 2
        y_center = (x2 + y2) / 2
        
        # 2. Calcular el tamaño original del bounding box
        width = x2 - y1
        height = y2 - x1
        
        # 3. Asegurarse de que el bounding box no se desplace
        # Calcular nuevas coordenadas con el centro bien posicionado (sin escalar ni distorsionar)
        x1_corregido = np.round(x_center - width / 2).astype(int)
        x2_corregido = np.round(x_center + width / 2).astype(int)
        y1_corregido = np.round(y_center - height / 2).astype(int)
        y2_corregido = np.round(y_center + height / 2).astype(int)
        
        # Retornar las coordenadas corregidas
        return x1_corregido, y1_corregido, x2_corregido, y2_corregido
