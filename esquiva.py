import numpy as np
from deteccion import YoloModelInterface, YoloModel
from segmentacion import MaskGeneratorInterface, BinaryMaskGenerator
import cv2

class Mapa():
    def __init__(self):
        self.nodos = []
        self.ruta_filtrada = []
        self.nodo_mas_cercano = None
        self.ruta = []
        self.mapa_color = np.ones((1080, 1920), dtype=np.uint8) * 255
        self.mapa_navegacion = None

    def run(self, detector: YoloModelInterface, segmentador: MaskGeneratorInterface):
        if detector.obtener_resultados:
            nodos = []
            pos_jugador = detector.obtener_pos_jugador()
            if pos_jugador is not None:
                _, x1p, y1p, x2p, y2p, _ = pos_jugador
                xp = int((x1p + x2p) / 2)
                yp = int((y1p + y2p) / 2)
                nodos_raw = detector.obtener_nodos()
                if nodos_raw is not None:
                    for nodo in nodos_raw:
                        _, x1n, y1n, x2n, y2n, _ = nodo
                        xn = int((x1n + x2n) / 2)
                        yn = int((y1n + y2n) / 2)
                        nodos.append((xn, yn))
                    
                    self.generar_mapa(segmentador.obtener_mascara(), (xp, yp), nodos)
                    self.mask_to_navigation_matrix()
                    if self.esta_cerca_de_obstaculo((xp, yp)):
                        print("CUIDAOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                    else:
                        print("Vaya tranquilo")
                    
                else:
                    self.generar_mapa(segmentador.obtener_mascara(), (xp, yp))
            else:
                self.generar_mapa(segmentador.obtener_mascara())
        self.mostrar_mapa()
           
    
    def generar_mapa(self, mapa, posicion_jugador=(), nodos=[]):
        #self.nodos = []
        #nodos_temporales = self.generar_nodos_temporales(mapa * 255)
        #self.nodos.extend(nodos_temporales)
        
        # Asegúrate de que el mapa tenga el tamaño correcto
        if mapa.ndim == 2:
            self.mapa_color = cv2.cvtColor(mapa * 255, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("El mapa debe ser una imagen en escala de grises")

        #for nodo_temp in nodos_temporales:
            #cv2.circle(self.mapa_color, nodo_temp, 2, (0, 255, 0), -1)

        if nodos:
            for nodo in nodos:
                cv2.circle(self.mapa_color, nodo, 5, (0, 0, 255), -1)
        if posicion_jugador:
            cv2.circle(self.mapa_color, posicion_jugador, 5, (255, 0, 0), -1)

        # Mostrar el mapa generado para depuración

    def mostrar_mapa(self):  
        cv2.imshow("Mapa Generado", self.mapa_color)

    # Función para leer la máscara RGB y convertirla a una matriz de navegación
    def mask_to_navigation_matrix(self):
        # Convertir la imagen RGB a una matriz de valores
        self.matrix = np.zeros((self.mapa_color.shape[0], self.mapa_color.shape[1]), dtype=int)
        
        # Definir colores a mapear
        # (Verde) Área navegable
        self.matrix[np.all(self.mapa_color == [255, 255, 255], axis=-1)] = 0  # Área libre

        # (Rojo) Obstáculo
        self.matrix[np.all(self.mapa_color == [0, 0, 0], axis=-1)] = 1  # Obstáculo
        
        # (Azul) Objetivo
        self.matrix[np.all(self.mapa_color == [0, 0, 255], axis=-1)] = 2  # Objetivo

        # (Blanco) Inicio
        self.matrix[np.all(self.mapa_color == [255, 0, 0], axis=-1)] = 3  # Punto de inicio


    def esta_cerca_de_obstaculo(self, posicion_jugador, umbral=20):
        """
        Detecta si el jugador está cerca de un obstáculo considerando un umbral de distancia.

        :param posicion_jugador: Tupla (x, y) con la posición del jugador en píxeles.
        :param umbral: Distancia mínima en píxeles para considerar que el jugador está cerca de un obstáculo.
        :return: True si el jugador está cerca de un obstáculo, False en caso contrario.
        """
        if self.matrix is None:
            return False  # Si la matriz de navegación no ha sido generada aún.

        x_jugador, y_jugador = posicion_jugador
        
        # Verificar si alguna celda de la matriz de navegación es un obstáculo (valor 1)
        for i in range(max(0, x_jugador - umbral), min(self.matrix.shape[0], x_jugador + umbral)):
            for j in range(max(0, y_jugador - umbral), min(self.matrix.shape[1], y_jugador + umbral)):
                if self.matrix[i, j] == 1:  # Si encontramos un obstáculo
                    return True
        
        return False
