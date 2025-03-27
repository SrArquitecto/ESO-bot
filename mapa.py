import numpy as np
from deteccion import YoloModelInterface, YoloModel
from segmentacion import MaskGeneratorInterface, BinaryMaskGenerator
import cv2
import heapq

class Mapa():
    def __init__(self):
        self.nodos = []
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
                    self.generar_mapa(segmentador.obtener_mascara(), (960, 540), nodos)
                    self.mask_to_navigation_matrix()
                    closest_target = self.get_closest_target((960,540), nodos)
                    if closest_target is not None:
                        path = self.astar((960,540), closest_target)
                        self.dibujar_ruta(path)
                    else: print("No se encontro ruta")
                    
                else:
                    self.generar_mapa(segmentador.obtener_mascara(), (xp, yp))
            else:
                self.generar_mapa(segmentador.obtener_mascara())
        self.mostrar_mapa()
           


    def generar_nodos_temporales(self, mapa, espaciado=10, tamano_nodo=10):
        filas, columnas = mapa.shape
        nodos_temporales = []

        # Generar nodos temporales con espaciado hasta el final de la máscara
        for x in range(0, filas - tamano_nodo, espaciado):
            for y in range(0, columnas - tamano_nodo, espaciado):
                if mapa[x, y] == 255:  # Solo agregar si es zona navegable
                    nodos_temporales.append((y, x))  # Invertir coordenadas para OpenCV

        return nodos_temporales
    
    def generar_mapa(self, mapa, posicion_jugador=(), nodos=[]):
        self.nodos = []
        nodos_temporales = self.generar_nodos_temporales(mapa * 255)
        self.nodos.extend(nodos_temporales)
        
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


    # Función de distancia Manhattan
    def manhattan_distance(self, start, end):
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    # Obtener el objetivo más cercano
    def is_blocked(self, point):
        x, y = point
        return self.matrix[y, x] == 1  # 1 representa un obstáculo en el mapa

    # Obtener el objetivo más cercano, pero asegurarse de que no esté bloqueado
    def get_closest_target(self, start, targets):
        closest_target = None
        min_distance = float('inf')
        
        for target in targets:
            if self.is_blocked(target):  # Si el objetivo está bloqueado, lo ignoramos
                continue
            
            dist = self.manhattan_distance(start, target)
            if dist < min_distance:
                min_distance = dist
                closest_target = target
        
        return closest_target  # Devolver None si no hay objetivos válidos

    # Implementación del A* básico
    def astar(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))  # (f, nodo)
        came_from = {}
        g_costs = {start: 0}
        f_costs = {start: self.manhattan_distance(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                # Reconstruir el camino
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Revertir el camino

            for neighbor in self.get_neighbors(current):
                if self.matrix[neighbor[1], neighbor[0]] == 1:  # Obstáculo
                    continue

                tentative_g_cost = g_costs[current] + 1  # Suponiendo costo uniforme

                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    came_from[neighbor] = current
                    g_costs[neighbor] = tentative_g_cost
                    f_costs[neighbor] = tentative_g_cost + self.manhattan_distance(neighbor, goal)
                    heapq.heappush(open_list, (f_costs[neighbor], neighbor))

        return None  # No se encontró un camino

    # Función para obtener los vecinos del nodo actual
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.matrix[0]) and 0 <= ny < len(self.matrix):
                neighbors.append((nx, ny))
        return neighbors

    def dibujar_ruta(self, ruta):
        if ruta:  
            for i in range(len(ruta) - 1):
                start_point = (ruta[i][0], ruta[i][1])
                end_point = (ruta[i + 1][0], ruta[i + 1][1])
                cv2.line(self.mapa_color, start_point, end_point, (0, 255, 255), 2)

