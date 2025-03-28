import numpy as np
from deteccion import YoloModelInterface, YoloModel
from segmentacion import MaskGeneratorInterface, BinaryMaskGenerator
import cv2
import heapq
import pyautogui


class Mapa():
    def __init__(self):
        self.ruta_filtrada = []
        self.nodo_mas_cercano = None
        self.ruta = []
        self.mapa_color = np.ones((1080, 1920), dtype=np.uint8) * 255
        self.mapa_navegacion = None

    def run(self, mask, coord_jugador, coord_nodos):

        self.generar_mapa(mask, coord_jugador, coord_nodos)
        
        closest_target = self.get_closest_target(coord_jugador, coord_nodos)
        print(closest_target)
        path = []
        if closest_target is not None:
            path = self.astar(coord_jugador, closest_target)
        if path:
            self.dibujar_ruta(path)
            print(path)
            self._filter_nodes_by_distance(path, step=100)
            print(self.ruta_filtrada)
            self._draw_nodes_on_map()
        self.mostrar_mapa()


        """
            nodos = []
            pos_jugador = detector.()
            if pos_jugador is not None:

                nodos_raw = detector.obtener_nodos()
                if nodos_raw is not None:
                    for nodo in nodos_raw:

                    self.generar_mapa(np.ones((1080, 1920), dtype=np.uint8), (xp, yp), nodos)
                    self.mask_to_navigation_matrix()
                    
                    if closest_target is not None:
                        
                        self.dibujar_ruta(path)
                        self._extraer_ruta(path)
                        print("RUTA:")
                        if self.ruta:
                            print(self.ruta)
                            
                            print("RUTA FILTRADA:")
                            print(self.ruta_filtrada)
                            self._draw_nodes_on_map()
                            if len(self.ruta_filtrada) >= 2:
                                self._align_camera_to_node(self.ruta_filtrada[1][0])
                    else: print("No se encontro ruta")
                    
                else:
                    self.generar_mapa(np.ones((1080, 1920), dtype=np.uint8), (xp, yp))
            else:
                self.generar_mapa(np.ones((1080, 1920), dtype=np.uint8))
        self.mostrar_mapa()
        """
    def generar_mapa(self, mapa, posicion_jugador, nodos):
        if mapa.ndim == 2:
            self.mapa_color = cv2.cvtColor(mapa * 255, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("El mapa debe ser una imagen en escala de grises")
        
        for nodo in nodos:
            cv2.circle(self.mapa_color, nodo, 5, (0, 0, 255), -1)
        if posicion_jugador:
            cv2.circle(self.mapa_color, posicion_jugador, 5, (255, 0, 0), -1)

        # Mostrar el mapa generado para depuración
        self._mask_to_navigation_matrix()

    def mostrar_mapa(self):  
        cv2.imshow("Mapa Generado", self.mapa_color)

        
    

    # Función para leer la máscara RGB y convertirla a una matriz de navegación
    def _mask_to_navigation_matrix(self):
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

    def euclidean_distance(self, start, end):
        return np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

    # Obtener el objetivo más cercano
    def is_blocked(self, point):
        x, y = point
        return self.matrix[y, x] == 1  # 1 representa un obstáculo en el mapa

    # Obtener el objetivo más cercano, pero asegurarse de que no esté bloqueado
    def get_closest_target(self, start, targets):
        closest_target = None
        min_distance = float('inf')

        # Filtramos los objetivos bloqueados para evitar cálculo innecesario
        valid_targets = [target for target in targets if not self.is_blocked(target)]
        
        for target in valid_targets:
            dist = self.manhattan_distance(start, target)
            if dist < min_distance:
                min_distance = dist
                closest_target = target

        return closest_target  # Devolver None si no hay objetivos válidos# Devolver None si no hay objetivos válidos

    # Implementación del A* básico


    def astar(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))  # Colocamos el nodo inicial
        came_from = {}
        g_costs = {start: 0}  # Costo acumulado desde el inicio
        f_costs = {start: self.euclidean_distance(start, goal)}  # Costo total estimado
        visited = set()  # Conjunto de nodos visitados

        while open_list:
            _, current = heapq.heappop(open_list)  # Obtenemos el nodo con el costo más bajo

            if current == goal:
                # Reconstruir la ruta desde el nodo final
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Devuelve la ruta invertida (de start a goal)

            visited.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in visited:  # Ignorar los nodos ya visitados
                    continue

                tentative_g_cost = g_costs[current] + 1  # Costo acumulado al vecino

                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    came_from[neighbor] = current
                    g_costs[neighbor] = tentative_g_cost
                    f_costs[neighbor] = tentative_g_cost + self.euclidean_distance(neighbor, goal)
                    heapq.heappush(open_list, (f_costs[neighbor], neighbor))  # Agregar el vecino a la cola

        return None  # No se encontró ruta

    # Función para obtener los vecinos del nodo actual
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (1, 1), (1, -1), (-1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.matrix[0]) and 0 <= ny < len(self.matrix):
                neighbors.append((nx, ny))
        return neighbors

    def dibujar_ruta(self, path):
        if path:  
            for i in range(len(path) - 1):
                start_point = (path[i][0], path[i][1])
                end_point = (path[i + 1][0], path[i + 1][1])
                cv2.line(self.mapa_color, start_point, end_point, (0, 255, 255), 2)
                

    def _extraer_ruta(self, ruta):
        self.ruta = []
        if ruta:
            for punto in ruta:
                self.ruta.append((punto[0], punto[1]))



    # -------------------------------
    # Filtrado de nodos
    # -------------------------------
    def _filter_nodes_by_distance(self, path, step=100):
        self.ruta_filtrada = []
        """Filtra los nodos para optimizar la ruta, tomando uno cada 'step' píxeles."""
        if not path:
            self.ruta_filtrada = []
        
        if path:
            self.ruta_filtrada = [path[0]]  # Primer nodo (posición inicial)
            last_node = path[0]

            for node in path[1:]:
                dist = np.linalg.norm(np.array(node) - np.array(last_node))
                if dist >= step:
                    self.ruta_filtrada.append(node)
                    last_node = node

            if self.ruta_filtrada[-1] != path[-1]:
                self.ruta_filtrada.append(path[-1])

    # -------------------------------
    # Dibujo de nodos en el mapa
    # -------------------------------
    def _draw_nodes_on_map(self, color=(0, 255, 0), radius=3):
        """Dibuja los nodos filtrados en el mapa."""
        for node in self.ruta_filtrada:
            cv2.circle(self.mapa_color, node, radius, color, -1)
        return self.mapa_navegacion

    # -------------------------------
    # Centrado directo de la cámara en X
    # -------------------------------
    def _align_camera_x_to_node(self, node_x, tolerance=5):
        """Mueve el ratón proporcionalmente para centrar el nodo en una sola iteración."""
        screen_width, _ = pyautogui.size()
        center_x = screen_width // 2

        delta_x = node_x - center_x

        if abs(delta_x) < tolerance:
            return  # El nodo ya está suficientemente centrado

        # Movimiento proporcional directo
        pyautogui.move(delta_x, 0, duration=0.1)
    def _align_camera_x_to_node2(self, node_x):
        """Mueve el ratón directamente para centrar el nodo en una sola iteración."""
        screen_width, _ = pyautogui.size()
        center_x = screen_width // 2

        # Movimiento directo en lugar de proporcional
        pyautogui.moveTo(node_x, pyautogui.position().y, 0)
    # -------------------------------
    # Comprobación de nodo fuera de la vista
    # -------------------------------
    def _is_node_in_view(self, node_x, margin=0.1):
        """Verifica si el nodo está dentro de una zona segura en pantalla."""
        screen_width, _ = pyautogui.size()
        left_limit = screen_width * margin
        right_limit = screen_width * (1 - margin)

        return left_limit < node_x < right_limit
