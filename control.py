import time
import threading
import os
import mss
import numpy as np
import cv2
from pynput import keyboard
from datetime import datetime
from deteccion import YoloModelInterface, YoloModel
from segmentacion import MaskGeneratorInterface, BinaryMaskGenerator
from mapa import Mapa
from concurrent.futures import ThreadPoolExecutor
import vgamepad as vg
import pyautogui


class Control():

    def __init__(self):
        self.capturar = False
        self.salir = False
        self.listener_thread = threading.Thread(target=self._run)
        self.listener_thread.start()
        self.mapa = None
        self.gamepad = vg.VX360Gamepad()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.f6:
                self.capturar = not self.capturar
            elif key == keyboard.Key.f12:
                self.salir = True
        except AttributeError:
            pass

    def _on_release(self, key):
        # Implement any functionality needed on key release
        pass

    def guardar_imagen(self, imagen, tstamp, nuevo_ancho=1024, nuevo_alto=576):
        """
        Guarda la imagen después de redimensionarla.
        :param imagen: Imagen a guardar (numpy array).
        :param tstamp: Timestamp para el nombre del archivo.
        :param nuevo_ancho: El nuevo ancho de la imagen después de redimensionarla.
        :param nuevo_alto: El nuevo alto de la imagen después de redimensionarla.
        """
        if not os.path.exists("./train/capturas"):
            os.makedirs("./train/capturas")
        
        # Redimensionar la imagen a las dimensiones deseadas
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
        
        # Guardar la imagen redimensionada
        with ThreadPoolExecutor() as executor:
            executor.submit(lambda: cv2.imwrite(f"./train/capturas/{tstamp}.jpg", imagen_redimensionada))

    def iniciar(self, detector: YoloModelInterface, segmentador: MaskGeneratorInterface):
            self.mapa = Mapa()
            start_time = time.time()
            tracker = None
            tracker_initialized = False
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                while True:
                    if self.salir:
                        print("adiosssssss!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        break
                    if not self.capturar:
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= 0.1:
                            screenshot = sct.grab(monitor)
                            imagen = np.array(screenshot)
                            imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
                            imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
                            imagen_gray = cv2.cvtColor(imagen_gray, cv2.COLOR_GRAY2BGR)

                            # Realizar inferencia
                            detector.inferencia(imagen_gray, dibujar=False)
                            segmentador.inferencia(imagen_bgr, dibujar=False)

                            
                            
                            self.mapa.run(segmentador.obtener_mascara(), detector.obtener_coord_jugador(), detector.obtener_coord_nodos())



                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break  # Cerrar las ventanas de OpenCV
                cv2.destroyAllWindows()
            

    def _run(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()
    
    
    def _align_camera_to_node(self, bbox, tolerance=5):
        """Mueve el ratón de forma gradual hacia el centro del nodo, comprobando si está centrado."""
        # Obtener las dimensiones de la pantalla
        center_x, center_y = 1920 // 2, 1080 // 2

         # Obtener las coordenadas del bounding box (x, y, width, height) del tracker
        _, x1, y1, x2, y2, _ = bbox

            # Calcular el centro del bounding box del tracker
        node_center_x = (x1 + x2) // 2
        node_center_y = (y1 + y2) // 2

            # Calcular la diferencia entre el centro de la pantalla y el centro del objeto
        delta_x = center_x - node_center_x
        print(delta_x)
            # Si el objeto ya está suficientemente centrado, no mover el ratón
        if abs(delta_x) < tolerance:
            print("El objeto ya está centrado.")
            return

            # Determinar la dirección en la que mover el ratón
        if delta_x > 0:
            # Si el objeto está a la izquierda del centro de la pantalla, mover a la derecha
            pyautogui.move(-50, 0)
            print(f"Moviendo ratón 50 píxeles a la derecha.")
        else:
                # Si el objeto está a la derecha del centro de la pantalla, mover a la izquierda
            pyautogui.move(50, 0)
            print(f"Moviendo ratón 50 píxeles a la izquierda.")
                # -------------------------------
                # Comprobación de nodo fuera de la vista

if __name__ == "__main__":
    control = Control()
    det= YoloModel("./models/det_nodos_ESO.pt")
    seg = BinaryMaskGenerator("./models/best_seg_obs.pt")
    control.iniciar(det, seg)
