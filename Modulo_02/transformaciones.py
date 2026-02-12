import numpy as np
import cv2

class TransformacionesEuclideanas:
    @staticmethod
    def rotacion(angulo):
        return np.array([
            [np.cos(angulo), -np.sin(angulo), 0],
            [np.sin(angulo), np.cos(angulo), 0],
            [0, 0, 1]
        ])
    
    @staticmethod
    def traslado(x,y):
        return np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])
    
    @staticmethod
    def afin(angulo=0, x=0, y=0, escala=1):
        return np.array([])

    @staticmethod
    def reflexion(eje='x'):
        if eje == 'x':  # Refleja sobre eje Y
            return np.array([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        elif eje == 'y':  # Refleja sobre eje X
            return np.array([[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, 1]])
        elif eje == 'origen':  # Refleja sobre origen
            return np.array([[-1, 0, 0],
                             [0, -1, 0],
                             [0, 0, 1]])
        
    @staticmethod
    def transformacion(imagen, M_inv, dsize):
        w,h = dsize
        imagen_transformada = np.zeros((h,w), dtype=imagen.dtype)
        print(imagen.shape, imagen_transformada.shape)
        print(h, w)
        for x in range(w):
            for y in range(h):
                p_destino_h = np.array([x, y, 1])
                p_origen_h = M_inv @ p_destino_h

                x_origen = p_origen_h[0] / p_origen_h[2]
                y_origen = p_origen_h[1] / p_origen_h[2]

                #Si está dentro de la imagen original
                if 0 <= x_origen < w and 0 <= y_origen < h:
                # Interpolación por vecino más cercano
                    y_idx = int(round(y_origen))%h
                    x_idx = int(round(x_origen))%w
                    imagen_transformada[y, x] = imagen[y_idx, x_idx]
        return imagen_transformada
    
    def transformacion_opencv(imagen, M, dsize):
        return cv2.warpAffine(imagen, M, dsize)
