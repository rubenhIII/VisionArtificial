import cv2

class FaceDetector:
    def __init__(self):
        # Cargar el modelo (descarga los archivos si no los tienes.
        # Hay que asegurarse que tanto deploy como res10 correspondan 
        # al mismo modelo, sino lanzará un error.)
        prototxt = "models/deploy.prototxt"
        caffemodel = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self._net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    def detect(self, image):
        # Cargar la imagen
        alto, ancho = image.shape[:2]

        # Preprocesar la imagen al formato que espera el modelo.
        # Intenta realizar la detección. Si algo sale mal, sólo
        # se verá la cámara y el error se mostrará en consola.
        try:
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
            # Realizar la detección
            self._net.setInput(blob)
            detecciones = self._net.forward()
        except Exception as e:
            print(f"Se encontró un error: {e}")
            return image
        else:
            # Dibujar los rectángulos en las caras detectadas
            for i in range(detecciones.shape[2]):
                confianza = detecciones[0, 0, i, 2]
                if confianza > 0.5:
                    caja = detecciones[0, 0, i, 3:7] * [ancho, alto, ancho, alto]
                    (x1, y1, x2, y2) = caja.astype("int")
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            return image

class CameraManager:
    def __init__(self):
        # Captura desde la fuente default
        self._camera = cv2.VideoCapture(0)
        self._face_detector = FaceDetector()
    
    def capture(self):
        print("Oprime q para salir de la captura de camara")
        while self._camera.isOpened():
            while True:
                ret, frame = self._camera.read()
                if not ret:
                    print("Error al leer frame desde la cámara")
                    break
                frame_processed = self._face_detector.detect(frame)
                # Mostrar el resultado
                cv2.imshow("Camara", frame_processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break

        self._camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = CameraManager()
    cam.capture()