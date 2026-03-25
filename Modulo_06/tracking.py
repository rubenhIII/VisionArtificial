import cv2
import numpy as np

# Parametros para Lucas Kanade
lk_params = dict( winSize  = (14, 14),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parametros para el detector de esquinas
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)

# Crea paleta de colores para los puntos a trackear
color = np.random.randint(0, 255, (100, 3))

def get_corners(img, corner_mask):
    corners = cv2.goodFeaturesToTrack(img, mask = corner_mask, **feature_params)
    return corners

def draw_corners(img, corners):
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),10,255,-1)

# Captura desde archivo de video
video = cv2.VideoCapture("img/robot.mp4")
print("Oprime q para salir de la captura de video")

while video.isOpened():
    ret, old_frame = video.read()
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(old_frame)

    corner_mask = np.zeros_like(old_frame, dtype=np.uint8)
    height, width = old_frame.shape
    corner_mask[1*height//8:8*height//8, 1*width//10:9*width//10] = 255

    p0 = get_corners(old_frame, corner_mask)

    while True:
        ret, frame = video.read()
        if not ret:
            # Reproduce infinitamente el video: Optimizar
            video = cv2.VideoCapture("img/robot.mp4")
            ret, old_frame = video.read()
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(old_frame)
            corner_mask = np.zeros_like(old_frame, dtype=np.uint8)
            height, width = old_frame.shape
            corner_mask[2*height//5:5*height//5, 1*width//10:9*width//10] = 255
            p0 = get_corners(old_frame, corner_mask)
            ret, frame = video.read()

            print("No hay frames por leer")
            #break

        frame_col = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calcula Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
        # Seleccionar buenos puntos / esquinas
        if p1 is not None:
            good_new = p1[st==1 & (err < 10)]
            good_old = p0[st==1 & (err < 10)]

        # Dibujar el seguimiento
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame_col = cv2.circle(frame_col, (int(a), int(b)), 5, color[i].tolist(), -1)
        
        # img:  Imagen para debug: Mascara usada en la detección de esquinas o lineas de track
        # frame_col: track de las esquinas con colores
        img = cv2.add(frame, corner_mask)
        cv2.imshow("Video", img)

        old_frame = frame.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # Re-detectar puntos si se pierden muchos (menos del 10%)
        if len(p0) < feature_params['maxCorners'] * 0.10:
            p0 = get_corners(old_frame, corner_mask)
            mask = np.zeros_like(frame)  # Reinicia máscara
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    break

video.release()
cv2.destroyAllWindows()