import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import sys
import argparse
parser = argparse.ArgumentParser(description = "Description for my parser")
parser.add_argument("-c", "--camSource", help = "source of camera (0 is default", required = False, default = "0")
argument = parser.parse_args()

mp_face_mesh = mp.solutions.face_mesh
serverAddress = ("127.0.0.1", 7070)
irisSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# 0 is for default camera and 1 for first usb Camera connected
#camSource = int(sys.argv[1])
camsource = int(argument.camSource)
cap = cv.VideoCapture(0)

#indices da iris
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [33]     # Left eye Left Corner
L_H_RIGHT = [133]   # Left eye Right Corner
R_H_LEFT = [362]    # Right eye Left Corner
R_H_RIGHT = [263]   # Right eye Right Corner

def euclidean_distance(point1, point2):
    x1, y1 =point1.ravel()
    x2, y2 =point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def vectorPos(point1, point2):
    x1, y1 =point1.ravel()
    x2, y2 =point2.ravel()
    dx = x2-x1
    dy = y2-y1
    return dx,dy

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position =""
    if ratio <= 0.42:
        iris_position="right"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position="center"
    else:
        iris_position = "left"
    return iris_position, ratio

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)   #Mediapipe  RGB ,  OpenCV BGR
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            #transformar formas quadradas em círculos, função do OpenCV fornece círculos delimitadores com base nos pontos fornecidos.
            #minEnclosingCircle que retorna, o centro (x,y) e o raio dos círculos, os valores de retorno são de ponto flutuante, necessario transformá-los em int.
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])


            # transforma pontos centrais em array np
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            

            #desenhe o círculo com base nos valores de retorno da minEnclosingCircle, através do CIRCLE que desenha a imagem do círculo com base no centro (x, y) e no raio
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

            #mostrar pontos nos cantos dos olhos
            cv.circle(frame, mesh_points[L_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)

            iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
            l_dx, l_dy = vectorPos(mesh_points[L_H_LEFT],center_left)


            #print(iris_pos)
            print("Left Eye Center X: %(lx)s Y: %(ly)s" % {'lx': l_cx, 'ly': l_cy})
            print("Right Eye Center X: %(rx)s Y: %(ry)s" % {'rx': r_cx, 'ry': r_cy})
            print("Left Iris Relative Pos Dx: %(dx)s Dy: %(dy)s" % {'dx': l_dx, 'dy': l_dy})
            packet = np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32)
            print("\n")
            irisSocket.sendto(bytes(packet), ("127.0.0.1",7070))
        cv.imshow("img", frame)
        key = cv.waitKey(1)
        if key ==ord("q"):
            break
cap.release()
cv.destroyAllWindows()