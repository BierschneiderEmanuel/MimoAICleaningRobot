
# MIT License

# Copyright (c) [2025] [Emanuel Bierschnedier]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
import cv2
import numpy as np
import time
import math
import os
import struct
import socket
#speach
import speech_recognition as sr
import pyttsx3
from base64 import b64decode
from pathlib import Path
from io import BytesIO
from urllib.parse import urlencode
import subprocess
from PIL import Image
#mediapipe
import mediapipe as mp
#bakllava
from tokenize import Pointfloat
from langchain_community.llms import Ollama
import base64
from IPython.display import HTML, display
from PIL import Image
import speech_recognition as sr
from pathlib import Path
from io import BytesIO
from urllib.parse import urlencode
import requests
import json
import jsonstreams
import os, sys
#yolo
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

sys.path.append(os.getcwd())
# yolov5 img path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
path_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

IS_CV_4 = cv2.__version__[0] == '4'
__version__ = "1.0"

X_MAX_RES_BAKLLAVA = 800
Y_MAX_RES_BAKLLAVA = 600
X_MAX_RES_ROBOT_ROS = 1920
Y_MAX_RES_ROBOT_ROS = 1080
X_MAX_RES_YOLO_V5 = 640
Y_MAX_RES_YOLO_V5 = 640
# Filename 
FILENAME = 'detect_facemesh_ai.jpg'
FILENAME_CROPPED = 'detect_facemesh_ai_0.jpg'
FILENAME_YOLO = 'detect_facemesh_ai_yolo.jpg'
class YoloInferencer:
    def __init__(self):
        #person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, 
        #bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, 
        #skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, 
        #fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, sofa, pottedplant, 
        #bed, diningtable, toilet, tvmonitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, 
        #clock, vase, scissors, teddy bear, hair drier, toothbrush
       
         print( """YOLOv8 model.""")
         self.model = YOLO("yolov8n.pt")

    def predict(self, img):
        results = self.model.predict(img)
        result_list = []
        for result in results:
            for box in result.boxes:
                left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
                width = right - left
                height = bottom - top
                center = (left + int((right-left)/2), top + int((bottom-top)/2))
                label = results[0].names[int(box.cls)]
                confidence = float(box.conf.cpu())
                cv2.rectangle(img, (left, top),(right, bottom), (255, 0, 0), 2)
                cv2.putText(img, label + " " + str(confidence), (left, bottom+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                result_list.append([label, left, top, right, bottom, float(confidence)])
        cv2.imshow('Filtered Frame', img)
        return result_list    

class FaceMeshDetector:
    """
    Face Mesh Detector to find 468 Landmarks using the mediapipe library.
    Helps acquire the landmark points in pixel format
    """

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        :param staticMode: In static mode, detection is done on each image: slower
        :param maxFaces: Maximum number of faces to detect
        :param minDetectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Finds face landmarks in BGR Image.
        :param img: Image to find the face landmarks in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, #self.mpFaceMesh.FACE_CONNECTIONS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    def findDistance(self,p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info

    def putTextRect(self, img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                    colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                    offset=10, border=None, colorB=(0, 255, 0)):
        """
        Creates Text with Rectangle Background
        :param img: Image to put text rect on
        :param text: Text inside the rect
        :param pos: Starting position of the rect x1,y1
        :param scale: Scale of the text
        :param thickness: Thickness of the text
        :param colorT: Color of the Text
        :param colorR: Color of the Rectangle
        :param font: Font used. Must be cv2.FONT....
        :param offset: Clearance around the text
        :param border: Outline around the rect
        :param colorB: Color of the outline
        :return: image, rect (x1,y1,x2,y2)
        """
        ox, oy = pos
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

        x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

        cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
        if border is not None:
            cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
        cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

        return img, [x1, y2, x2, y1]

class MycobotServer(object):
    def __init__(self, host, port):
        HOST = '192.168.178.31'    # The remote host
        PORT = 9000                # The same port as used by the server
        s = None
        for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM):
            af, socktype, proto, canonname, sa = res
            try:
                self.s = socket.socket(af, socktype, proto)
            except OSError as msg:
                self.s = None
                continue
            try:
                self.s.connect(sa)
            except OSError as msg:
                self.s.close()
                self.s = None

    def close_connection(self):
        if self.s != None:
            print("Closing connection...")
            self.s.close()
        else:
            print("Connect socker error...")

    def get_coords(self):
        try:
            print("sendall get_coords")
            values = [float(9999.9), float(0.0), float(0.0), float(0.0), float(0.0), float(0.0), float(0.0)]
            data = struct.pack("<7f", *values)
            self.s.sendall(data)
            while True:
                print("await response------------------")
                try:
                    print("await request response get coords--------")
                    data_answ = self.s.recv(28)
                    values_answ = struct.unpack("<7f", data_answ)
                    rounded_val = [round(each_val, 2) for each_val in values_answ]
                    rcv_coords = [values_answ[1], values_answ[2], values_answ[3], values_answ[4], values_answ[5], values_answ[6]]
                    print("rcv coords: ", rcv_coords)
                    return rcv_coords
                except Exception as e:
                    print("str: ie", str(e))
                    self.close_connection()
        except Exception as e:
            print("str: e", str(e))
            self.close_connection()

    def get_angles(self):
        try:
            print("sendall get_angles")
            values = [float(5555.5), float(0.0), float(0.0), float(0.0), float(0.0), float(0.0), float(0.0)]
            data = struct.pack("<7f", *values)
            self.s.sendall(data)
            while True:
                print("await response------------------")
                try:
                    print("wait request response get angles--------")
                    data_answ = self.s.recv(28)
                    values_answ = struct.unpack("<7f", data_answ)
                    rounded_val = [round(each_val, 2) for each_val in values_answ]
                    rcv_angles = [values_answ[1], values_answ[2], values_answ[3], values_answ[4], values_answ[5], values_answ[6]]
                    print("rcv angles: ", rcv_angles)
                    return rcv_angles
                except Exception as e:
                    print("str: ie", str(e))
                    self.close_connection()
        except Exception as e:
            print("str: e", str(e))
            self.close_connection()

    def is_gripper_moving(self):
        try:
            print("sendall is_gripper_moving")
            values = [float(2222.2), float(0.0), float(0.0), float(0.0), float(0.0), float(0.0), float(0.0)]
            data = struct.pack("<7f", *values)
            self.s.sendall(data)
            while True:
                print("await response------------------")
                try:
                    print("await request response is gripper moving--------")
                    data_answ = self.s.recv(28)
                    values_answ = struct.unpack("<7f", data_answ)
                    rounded_val = [round(each_val, 1) for each_val in values_answ]
                    rcv_is_gripper_moving_arr = [rounded_val[1], rounded_val[2], rounded_val[3], rounded_val[4], rounded_val[5], rounded_val[6]]
                    rcv_is_gripper_moving, _= rcv_is_gripper_moving_arr[:2]
                    is_gripper_moving = int(rcv_is_gripper_moving)
                    print("rcv is_gripper_moving: ", is_gripper_moving)
                    return int(is_gripper_moving)
                except Exception as e:
                    print("str: ie", str(e))
                    self.close_connection()
        except Exception as e:
            print("str: e", str(e))
            self.close_connection()

    def send_coords(self, coords_to_send):
        try:
            print("sendall send_coords")
            coords_arr = [float(8888.8), coords_to_send[0], coords_to_send[1], coords_to_send[2], coords_to_send[3], coords_to_send[4], coords_to_send[5]]
            coords_data = struct.pack("<7f", *coords_arr)
            self.s.sendall(coords_data)
        except Exception as e:
            print("str: e", str(e))
            self.close_connection()

    def send_angles(self, angles_to_send):
        try:
            print("sendall send_angles")
            angles_arr = [float(6666.6), angles_to_send[0], angles_to_send[1], angles_to_send[2], angles_to_send[3], angles_to_send[4], angles_to_send[5]]
            angles_data = struct.pack("<7f", *angles_arr)
            self.s.sendall(angles_data)
        except Exception as e:
            print("str: e", str(e))
            self.close_connection()

    def set_gripper_state(self, gripper_state, gripper_speed):
        try:
            print("sendall send_gripper_state")
            gripper_state_to_send = [float(gripper_state), float(gripper_speed), float(0), float(0), float(0), float(0)]
            gripper_set_arr = [float(3333.3), gripper_state_to_send[0], gripper_state_to_send[1], gripper_state_to_send[2], gripper_state_to_send[3], gripper_state_to_send[4], gripper_state_to_send[5]] #gripper_state_to_send[0] actual gripper_state and gripper_state_to_send[1] gripper duration ctrl flag
            gripper_data = struct.pack("<7f", *gripper_set_arr)
            self.s.sendall(gripper_data)
        except Exception as e:
            print("str: e", str(e))
            self.close_connection()

    def send_cartesian_waypoint(self, cartesian_waypoint_to_send):
        try:
            print("sendall send_cartesian_waypoint")
            cartesian_waypoint_arr = [float(11111.1), cartesian_waypoint_to_send[0], cartesian_waypoint_to_send[1], cartesian_waypoint_to_send[2], cartesian_waypoint_to_send[3], cartesian_waypoint_to_send[4], cartesian_waypoint_to_send[5]]
            cartesian_waypoint_data = struct.pack("<7f", *cartesian_waypoint_arr)
            self.s.sendall(cartesian_waypoint_data)
        except Exception as e:
            print("str: e", str(e))
            self.close_connection()

    def send_cartesian_waypoint_stop_and_reverse(self, cartesian_waypoint_to_send):
        try:
            print("sendall send_cartesian_waypoint_stop_and_reverse")
            cartesian_waypoint_arr = [float(22222.2), cartesian_waypoint_to_send[0], cartesian_waypoint_to_send[1], cartesian_waypoint_to_send[2], cartesian_waypoint_to_send[3], cartesian_waypoint_to_send[4], cartesian_waypoint_to_send[5]]
            cartesian_waypoint_data = struct.pack("<7f", *cartesian_waypoint_arr)
            self.s.sendall(cartesian_waypoint_data)
        except Exception as e:
            print("str: e", str(e))
            self.close_connection()         

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class Object_detect():
    def __init__(self, camera_x=150, camera_y=-10):
        # inherit the parent class
        super(Object_detect, self).__init__()
        # get path of file
        dir_path = os.path.dirname(__file__)
        #Joint Limit:
        #joint 1: -170 ~ +170
        #joint 2: -170 ~ +170
        #joint 3: -170 ~ +170
        #joint 4: -170 ~ +170
        #joint 5: -170 ~ +170
        #joint 6: -180 ~ +180
        # which robot: USB* is m5; ACM* is wio; AMA* is raspi
        self.robot_m5 = os.popen("ls /dev/ttyUSB*").readline()[:-1]
        self.robot_wio = os.popen("ls /dev/ttyACM*").readline()[:-1]
        self.robot_raspi = os.popen("ls /dev/ttyAMA*").readline()[:-1]
        self.robot_jes = os.popen("ls /dev/ttyTHS1").readline()[:-1]
        self.raspi = False
        if "dev" in self.robot_m5:
            self.Pin = [2, 5]
        elif "dev" in self.robot_wio:
            self.Pin = [20, 21]
            for i in self.move_coords:
                i[2] -= 20
        elif "dev" in self.robot_raspi or "dev" in self.robot_jes:
            import RPi.GPIO as GPIO

        # choose place to set cube
        self.color = 0
        # parameters to calculate camera clipping parameters
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        # set cache of real coord
        self.cache_x = self.cache_y = 0
        # set color HSV
        self.HSV = {
            "yellow": [np.array([11, 115, 70]), np.array([40, 255, 245])],
            "red": [np.array([0, 43, 46]), np.array([8, 255, 255])],
            "green": [np.array([35, 43, 46]), np.array([77, 255, 255])],
            "blue": [np.array([100, 43, 46]), np.array([124, 255, 255])],
            "cyan": [np.array([78, 43, 46]), np.array([99, 255, 255])],
        }
        # use to calculate coord between cube and mycobot
        self.sum_x1 = self.sum_x2 = self.sum_y2 = self.sum_y1 = 0
        # The coordinates of the grab center point relative to the mycobot
        self.camera_x, self.camera_y = camera_x, camera_y
        # The coordinates of the cube relative to the mycobot
        self.c_x, self.c_y = 0, 0
        # The ratio of pixels to actual values
        self.ratio = 0

    # init mycobot
    def run(self):
        for _ in range(5):
            #mycobot.pub_angles([65, -3, -90, 90, 10, 0], 50) #set face
            print(_)
            time.sleep(0.5)

    # draw aruco
    def draw_marker(self, img, x, y):
        # draw rectangle on img
        cv2.rectangle(
            img,
            (x - 20, y - 20),
            (x + 20, y + 20),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.FONT_HERSHEY_COMPLEX,
        )
        # add text on rectangle
        cv2.putText(img, "({},{})".format(x, y), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (243, 0, 0), 2,)

    # get points of two aruco
    def get_calculate_params(self, img):
        # Convert the image to a gray image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect ArUco marker.
        corners, ids, rejectImaPoint = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        """
        Two Arucos must be present in the picture and in the same order.
        There are two Arucos in the Corners, and each aruco contains the pixels of its four corners.
        Determine the center of the aruco by the four corners of the aruco.
        """
        if len(corners) > 0:
            if ids is not None:
                if len(corners) <= 1 or ids[0] == 1:
                    return None
                x1 = x2 = y1 = y2 = 0
                point_11, point_21, point_31, point_41 = corners[0][0]
                x1, y1 = int((point_11[0] + point_21[0] + point_31[0] + point_41[0]) / 4.0), int(
                    (point_11[1] + point_21[1] + point_31[1] + point_41[1]) / 4.0)
                point_1, point_2, point_3, point_4 = corners[1][0]
                x2, y2 = int((point_1[0] + point_2[0] + point_3[0] + point_4[0]) / 4.0), int(
                    (point_1[1] + point_2[1] + point_3[1] + point_4[1]) / 4.0)
                return x1, x2, y1, y2
        return None

    # set camera clipping parameters
    def set_cut_params(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        print(self.x1, self.y1, self.x2, self.y2)

    # set parameters to calculate the coords between cube and mycobot
    def set_params(self, c_x, c_y, ratio):
        self.c_x = c_x
        self.c_y = c_y
        self.ratio = 220.0/ratio

    # calculate the coords between cube and mycobot
    def get_position(self, x, y):
        return ((y - self.c_y)*self.ratio + self.camera_x), ((x - self.c_x)*self.ratio + self.camera_y)

    def transform_frame(self, frame):
        # enlarge the image by 1.5 times
        fx = 1.5
        fy = 1.5
        frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy,
                           interpolation=cv2.INTER_CUBIC)
        if self.x1 != self.x2:
            # the cutting ratio here is adjusted according to the actual situation
            frame = frame[int(self.y2*0.2):int(self.y1*1.15),
                          int(self.x1*0.7):int(self.x2*1.15)]
        return frame

    def detect_aruco(self, gray):
        # Get ArUco marker dict that can be detected.
        aruco_dict_val = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Get ArUco marker params.
        aruco_params_val = cv2.aruco.DetectorParameters_create()
        camera_matrix_val = np.array([
            [781.33379113, 0., 347.53500524],
            [0., 783.79074192, 246.67627253],
            [0., 0., 1.]])
        dist_coeffs_val = np.array(([[3.41360787e-01, -2.52114260e+00, -1.28012469e-03, 6.70503562e-03, 2.57018000e+00]]))
        corners, ids, rejectImaPoint = cv2.aruco.detectMarkers(gray, aruco_dict_val, parameters=aruco_params_val)
        if len(corners) > 0:
            if ids is not None:
                # get informations of aruco # '''https://stackoverflow.com/questions/53303730/what-is-the-value-for-markerlength-in-aruco-estimateposesinglemarkers'''
                ret = cv2.aruco.estimatePoseSingleMarkers(corners, 0.025, camera_matrix_val, dist_coeffs_val)
                # rvec:rotation offset,tvec:translation deviator
                (rvec, tvec) = (ret[0], ret[1])	    
                (rvec - tvec).any()
                xyz = tvec[0, 0, :] * 1000
                rpy = rvec[0,0,:]  
        return  corners


    def generate_pipe_json_response(self, img_dat, prmpt="Is it dirty or messy in the picture?"):
        data = json.dumps({"model": "bakllava", "prompt": "Is it dirty or messy in the picture?", "images": img_dat})
        full_response = ""; line = ""
        with jsonstreams.Stream(jsonstreams.Type.array, filename='./response_log.txt') as output:
            while True:
                try: 
                    line = requests.post("http://localhost:11434/api/generate", data = data)
                except IOError as e: 
                        print("An exception occurred:", e.errno)
                if not line:
                    break
                try:
                    record = line.text.split("\n")
                    for i in range(len(record)-1):
                        data = json.loads(record[i].replace('\0', ''))
                        if "response" in data:
                            full_response += data["response"]
                            with output.subobject() as output_e:
                                output_e.write('response', data["response"])
                        else:
                            return full_response.replace('\0', '')
                    if len(record)==1:
                        data = json.loads(record[0].replace('\0', ''))
                        if "error" in data:
                            full_response += data["error"]
                            with output.subobject() as output_e:
                                output_e.write('error', data["error"])
                    return full_response.replace('\0', '')
                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", error)
        return full_response.replace('\0', '')


    def convert_base64_destination(self, pil_image):
        """
        Convert PIL images to Base64 encoded strings
        :param pil_image: PIL image
        :return: Re-sized Base64 string
        """
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")  # You can change the format if needed
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def convert_yolo_v5_coordinates_to_ros_resolution(self, x_min, y_min, x_max, y_max):
        # convert coordinates in resolution 640x640 to ros resolution coordinates
        new_x_min = int((x_min / X_MAX_RES_YOLO_V5)*X_MAX_RES_ROBOT_ROS)
        new_y_min = int((y_min / Y_MAX_RES_YOLO_V5)*Y_MAX_RES_ROBOT_ROS)
        new_x_max = int((x_max / X_MAX_RES_YOLO_V5)*X_MAX_RES_ROBOT_ROS)
        new_y_max = int((y_max / Y_MAX_RES_YOLO_V5)*Y_MAX_RES_ROBOT_ROS)
        return (new_x_min, new_y_min, new_x_max, new_y_max)
    
    def yolov5_detection_run(self, _yolo_inf, _cap, _engine, start_ros_movement):
        target_detected = ""; detect_center_x = 0; detect_center_y = 0; size_distance_target = 0
        success, frame_bakllava = _cap.read()
        frame_bakllava_show = cv2.resize(frame_bakllava, (X_MAX_RES_ROBOT_ROS,Y_MAX_RES_ROBOT_ROS), interpolation=cv2.INTER_AREA)
        cv2.imshow("figure", frame_bakllava_show)
        frame_bakllava_uncropped = cv2.resize(frame_bakllava, (X_MAX_RES_BAKLLAVA,Y_MAX_RES_BAKLLAVA), interpolation=cv2.INTER_AREA)
        # Saving the image 
        if os.path.exists(FILENAME):
            os.remove(FILENAME)
        cv2.imwrite(FILENAME, frame_bakllava_uncropped)
        im = Image.open(FILENAME).convert('L')
        #crop to 533x600 for BAKLLAVA
        im = im.crop((X_MAX_RES_BAKLLAVA/3, 0, X_MAX_RES_BAKLLAVA, Y_MAX_RES_BAKLLAVA)) #crop out gripper take only 2/3 of the right area #img_left_area = (0, 0, 400, 600) #img_right_area = (400, 0, 800, 600)
        if os.path.exists(FILENAME_CROPPED):
            os.remove(FILENAME_CROPPED)
        im.save(FILENAME_CROPPED)
        #overwrite org file to enable tracking of objects in x,y coordinates
        #frame_org #640x480 #yolov5 inference with 640x640 padded picture
        im_yolo_image = Image.open(FILENAME_CROPPED).convert('L')
        #crop to 640x600 to mask out any unwanted area and try to check for 
        im_yolo_image = im_yolo_image.crop((0, 0, 640, 480))#crop out gripper take only 2/3 of the right area #img_left_area = (0, 0, 400, 600) #img_right_area = (400, 0, 800, 600)
        if os.path.exists(FILENAME_CROPPED):
            os.remove(FILENAME_YOLO)
        im_yolo_image.save(FILENAME_YOLO)
        img_yolo = cv2.imread(FILENAME_YOLO)
        yolo_ret = _yolo_inf.predict(img_yolo)
        for i in range(0, len(yolo_ret)):
            if yolo_ret[i][0] == "fork" \
               or yolo_ret[i][0] == "banana": #\
                # or yolo_ret[i][0] == "person":
                target_detected = yolo_ret[i][0]
                #calculate new center target of the dectected reqctangle object
                detect_xmin, detect_ymin, detect_xmax, detect_ymax = self.convert_yolo_v5_coordinates_to_ros_resolution(yolo_ret[i][1], yolo_ret[i][2], yolo_ret[i][3], yolo_ret[i][4])
                detect_center_x = (detect_xmin + detect_xmax) / 2
                detect_center_y = (detect_ymin + detect_ymax) / 2
                width_pxl_to_size = (detect_xmax - detect_xmin) #set new width determined by object size # set depth information
                f = im_yolo_image.size[1]
                W = 3 #3cm real width of a spoon or fork
                d = (W * f) / width_pxl_to_size
                size_distance_target = d*10 #scale to cm
                size_distance_target *= 1.6875 # for 640x640 -> for 1920x1080 this is * 1.6875 (1080/640)
                print("detect_center_x", detect_center_x)
                print("detect_center_y", detect_center_y)
                print("size_distance_target", size_distance_target)
                if start_ros_movement == False: # if cleanup not started
                    try:
                        start_ros_movement = True
                        #reset flags after movement
                        #start_yolo_detection = False #reset detection flags
                        engine.say("it is messy i am starting the cleanup")
                        engine.runAndWait()
                        # Catch if error
                    except Exception as error:
                        # handle the exception
                        print("An exception occurred:", error)
                        engine.runAndWait()
        return target_detected, start_ros_movement, detect_center_x, detect_center_y, size_distance_target
        
if __name__ == "__main__":
    HOST = socket.gethostbyname(socket.gethostname())
    PORT = 9000
    mycobot = MycobotServer("192.168.178.31",PORT)
    mycobot.send_angles([float(65), float(-3), float(-90), float(90), float(10), float(10)])
    ret_coord = mycobot.get_coords()
    time.sleep(1)
    #init speach engine
    engine = pyttsx3.init()
    # open the camera
    cap_num = 0
    cap = cv2.VideoCapture(cap_num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # yolov5 init
    print("yolov5 init")
    yolo_inf = YoloInferencer()
    print("facemesh detect init")
    detector = FaceMeshDetector(staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5)
    cnt=1
    fps_limiter=0
    # init a class of Object_detect
    detect = Object_detect()
    # init mycobot
    detect.run()
    _init_ = 20
    nparams = 0
    resync_cnt = 0
    first_run = True
    frame_cnt = 0
    start_angles = [] 
    initial_angles = []
    moving_coords = []
    detect_center_x = 0
    detect_center_y = 0
    size_distance_target = 0
    set_gripper_async_flag = 0
    set_gripper_async_flag_max_cnt = 0
    set_async_flag = 0
    set_async_flag_max_cnt = 0
    start_yolo_detection = False
    start_ros_movement = False
    human_person_detected = False
    w = 0
    while initial_angles == []:
        initial_angles = mycobot.get_angles()
    while moving_coords == []:
        moving_coords = mycobot.get_coords()
    print("initial_current_cords=", moving_coords)
    while True:
        # read camera
        success, frame = cap.read()
        ####################### FACEMESH #######################
        frame = cv2.resize(frame, (X_MAX_RES_ROBOT_ROS,Y_MAX_RES_ROBOT_ROS), interpolation=cv2.INTER_AREA)
        frame_org = frame
        frame, faces = detector.findFaceMesh(frame, draw=True)
        ####################### FACEMESH #######################
        if _init_ > 0:
            print("_init", _init_)
            _init_ -= 1
            continue
        # calculate the parameters of camera clipping
        cv2.imshow("figure", frame)
        cv2.waitKey(1)

        detect_center_x = 0
        detect_center_y = 0
        # Catch if error  
        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]

            #162 21 54 103 67 109 10 338 297 332 284 251
            cnt=338
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=10
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=21
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=54
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=67
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=103
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=109
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=162
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=251
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 255), offset=10)
            cnt=284
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=297
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)
            cnt=332
            pointOne = face[cnt]
            circlepointOne = (pointOne[0], pointOne[1])
            cv2.circle(frame, circlepointOne, 5, (255, 0, 0), cv2.FILLED)
            detector.putTextRect(frame, f'{int(cnt)}', circlepointOne, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=10)

            #https://www.computervision.zone/lessons/video-lesson-16/
            width_pxl, _ = detector.findDistance(pointLeft, pointRight)
            #f = flocal lenght
            #width_pxl = width in pixels (ccd sensor width 1920)
            #d = distance in cm (object to camera)
            #W = object width in cm
            W = 6.3  #cm std. disctance between eyes is 6.3cm
            # # Finding the Focal Length
            f = frame.shape[1]
            d = (W * f) / width_pxl
            # Setting distance
            w = d
            detector.putTextRect(frame, f'Depth: {int(d)}cm', (face[10][0] - 100, face[10][1] - 50), scale=2)
            cnt=10 #center point of face
            pointOne = face[cnt]
            detect_center_x = pointOne[0]
            detect_center_y = pointOne[1]
            frame = cv2.circle(frame, (detect_center_x, detect_center_y), 10, (0, 255, 255), 2)
            size_distance_target = w
            print("distance to face: ", size_distance_target) #for mobile the gripper size of the target is:
            
            if(detect_center_x != 0 and detect_center_y != 0) and (start_ros_movement == False) and (start_yolo_detection == False):
                try:
                    engine.say("Hello my human friend!")
                    mycobot.send_angles([float(65), float(-3), float(-3), float(-12), float(-10), float(10)])
                    mycobot.send_angles([float(65), float(-3), float(-3), float(-12), float(10), float(10)])
                    mycobot.send_angles([float(65), float(-3), float(-3), float(-12), float(-10), float(10)])
                    human_person_detected = True
                    engine.runAndWait()
                    # Catch if error
                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", error)
                    engine.runAndWait()

        #preset detection data
        frame_bakllava = None
        if human_person_detected == True and start_ros_movement == False and \
           start_yolo_detection == False and (first_run == True or resync_cnt == 0 or set_gripper_async_flag == 0): # do not start detection during first run #do not detect during movement in case resync count is not 0(no movemnt in case resyc_cnt is zero)
            #  for audio input
            while True:
                success, frame_bakllava = cap.read()
                frame_bakllava = cv2.resize(frame_bakllava, (X_MAX_RES_ROBOT_ROS,Y_MAX_RES_ROBOT_ROS), interpolation=cv2.INTER_AREA)
                cv2.imshow("figure", frame_bakllava)
                # calculate the parameters of camera clipping
                cv2.waitKey(1)
                # Init speech recognizer
                r = sr.Recognizer()
                # Wait for audio input
                with sr.Microphone() as source:
                    print("Speak now:")
                    audio = r.listen(source)
                # Recognize the audio
                try:
                    response_text = ""
                    prompt = r.recognize_google(audio, language="en-EN", show_all=False)
                    print("You asked:", prompt)
                    if(("is" in prompt.lower()) or ("it" in prompt.lower())):
                        # Generate AI response
                        bakllava_frame = cv2.resize(frame_bakllava, (X_MAX_RES_BAKLLAVA,Y_MAX_RES_BAKLLAVA), interpolation=cv2.INTER_AREA)
                        # Saving the image
                        if os.path.exists(FILENAME):
                            os.remove(FILENAME)
                        cv2.imwrite(FILENAME, bakllava_frame) 
                        im = Image.open(FILENAME).convert('L')
                        #crop to 533x600 for BAKLLAVA
                        im = im.crop((X_MAX_RES_BAKLLAVA/3, 0, X_MAX_RES_BAKLLAVA, Y_MAX_RES_BAKLLAVA)) #crop out gripper take only 2/3 of the right area #img_left_area = (0, 0, 400, 600) #img_right_area = (400, 0, 800, 600)
                        if os.path.exists(FILENAME_CROPPED):
                            os.remove(FILENAME_CROPPED)
                        im.save(FILENAME_CROPPED)
                        pil_image = Image.open(FILENAME_CROPPED)
                        image_b64 = detect.convert_base64_destination(pil_image)
                        str_prompt = str(prompt.lower())
                        llm_with_image_context = detect.generate_pipe_json_response([image_b64], str_prompt)
                        if llm_with_image_context != "":
                            print(llm_with_image_context)
                            # Speak the response
                            engine.say(llm_with_image_context)
                            engine.runAndWait()
                            frame = frame_bakllava #use new image for processing
                            if(("messy" in llm_with_image_context.lower()) or ("is dirty" in llm_with_image_context.lower()) or ("yes" in llm_with_image_context.lower())):
                                start_yolo_detection = True # set yolo inference start flag because its messy
                            break              
                # Catch if error
                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", error)
                    engine.runAndWait()

        if start_yolo_detection == True:
            target_detected, start_ros_movement, detect_center_x, detect_center_y, size_distance_target = detect.yolov5_detection_run(yolo_inf, cap, engine, start_ros_movement)

        #run yolo5 here set new target 
        if(start_ros_movement==True and target_detected!=""):
            cv2.imshow("figure", frame)
            print("size_distance_target ", size_distance_target)
            print("first_run", first_run)
            #ctrl gripper
            set_gripper_async_flag_max_cnt = 5 #20
            if size_distance_target < 40 and set_gripper_async_flag < set_gripper_async_flag_max_cnt: #set_async_flag run count <20                       
                use_gripper_flag = 1
                if use_gripper_flag == 1:
                    gripper_open = 0
                    if set_gripper_async_flag == 0 or set_gripper_async_flag < set_gripper_async_flag_max_cnt/2:
                        set_gripper_async_flag = 1
                        mycobot.set_gripper_state(gripper_open,95)
                        print("set gripper open: {}".format(gripper_open))
                        time.sleep(1)
                    else:

                        set_gripper_async_flag += 1
                        gripper_close = 1
                        mycobot.set_gripper_state(gripper_close,95)
                        print("set gripper close: {}".format(gripper_close))
                        mycobot.send_angles([float(90), float(-20), float(-20), float(-20), float(0), float(0)])
                        start_ros_movement = False #reset movement flags after gripper action

            frame_limit_num = 2 #25
            frame_resync_limit_num = 5 #35
            print("set_gripper_async_flag", set_gripper_async_flag)
            print("frame_limit_num", frame_limit_num)
            print("resync_cnt", resync_cnt)
            print("frame_cnt", frame_cnt)
            if (first_run == True or resync_cnt > frame_resync_limit_num):
                resync_cnt = 0
                if set_gripper_async_flag >= set_gripper_async_flag_max_cnt:
                    set_gripper_async_flag = 0
                    print("set gripper async flag STOP: ", set_gripper_async_flag)
                print(" === RESYNC === ") #ROS2 KDL IK Solver orientation
                print("  Y     X       ")
                print("  |    /        ")
                print("  |   /         ")
                print("  |  /          ")
                print("  | /           ")
                print("  |/-----------Z")

                # initial starting point to initiate tracking
                ##############################
                # CHANGE CHANG ###############
                ########################## 
                ######!!!!!!!!!!!!!!!!#########
                temp_x = detect_center_x
                temp_y = detect_center_y                  
                temp_z = size_distance_target
                # initial starting point to initiate static movement
                initial_size_distance_target = w
                print("initial_target_size ", initial_size_distance_target)
                print("initial_angles=", initial_angles)
                print("initial last detecting point_x,y,z=", temp_x, temp_y, temp_z)
                moving_coords = []
                while moving_coords == []:
                    moving_coords = mycobot.get_coords()
                print("c_cords=", moving_coords)
                first_run = False
            else: # limit to frame_limit_num frames
                if frame_cnt > frame_limit_num:
                    print("____________________________________")
                    frame_cnt = 0
                    # calculate current error
                    error_x = detect_center_x - temp_x
                    error_y = detect_center_y - temp_y
                    error_z = size_distance_target - temp_z
                    print("error_x,y,z=", error_x, error_y, error_z)
                    # project in cartesian form for inverse kinematics of a planar robot and scale 
                    # interchanged y and z for own solver here because of own ik
                    # gripper points up
                    #   Y     X
                    #   |    /
                    #   |   /
                    #   |  /
                    #   | /
                    #   |/-----------Z
                    rel_diff_z = -error_y * 0.15   #0.15   *0.15 because 1920x1080 needs to be normalized to centimeter of robot
                    rel_diff_y = -error_x * 0.15   #0.15   *0.15 because 1920x1080 needs to be normalized to centimeter of robot
                    rel_diff_x = -error_z * 1      #0.5    *1 because tras distance needs to be normalized to centimerter
                    ###rel_diff_x = +_BArel_diff_x #+rel_diff_x is move back if tras comes near and w is small in case face is near
                    rel_diff_x = -rel_diff_x #-rel_diff_x is move forward if tras comes near and w is small in case tras is near
                    moving_coords[0] += rel_diff_x
                    moving_coords[2] += rel_diff_y
                    print("w: ", w)
                    print("rel_diff_z,y,x=", rel_diff_z, rel_diff_y, rel_diff_x)
                    # calculate current error
                    rel_diff_z_min = 5
                    rel_diff_z_max = 40
                    rel_diff_y_min = 5
                    rel_diff_y_max = 80
                    rel_diff_x_min = 5
                    rel_diff_x_max = 80
                    # prohibit move to objects out of reach
                    if (abs(rel_diff_z) > rel_diff_z_min and abs(rel_diff_z) < rel_diff_z_max) or \
                        (abs(rel_diff_y) > rel_diff_y_min and abs(rel_diff_y) < rel_diff_y_max) or \
                        (abs(rel_diff_x) > rel_diff_x_min and abs(rel_diff_x) < rel_diff_x_max):
                        scaler = 1000.0
                        if set_async_flag == 0: #only execute if there is no async movement
                            #x forward / backward #y left / right #z up / down
                            mycobot.send_cartesian_waypoint([float(rel_diff_x)/scaler, float(rel_diff_y)/scaler, float(rel_diff_z)/scaler, float(0), float(0), float(0)])
                            #start_ros_movement = False #reset movement flags after gripper action
                            start_yolo_detection = False #reset detection flags
                        # update resync counter due to rounding and other precision limitations
                        print("  ")
                        resync_cnt += 1
                    else: # if no movment, restore last state
                        if(abs(rel_diff_z) < rel_diff_z_min or abs(rel_diff_z) > rel_diff_z_max):
                            print("-->DIFF Z ERROR<--")
                        if(abs(rel_diff_y) < rel_diff_y_min or abs(rel_diff_y) > rel_diff_y_max):
                            print("-->DIFF Y ERROR<--")
                        if(abs(rel_diff_x) < rel_diff_x_min or abs(rel_diff_x) > rel_diff_x_max):
                            print("-->DIFF X ERROR<--") 
                        moving_coords[0] -= rel_diff_x
                        moving_coords[2] -= rel_diff_y
                    print("#################################")
                else: # set update frame rate to 5 and ensure enought time for the actual movement
                    if frame_cnt < 2: #5
                        temp_x = detect_center_x
                        temp_y = detect_center_y
                        temp_z = size_distance_target
                    frame_cnt += 1
