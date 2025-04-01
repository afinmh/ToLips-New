import sys
import os
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from GUI import Ui_MainWindow
import numpy as np
import dlib
import imutils

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

PREDICTOR_PATH = resource_path("shape_predictor_68_face_landmarks.dat")

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


class ShowImage(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__() 
        self.setupUi(self)
        self.Image = None 
        self.LoadCitra.clicked.connect(self.load)
        self.Proses.clicked.connect(self.facedetect)
        self.comboBox.currentIndexChanged.connect(self.printSelectedColor)
        self.Custom.clicked.connect(self.cuscolor)

    def cuscolor(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.png *.jpg *.jpeg)", options=options)
        if fileName:
            color_img = cv2.imread(fileName)
            color_center_pixel = color_img[color_img.shape[0] // 2, color_img.shape[1] // 2]
            value = [int(channel) for channel in color_center_pixel[::-1]]
            self.cust = value
            self.applyLipstick(self.path,value)
            self.comboBox.addItem("Custom") 


    def facedetect(self):
        class TooManyFaces(Exception):
            pass

        class NoFaces(Exception):
            pass

        def get_landmarks(im):
            rects = detector(im, 1)
            if len(rects) > 1:
                raise TooManyFaces
            if len(rects) == 0:
                raise NoFaces
            return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        def annotate_landmarks(im, landmarks):
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        def crop_face(image, landmarks):
            jawline_points = landmarks[0:17]
            jawline = cv2.convexHull(jawline_points)
            mouth_points = landmarks[48:68]
            mouth = cv2.convexHull(mouth_points)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [jawline, mouth], -1, (255, 255, 255), -1)

            cropped_face = cv2.bitwise_and(image, image, mask=mask)
            x, y, w, h = cv2.boundingRect(mask)
            cropped_face = cropped_face[y:y+h, x:x+w]
            return cropped_face

        path = self.Image.copy()
        landmarks = get_landmarks(path)
        image_with_landmarks = annotate_landmarks(path, landmarks)
        cropped_face = crop_face(path, landmarks)
        self.Image = image_with_landmarks
        self.displayImage(2)


        hsv_image = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 50, 50])
        upper_bound = np.array([179, 255, 255])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        total_pixels = cv2.countNonZero(mask)
        dominant_percentage = (total_pixels / (cropped_face.shape[0] * cropped_face.shape[1])) * 100

        if dominant_percentage > 3:  
            avg_hsv_color = cv2.mean(hsv_image, mask=mask)[:3]
            hue, saturation, value = avg_hsv_color 
            saturation_percent = (saturation / 255) * 100
            value_percent = (value / 255) * 100
            print("Hue:", int(hue))
            print("Saturation:", int(saturation_percent), "%")
            print("Value:", int(value_percent), "%")

            self.huelabel.setText("Hue: " + str(int(hue)))
            self.satlabel.setText("Saturation: " + str(int(saturation_percent)) + "%")
            self.vallabel.setText("Value: " + str(int(value_percent)) + "%")

            sawo_matang_range = {
                "hue": (8, 15),
                "saturation": (35, 63),
                "value": (46, 77)
            }

            kuning_langsat_range = {
                "hue": (9, 13),
                "saturation": (31, 56),
                "value": (72, 83)
            }

            fair_range = {
                "hue": (6, 23),
                "saturation": (26, 38),
                "value": (67, 88)
            }

            if (
                kuning_langsat_range["hue"][0] <= hue <= kuning_langsat_range["hue"][1] and
                kuning_langsat_range["saturation"][0] <= saturation_percent <= kuning_langsat_range["saturation"][1] and
                kuning_langsat_range["value"][0] <= value_percent <= kuning_langsat_range["value"][1]
            ):
                self.tonelabel.setText("Warna Kulit: Kuning Langsat")
            elif (
                sawo_matang_range["hue"][0] <= hue <= sawo_matang_range["hue"][1] and
                sawo_matang_range["saturation"][0] <= saturation_percent <= sawo_matang_range["saturation"][1] and
                sawo_matang_range["value"][0] <= value_percent <= sawo_matang_range["value"][1]
            ):
                self.tonelabel.setText("Warna Kulit: Sawo Matang")
            elif (
                fair_range["hue"][0] <= hue <= fair_range["hue"][1] and
                fair_range["saturation"][0] <= saturation_percent <= fair_range["saturation"][1] and
                fair_range["value"][0] <= value_percent <= fair_range["value"][1]
            ):
                self.tonelabel.setText("Warna Kulit: Fair / Cerah")
            else:
                self.tonelabel.setText("Warna Kulit: Tidak Dikenali")
        
            self.comboBox.clear()
            if self.tonelabel.text() == "Warna Kulit: Kuning Langsat":
                self.comboBox.addItems(["Mid-Light", "Flush It Red", "Night Purple", "Crimson Love"])
            elif self.tonelabel.text() == "Warna Kulit: Sawo Matang":
                self.comboBox.addItems(["Mid-Dark","Peach Addict", "Ambitious", "Heroine"])
            elif self.tonelabel.text() == "Warna Kulit: Fair / Cerah":
                self.comboBox.addItems(["Semua Warna", "Flush It Red", "Night Purple", "Crimson Love","Peach Addict", "Ambitious", "Heroine"])
            else:
                self.comboBox.addItems(["Default 1", "Flush It Red", "Night Purple", "Crimson Love","Peach Addict", "Ambitious", "Heroine"])
        else:
            print("Warna dominan tidak ditemukan")
        
        
    def load(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.png *.jpg *.jpeg)", options=options)
        if fileName:
            self.label.clear()
            self.path = fileName
            self.Image = cv2.imread(fileName)
            self.displayImage(1)

    def changeLipstick(self, img, value):
        img = cv2.resize(img, (0, 0), None, 1, 1)
        imgOriginal = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(imgGray)

        if len(faces) == 0:
            print("Tidak ada wajah terdeteksi.")
            return imgOriginal

        imgColorLips = np.zeros_like(img)

        for face in faces:
            facial_landmarks = predictor(imgGray, face)
            points = np.array([[facial_landmarks.part(i).x, facial_landmarks.part(i).y] for i in range(48, 61)])

            imgLips = self.getMaskOfLips(img, points)
            imgColorLips[:] = (value[2], value[1], value[0])
            imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)

            imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
            imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, 0.4, 0)

        return imgColorLips

    def getMaskOfLips(self, img, points):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        return mask

    def applyLipstick(self, face_image_path, color_values):
        face_img = cv2.imread(face_image_path)
        if face_img is None:
            print("Gambar wajah tidak dapat dibaca.")
            return None

        face_img = imutils.resize(face_img, width=400)
        value = tuple(color_values)
        imgLipstick = self.changeLipstick(face_img, value)
        self.Image = imgLipstick
        self.displayImage(2)

    def printSelectedColor(self, index):
        selected_color = self.comboBox.currentText()
        if selected_color == "Custom":
            a = self.cust        
            self.applyLipstick(self.path, a)        
        elif selected_color == "Flush It Red":
            a = (176, 8, 31)
            self.applyLipstick(self.path, a)
            self.lipstik.setText("Maybelline Sensational Liquid Matte – Flush it Red")
        elif selected_color == "Night Purple":
            a = (147, 60, 103)
            self.applyLipstick(self.path, a)
            self.lipstik.setText("Safi Matte It Perfect Lip Cream – Night Purple")
        elif selected_color == "Crimson Love":
            a = (93, 38, 70)
            self.applyLipstick(self.path, a)
            self.lipstik.setText("Silkygirl Gen Matte Lip Cream – Crimson Love")
        elif selected_color == "Peach Addict":
            a = (210, 91, 95)
            self.applyLipstick(self.path, a)
            self.lipstik.setText("Maybelline Sensational Liquid Matte – Peach Addict")
        elif selected_color == "Ambitious":
            a = (196, 8, 43)
            self.applyLipstick(self.path, a)
            self.lipstik.setText("Maybelline Superstay Matte Ink - 220 Ambitious")
        elif selected_color == "Heroine":
            a = (255, 56, 49)
            self.applyLipstick(self.path, a)
            self.lipstik.setText("Maybelline Superstay Matte Ink - 25 Heroine")
        print(selected_color)

    def displayImage(self, windows = 1 & 2):
        qformat = QImage.Format_Indexed8
        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1],self.Image.shape[0],self.Image.strides[0],qformat)
        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)
            self.label.setPixmap(QPixmap.fromImage(img))

        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label.setScaledContents(True) 

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('ToLips - Tone Up Your Lips')
window.show()
sys.exit(app.exec_())
