# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 20, 301, 361))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(440, 20, 301, 361))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.LoadCitra = QtWidgets.QPushButton(self.centralwidget)
        self.LoadCitra.setGeometry(QtCore.QRect(60, 390, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.LoadCitra.setFont(font)
        self.LoadCitra.setObjectName("LoadCitra")
        self.Proses = QtWidgets.QPushButton(self.centralwidget)
        self.Proses.setGeometry(QtCore.QRect(280, 390, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Proses.setFont(font)
        self.Proses.setObjectName("Proses")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(440, 410, 191, 31))
        self.comboBox.setObjectName("comboBox")
        self.LabelRec = QtWidgets.QLabel(self.centralwidget)
        self.LabelRec.setGeometry(QtCore.QRect(440, 390, 301, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.LabelRec.setFont(font)
        self.LabelRec.setObjectName("LabelRec")
        self.Custom = QtWidgets.QPushButton(self.centralwidget)
        self.Custom.setGeometry(QtCore.QRect(660, 410, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Custom.setFont(font)
        self.Custom.setObjectName("Custom")
        self.huelabel = QtWidgets.QLabel(self.centralwidget)
        self.huelabel.setGeometry(QtCore.QRect(60, 440, 281, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.huelabel.setFont(font)
        self.huelabel.setObjectName("huelabel")
        self.satlabel = QtWidgets.QLabel(self.centralwidget)
        self.satlabel.setGeometry(QtCore.QRect(60, 460, 281, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.satlabel.setFont(font)
        self.satlabel.setObjectName("satlabel")
        self.vallabel = QtWidgets.QLabel(self.centralwidget)
        self.vallabel.setGeometry(QtCore.QRect(60, 480, 281, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.vallabel.setFont(font)
        self.vallabel.setObjectName("vallabel")
        self.tonelabel = QtWidgets.QLabel(self.centralwidget)
        self.tonelabel.setGeometry(QtCore.QRect(60, 500, 281, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tonelabel.setFont(font)
        self.tonelabel.setObjectName("tonelabel")
        self.lipstik = QtWidgets.QLabel(self.centralwidget)
        self.lipstik.setGeometry(QtCore.QRect(440, 450, 301, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lipstik.setFont(font)
        self.lipstik.setObjectName("lipstik")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ToLips"))
        self.LoadCitra.setText(_translate("MainWindow", "Upload"))
        self.Proses.setText(_translate("MainWindow", "Proses"))
        self.LabelRec.setText(_translate("MainWindow", "Rekomendasi:"))
        self.Custom.setText(_translate("MainWindow", "Custom"))
        self.huelabel.setText(_translate("MainWindow", "Hue:"))
        self.satlabel.setText(_translate("MainWindow", "Saturasi:"))
        self.vallabel.setText(_translate("MainWindow", "Value:"))
        self.tonelabel.setText(_translate("MainWindow", "Warna Kulit:"))
        self.lipstik.setText(_translate("MainWindow", "Lipstik: "))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
