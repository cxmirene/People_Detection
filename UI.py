# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import time
import numpy as np
import cv2 as cv
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QPixmap
from datetime import datetime
from Detect import Detect
from SGBM import SGBM

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.timer_camera = QtCore.QTimer()
        self.set_ui()
        self.slot_init()
        self.TIME = 0
        self.detector = Detect()
        self.sgbm = SGBM()
        self.sgbm.Init_SGBM()
        self.Camera = cv.VideoCapture()
        self.CAM_NUM = 1

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.button_start = QtWidgets.QPushButton('开始识别')  # 建立用于打开摄像头的按键
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_model2 = QtWidgets.QPushButton('MobileNet2')  # 建立用于退出程序的按键
        self.button_model3 = QtWidgets.QPushButton('MobileNet3')  # 建立用于退出程序的按键
        self.button_start.setMinimumHeight(200)  # 设置按键大小
        self.button_close.setMinimumHeight(200)
        self.button_model2.setMinimumHeight(200)
        self.button_model3.setMinimumHeight(200)

        self.button_close.move(10, 100)  # 移动按键
        # '''信息显示'''
        # self.textEdit = QTextEdit()
        # self.textEdit.setFixedSize(400, 800)
        '''显示的视频窗口'''
        self.pix = QPixmap('./background.jpg')
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(960, 720)  # 给显示视频的Label设置大小为641x481
        self.label_show_camera.setStyleSheet('background-color:rgb(96,96,96)')#设置背景颜色
        self.label_show_camera.setPixmap(self.pix)
        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_main.addWidget(self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        # self.__layout_main.addWidget(self.textEdit)
        '''把按键加入到按键布局中'''
        self.__layout_fun_button.addWidget(self.button_start)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_model2)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_model3)  # 把打开摄像头的按键放到按键布局中
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_start.clicked.connect(self.button_start_clicked)  # 若该按键被点击，则调用button_start_clicked()
        self.button_model2.clicked.connect(self.button_model2_clicked)  # 若该按键被点击，则调用button_start_clicked()
        self.button_model3.clicked.connect(self.button_model3_clicked)  # 若该按键被点击，则调用button_start_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
    '''槽函数之一'''
    def button_start_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.Camera.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self,'warning',"请检查相机于电脑是否连接正确",buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(1)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_start.setText('结束识别')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.Camera.release()
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_start.setText('开始识别')

    def button_model2_clicked(self):
        self.detector.Init_Net("mb2-ssd")

    def button_model3_clicked(self):
        self.detector.Init_Net("mb3-large-ssd")

    def SegmentFrame(self, Fream):
        double = cv.resize(Fream, (640, 240), cv.INTER_AREA)
        left = double[0:240,0:320]
        right = double[0:240,320:640]
        return left, right

    def show_camera(self):
        # Camera = cv.VideoCapture(1)
        # if not Camera.isOpened():
        #     print("Could not open the Camera")
        #     sys.exit()

        ret, Fream = self.Camera.read()
        # cv.imwrite("Two.jpg",Fream)
        os.system("./camera.sh")
        while(1):
            ret, Fream = self.Camera.read()
            if not ret:
                break
            LeftImage, RightImage = self.SegmentFrame(Fream)
            # start = time()
            result_rect = self.detector.detect(LeftImage)
            distance, disp = self.sgbm.Coordination(LeftImage, RightImage, result_rect)
            result = LeftImage.copy()
            for i in range(0,len(result_rect)):
                cv.rectangle(result, (result_rect[i][0], result_rect[i][1]), (result_rect[i][2], result_rect[i][3]), (255, 255, 0), 4)
                cv.putText(result, str(distance[i]),
                                (result_rect[i][0] + 20, result_rect[i][1] + 40),
                                cv.FONT_HERSHEY_SIMPLEX,
                                1,  # font scale
                                (255, 0, 255),
                                2)  # line type
            # end = time()
            # seconds = end - start
            # fps = 1/seconds
            # print( "Estimated frames per second : {0}".format(fps))
            # cv.imshow("left",LeftImage)
            # cv.imshow("right",RightImage)
            # cv.imshow("disp",disp)
            # cv.imshow("result", coordinate)
            disp = cv.cvtColor(disp, cv.COLOR_GRAY2BGR)
            htich1 = np.hstack((LeftImage, RightImage))
            htich2 = np.hstack((disp, result))
            vtich = np.vstack((htich1, htich2))
            # cv.imshow("result", vtich)

            if cv.waitKey(1)==ord('q'):
                break
            show = cv.resize(vtich, (960, 720))  # 把读到的帧的大小重新设置为 1280x960
            show = cv.cvtColor(show, cv.COLOR_BGR2RGB)
        # while(1):
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
            return

# app = QtWidgets.QApplication(sys.argv)
# ui = Ui_MainWindow()
# ui.show()
# sys.exit(app.exec_())
