import sys
import os
from glob import glob
from PySide2 import QtWidgets,QtCore,QtGui
from PySide2.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide2.QtCore import QDir, QTimer,Slot
from PySide2.QtGui import QPixmap,QImage
from ui_mainwindow import Ui_MainWindow
import cv2
import myframe
import winsound
import pyglet  # 获取报警资源
from threading import Thread
import dlib  # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import wx  # 构造显示界面的GUI
import wx.xrc
import wx.adv
from scipy.spatial import distance as dist  # 欧几里得距离
from imutils import face_utils  # 一系列使得opencv 便利的功能，包括图像旋转、缩放、平移，骨架化、边缘检测、显示
import numpy as np  # 数据处理的库 numpy
import argparse
import time
import math
from pydub import AudioSegment
from pydub.playback import play

# 终端参数构建
ALARM_ON = False
ap = argparse.ArgumentParser()
# 默认为0，就是摄像头获取视频
ap.add_argument("-w", "--webcam", type=int, default=0,help="index of webcam on system")
ap.add_argument("-a", "--alarm",help="index of alarm on system")
args = vars(ap.parse_args())

# 定义三个变量，分别用来控制识别的结果
phone_num = 0
drink_num = 0
smok_num = 0

# 眼睛闭合判断
EYE_AR_THRESH = 0.2        # 眼睛长宽比
EYE_AR_CONSEC_FRAMES = 3    # 闪烁阈值

#嘴巴开合判断
MAR_THRESH = 0.65           # 打哈欠长宽比
MOUTH_AR_CONSEC_FRAMES = 3  # 闪烁阈值

# 定义检测变量，并初始化
COUNTER = 0                 #眨眼帧计数器
TOTAL = 0                   #眨眼总数
mCOUNTER = 0                #打哈欠帧计数器
mTOTAL = 0                  #打哈欠总数
ActionCOUNTER = 0           #分心行为计数器器

# 疲劳判断变量
Roll = 0                    #整个循环内的帧
Rolleye = 0                 #循环内闭眼帧数
Rollmouth = 0               #循环内打哈欠数

path="phone.wav"
# 报警声函数定义
def sound_alarm(path):
    # play an alarm sound
    # music = pyglet.resource.media(path)
    # music.play()
    # pyglet.app.run()
    alarm = AudioSegment.from_wav(path)
    play(alarm)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # 打开文件类型，用于类的定义
        self.f_type = 0

    def window_init(self):
        # 设置控件属性
        # 设置label的初始值
        self.label.setText("请打开摄像头")
        self.label_2.setText("疲劳检测：")
        self.label_3.setText("眨眼次数：0")
        self.label_4.setText("哈欠次数：0")
        self.label_5.setText("行为检测：")
        self.label_6.setText("手机")
        self.label_7.setText("抽烟")
        self.label_8.setText("喝水")
        self.label_9.setText("是否存在分心行为")
        self.label_10.setText("是否为疲劳状态")
        self.menu.setTitle("打开")
        self.actionOpen_camera.setText("打开摄像头")
        # 菜单按钮 槽连接 到函数
        self.actionOpen_camera.triggered.connect(CamConfig_init)
        # 自适应窗口缩放
        self.label.setScaledContents(True)

# 定义摄像头类
class CamConfig:
    global ALARM_ON
    def __init__(self):
        Ui_MainWindow.printf(window,"正在打开摄像头请稍后...")
        # 设置时钟
        self.v_timer = QTimer()
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap:
            Ui_MainWindow.printf(window,"打开摄像头失败")
            return
        # 设置定时器周期，单位毫秒
        self.v_timer.start(10)
        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)
        # 在前端UI输出提示信息
        Ui_MainWindow.printf(window,"载入成功，开始运行程序")
        Ui_MainWindow.printf(window,"")
        Ui_MainWindow.printf(window,"开始执行疲劳检测...")
        window.statusbar.showMessage("正在使用摄像头...")

    def show_pic(self):
        # 全局变量
        # 在函数中引入定义的全局变量
        global EYE_AR_THRESH,EYE_AR_CONSEC_FRAMES,MAR_THRESH,MOUTH_AR_CONSEC_FRAMES,COUNTER,TOTAL,mCOUNTER,mTOTAL,ActionCOUNTER,Roll,Rolleye,Rollmouth,ALARM_ON,phone_num,drink_num,smok_num

        # 读取摄像头的一帧画面
        success, frame = self.cap.read()
        print(type(frame))
        if success:
            # 检测
            # 将摄像头读到的frame传入检测函数myframe.frametest()
            ret,frame = myframe.frametest(frame)
            lab,eye,mouth = ret
            if len(ret[0])>0:
                # ret和frame，为函数返回
                # ret为检测结果，ret的格式为[lab,eye,mouth],lab为yolo的识别结果包含'phone' 'smoke' 'drink',eye为眼睛的开合程度（长宽比），mouth为嘴巴的开合程度
                # frame为标注了识别结果的帧画面，画上了标识框
                # 分心行为判断
                # 分心行为检测以50帧为一个循环
                ActionCOUNTER += 1

                # 如果检测到分心行为
                # 将信息返回到前端ui，使用红色字体来体现
                # 并加ActionCOUNTER减1，以延长循环时间
                for i in lab:
                    if(i == "phone"):
                        phone_num += 1
                        if phone_num%20==0:
                            path = "sound/phone.wav"
                            window.label_6.setText("<font color=red>正在用手机</font>")
                            window.label_9.setText("<font color=red>请不要玩手机......</font>")
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=sound_alarm,args=(path,))
                                t.setDaemon(True)
                                t.start()




                            if ActionCOUNTER > 0:
                                ActionCOUNTER -= 1
                            phone_num += 20
                    else:
                        ALARM_ON = False

                    if(i == "smoke"):
                        smok_num += 1
                        if smok_num%20==0:
                            path = "sound/somking.wav"
                            window.label_7.setText("<font color=red>正在抽烟</font>")
                            window.label_9.setText("<font color=red>请把烟熄灭......</font>")
                            if not ALARM_ON:
                                ALARM_ON = True
                                t1 = Thread(target=sound_alarm,args=(path,))
                                t1.setDaemon(True)
                                t1.start()
                            if ActionCOUNTER > 0:
                                ActionCOUNTER -= 1
                            smok_num += 20
                    else:
                        ALARM_ON = False

                    if(i == "drink"):
                        drink_num += 1
                        if drink_num%20 == 0:
                            path = "sound/drinking.wav"
                            window.label_8.setText("<font color=red>正在喝水</font>")
                            window.label_9.setText("<font color=red>请不要喝水了......</font>")
                            if not ALARM_ON:
                                ALARM_ON = True
                                t2 = Thread(target=sound_alarm,args=(path,))
                                t2.setDaemon(True)
                                t2.start()
                            if ActionCOUNTER > 0:
                                ActionCOUNTER -= 1
                            drink_num += 20
                    else:
                        ALARM_ON = False

                # 如果超过15帧未检测到分心行为，将label修改为平时状态
                if ActionCOUNTER == 50:
                    window.label_6.setText("手机")
                    window.label_7.setText("抽烟")
                    window.label_8.setText("喝水")
                    window.label_9.setText("")
                    ActionCOUNTER = 0

                # 疲劳判断
                # 眨眼判断
                if eye < EYE_AR_THRESH:
                    # 如果眼睛开合程度小于设定好的阈值
                    # 则两个和眼睛相关的计数器加1
                    COUNTER += 1
                    Rolleye += 1
                else:
                    # 如果连续2次都小于阈值，则表示进行了一次眨眼活动
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        window.label_3.setText("眨眼次数：" + str(TOTAL))
                        # 重置眼帧计数器
                        COUNTER = 0

                # 哈欠判断，同上
                if mouth > MAR_THRESH:
                    mCOUNTER += 1
                    Rollmouth += 1
                else:
                    # 如果连续3次都小于阈值，则表示打了一次哈欠
                    if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                        mTOTAL += 1
                        window.label_4.setText("哈欠次数：" + str(mTOTAL))
                        # 重置嘴帧计数器
                        mCOUNTER = 0

                # 每打5次哈欠，表示有犯困的可能性,就进行提醒
                if mTOTAL!=0 and mTOTAL%5==0:
                    window.label_10.setText("<font color=red>有疲劳的风险存在...</font>")
                    path = "sound/tired_sound.wav"
                    if not ALARM_ON:
                        ALARM_ON = True
                        t2 = Thread(target=sound_alarm, args=(path,))
                        print('ing....tired_sound')
                        t2.setDaemon(True)
                        t2.start()
                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                    mTOTAL+=1
                else:
                    ALARM_ON = False

                # 将画面显示在前端UI上
                show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                window.label.setPixmap(QPixmap.fromImage(showImage))

                # 疲劳模型
                # 疲劳模型以50帧为一个循环
                # 每一帧Roll加1
                Roll += 1
                # 当检测满40帧时，计算模型得分
                if Roll == 40:
                    # 计算Perclos模型得分
                    perclos = (Rolleye / Roll)
                    print(perclos)
                    # print(perclos)
                    # 在前端UI输出perclos值
                    Ui_MainWindow.printf(window,"过去50帧中，Perclos得分为"+str(round(perclos,3)))
                    # 当过去的50帧中，Perclos模型得分超过0.38时，判断为疲劳状态

                    if perclos > 0.12 and perclos<0.6:
                        # print(perclos)
                        Ui_MainWindow.printf(window,"当前处于疲劳驾驶状态")
                        window.label_10.setText("<font color=red>已处于疲劳驾驶状态！</font>")
                        path = "sound/tired.wav"
                        if not ALARM_ON:
                            ALARM_ON = True
                            t2 = Thread(target=sound_alarm, args=(path,))
                            print('ing....tired')
                            t2.setDaemon(True)
                            t2.start()
                        if ActionCOUNTER > 0:
                            ActionCOUNTER -= 1
                        Ui_MainWindow.printf(window,"")
                    else:
                        ALARM_ON = False

                    if perclos > 0.6:
                        # print(perclos)
                        Ui_MainWindow.printf(window,"当前处于梦游驾驶状态")
                        window.label_10.setText("<font color=red>已处于梦游驾驶状态！</font>")
                        path = "sound/sleep.wav"
                        if not ALARM_ON:
                            ALARM_ON = True
                            t2 = Thread(target=sound_alarm, args=(path,))
                            t2.setDaemon(True)
                            t2.start()
                        if ActionCOUNTER > 0:
                            ActionCOUNTER -= 1
                        Ui_MainWindow.printf(window, "")
                        Ui_MainWindow.printf(window,"")
                    else:
                        ALARM_ON = False

                    if perclos < 0.12:
                        Ui_MainWindow.printf(window,"当前处于清醒状态")
                        window.label_10.setText("清醒")
                        Ui_MainWindow.printf(window,"")

                    # 归零
                    # 将三个计数器归零
                    # 重新开始新一轮的检测
                    Roll = 0
                    Rolleye = 0
                    Rollmouth = 0
                    Ui_MainWindow.printf(window,"重新开始执行疲劳检测...")
            else:
                TOTAL = 0
                ActionCOUNTER = 0
                mTOTAL = 0
                window.label_6.setText("手机")
                window.label_7.setText("抽烟")
                window.label_8.setText("喝水")
                window.label_3.setText("眨眼次数：" + str(TOTAL))
                window.label_4.setText("哈欠次数：" + str(mTOTAL))
                window.label_9.setText("")
                window.label_10.setText("<font color=red>无人驾驶状态...</font>")

def CamConfig_init():
    window.f_type = CamConfig()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.window_init()
    window.show()
    sys.exit(app.exec_())