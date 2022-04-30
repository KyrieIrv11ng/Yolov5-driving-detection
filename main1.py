import sys
import os

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
import math


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


text = []
flag = True

#视频转换为图像
def video_convert2_images(video,convert_image):

    vc = cv2.VideoCapture(video)  # 读入视频文件，命名cv
    n = 1  # 计数

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False


    timeF = 2  # 视频帧计数间隔频率

    i = 0
    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            i += 1
            print(i)
            cv2.imwrite(convert_image.format(i), frame)  # 存储为图像
        n = n + 1
        # cv2.imshow("1",frame)
        # cv2.waitKey(0)
    vc.release()




#读取文件夹下图像
def read_directory(directory_name):
    array_of_img = []
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    img_name_list = os.listdir(directory_name)
    img_name_list.sort(key=lambda x: int(x[:-4]))  # 将'.jpg'左边的字符转换成整数型进行排序

    for filename in img_name_list:
        #print(filename) #just for test
        if(filename.endswith(".jpg") or filename.endswith(".png")):

            #img is used to store the image data
            img = cv2.imread(directory_name + "/" + filename)
            array_of_img.append(img)
            #print(img)
            # print(array_of_img)
    return array_of_img


def show_pic(frame):
    info = []
    # 全局变量
    # 在函数中引入定义的全局变量
    global EYE_AR_THRESH,EYE_AR_CONSEC_FRAMES,MAR_THRESH,MOUTH_AR_CONSEC_FRAMES,COUNTER,TOTAL,mCOUNTER,mTOTAL,ActionCOUNTER,Roll,Rolleye,Rollmouth,ALARM_ON,phone_num,drink_num,smok_num,flag


    # 检测
    # 将摄像头读到的frame传入检测函数myframe.frametest()
    ret,frame = myframe.frametest(frame)
    lab,eye,mouth = ret

    info.append("eye opening:" + str(eye))
    info.append("mouth opening:" + str(mouth))


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
                    text.append("正在玩手机...\n")
                    info.append("taking phone")


                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                    phone_num += 20


            if(i == "smoke"):
                smok_num += 1
                if smok_num%20==0:
                    text.append("正在抽烟...\n")
                    info.append("smoking")

            if(i == "drink"):
                drink_num += 1
                if drink_num%20 == 0:
                    text.append("正在喝水...\n")
                    info.append("drinking")


        # 如果超过15帧未检测到分心行为，将label修改为平时状态
        if ActionCOUNTER == 50:

            ActionCOUNTER = 0

        # 疲劳判断
        # 眨眼判断
        if eye==0:
            info.append("cant detect eyes")
        elif eye < EYE_AR_THRESH:
            info.append("eyes closing")
            # 如果眼睛开合程度小于设定好的阈值
            # 则两个和眼睛相关的计数器加1
            COUNTER += 1
            Rolleye += 1
        else:
            info.append("eyes opening")
            # 如果连续2次都小于阈值，则表示进行了一次眨眼活动
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                text.append("第"+str(TOTAL)+'次眨眼\n')
                # 重置眼帧计数器
                COUNTER = 0

        # 哈欠判断，同上
        if mouth == 0:
            info.append("cant detect mouth")
        elif mouth > MAR_THRESH:
            info.append("yawning")
            mCOUNTER += 1
            Rollmouth += 1
        else:
            info.append("not yawn")
            # 如果连续3次都小于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                mTOTAL += 1
                text.append("第"+str(mTOTAL)+'次打哈欠\n')
                # 重置嘴帧计数器
                mCOUNTER = 0

                flag = True


        # 每打5次哈欠，表示有犯困的可能性,就进行提醒
        if mTOTAL!=0 and mTOTAL%5==0 and flag is True:
            text.append("打哈欠已经{}次了，有疲劳的风险存在...\n".format(mTOTAL))
            flag = False



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
            text.append("过去40帧中，Perclos得分为"+str(round(perclos,3))+'\n')

            # 当过去的50帧中，Perclos模型得分超过0.38时，判断为疲劳状态

            if perclos > 0.12 and perclos<0.6:
                # print(perclos)
                text.append("当前处于疲劳驾驶状态\n")


            if perclos > 0.6:
                # print(perclos)
                text.append("当前处于梦游驾驶状态\n")


            if perclos < 0.12:
                text.append("当前处于清醒状态\n")


            # 归零
            # 将三个计数器归零
            # 重新开始新一轮的检测
            Roll = 0
            Rolleye = 0
            Rollmouth = 0
            text.append("重新开始执行疲劳检测...\n")

    else:
        TOTAL = 0
        ActionCOUNTER = 0
        mTOTAL = 0
        text.append("无人驾驶状态...\n")


    # 将每帧图像检测信息显示在图像上
    y0, dy = 40, 10
    i = 0
    for txt in info:
        y = y0 + i * dy
        i = i + 1
        cv2.putText(frame, txt, (0, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)

    return frame


#检测疲劳
def detect_fatigue(input_dir,output_dir):


    # 读取需要检测的文件夹
    img_list = read_directory(input_dir)
    index = 0
    for img in img_list:
        index = index + 1
        # 检测
        img_detect = show_pic(img)
        #输出检测结果
        output_filename = os.path.join(output_dir,'{}.jpg'.format(index))
        cv2.imwrite(output_filename,img_detect)

        # cv2.imshow("1",img_detect)
        # cv2.waitKey(0)

    # 将所有帧检测结果输出到result.txt文本
    output_result_filename = os.path.join(output_dir,'detect_result.txt')


    file = open(output_result_filename, 'w')
    file.write("".join(text))
    file.close()





if __name__ == '__main__':
    #视频转换为图像
    # video_convert2_images(r'../Yolov5-driving-detection/video/2.mp4',r'../Yolov5-driving-detection/images/2/{}.jpg')

    # (需要检测的图像文件夹路径,检测结果文件夹路径)
    detect_fatigue(r'../Yolov5-driving-detection/images/resolution_augment',
                   r'../Yolov5-driving-detection/images/detect_after_resolution_augment')
