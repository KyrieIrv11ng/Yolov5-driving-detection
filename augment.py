import cv2
import numpy as np
import math

import os
# 视角变换扩增方法

# 计算角度
def rad(x):
    return x * np.pi / 180

    # 双线性差值算法
def biLinearInterpolation(img, dstH, dstW):
    dstH = int(dstH)
    dstW = int(dstW)
    scrH, scrW, channel = img.shape
    img = np.pad(img, ((0, 1), (0, 1), (0, 0)), 'constant')
    retimg = np.zeros((int(dstH), int(dstW), 3), dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx = (i + 1) * (scrH / dstH) - 1
            scry = (j + 1) * (scrW / dstW) - 1
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            retimg[i, j] = (1 - u) * (1 - v) * img[x, y] + u * (1 - v) * img[x + 1, y] + (1 - u) * v * img[
                x, y + 1] + u * v * img[x + 1, y + 1]
    return retimg


#视角变换扩增图像
def perspectiveTrans(img, anglex, angley, anglez, H):

    # 扩展图像，保证内容不超出可视范围
    img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)
    h, w = img.shape[0:2]
    size = [w, h]
    fov = 21

    # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))

    z_old = z
    # 高度变换
    z = z + H

    # 如果变换小于0，重新生成
    if (z < 0):
        z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))

    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    # 四对点的生成
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    # print(dst1, dst2, dst3, dst4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    # print(dst)
    # 生成透视变换矩阵
    warpR = cv2.getPerspectiveTransform(org, dst)
    # print(warpR)
    # opencv透视变换
    result = cv2.warpPerspective(img, warpR, (h, w))

    height, width = result.shape[0:2]

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    xmin = width
    ymin = height
    xmax = 0
    ymax = 0
    # 裁剪掉多余的黑色背景
    for i in range(height):
        for j in range(width):
            if (gray[i, j] != 0):
                xmin = min(xmin, j)
                xmax = max(xmax, j)
                ymin = min(ymin, i)
                ymax = max(ymax, i)


    # 裁剪最后得到的图像
    result = result[ymin:ymax, xmin:xmax]


    result = biLinearInterpolation(result,(ymax - ymin)*z/z_old,(xmax - xmin)*z/z_old)


# 缩放所变换的标注坐标不需要重新变换
    return result


# 分辨率变换模型
# ----------------------------------------------------------------------------------------
# 参数控制：
# defect_img：缺陷图像；类型：img
# s：缩放倍数；取值范围：（0.001,1）；类型：double；示例：2
# 返回值：img
def resolutionTrans(img, s):

    rows, cols, channels = img.shape
    img_large = biLinearInterpolation(img, rows * s, cols * s)
    rows, cols, channels = img_large.shape
    img_small = biLinearInterpolation(img_large, rows / s, cols / s)

    return img_small


#扩增图像
def augment(input_img_dir,output_img_dir,para):
    array_of_img = read_directory(input_img_dir)
    index = 0
    for img in array_of_img:
        img_new = perspectiveTrans(img,para[0],para[1],para[2],para[3])

        img_new_new = resolutionTrans(img_new,para[4])
        # ret, frame = myframe.frametest(img)
        # lab, eye, mouth = ret
        # print(lab,eye,mouth)
        filename = output_img_dir+'/'+str(index)+".jpg"
        # print(filename)
        index = index + 1
        print("扩增第{}张图片".format(index))
        cv2.imwrite(filename,img_new)
        # cv2.imshow("res",img_new)
        # cv2.waitKey(0)

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
        print(filename)

    return array_of_img




if __name__ == '__main__':
    #[相机上下转动角度，相机左右转动角度,不调整,相机视距,分辨率缩小因子]
    para = [0,0,0,0,0.6]

    # (原图像路径/扩增图像路径)
    augment(r'../Yolov5-driving-detection/images/video2images',
            r'../Yolov5-driving-detection/images/resolution_augment',
            para)
