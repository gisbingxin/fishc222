from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np


# 计算2% 98%处值作为最小 最大值
def minmaximg(width, heigh, img):
    img_fla = img.flatten()  # 展成一维数组
    img_fla = np.sort(img_fla) #排序
    uni_img = np.unique(img_fla)
    len_img = np.alen(uni_img)
    #print('len:',len_img)
    num = np.zeros(len_img, dtype=np.float32)     #唯一值数组的长度，即灰度值的个数
    print('minmaxing')
    for i in range(0,len_img):   #不同灰度值分别计数
        #print('i in len',i)
        num[i] =  np.sum(img_fla == uni_img[i])#img_fla中灰度值等于uni_img[i]的个数

    p = 0
    min = 0
    max = 0
    #print(np.sum(num))
    for j in np.arange(len_img + 1):    #从最小灰度值开始累加，当像元数量累加大于2%，98%时break
        p = num[j] / (width * heigh) + p
        #print('j in min',j)
        if p >= 0.02:
            min = uni_img[j]
            break
    p = 0

    for j in np.arange(len_img + 1):
        p = num[j] / (width * heigh) + p
        # print('/n,%', num[j])
        # print('/n,%', p)
        # print('/n,%', j)
        #print('j in max', j)
        if p >= 0.98:
            max = uni_img[j]
            break
    return min, max


def linearstretching(img, min, max):
    print('image streching.....',min,max)
    img = np.where(img > min, img, min)
    img = np.where(img < max, img, max)
    img = (img - min) / (max - min) * 255
    return img

def strechimg_pan(xsize,ysize,img_data,minmax=True):

    print('The pan image will be shown shortly')
    temp = np.empty((ysize, xsize)).astype(np.float32)

    temp = img_data

    # temp[:, :, 1] = g
    # temp[:, :, 2] = b
    temp1 = temp.astype(np.float32)
    if minmax:
        r_minvalue, r_maxvalue = minmaximg(ysize, xsize, temp1)  # 三个波段分别计算2%和98%处的灰度值，并设置最大最小值
        # g_minvalue, g_maxvalue = minmaximg(ysize, xsize, temp1[:, :, 1])
        # b_minvalue, b_maxvalue = minmaximg(ysize, xsize, temp1[:, :, 2])
        print('max in stretchpan', r_maxvalue)
        temp1 = linearstretching(temp1, r_minvalue, r_maxvalue)  # 三个波段分别将最大最小值之间的灰度值拉伸到0-255
        # temp1[:, :, 1] = linearstretching(temp1[:, :, 1], g_minvalue, g_maxvalue)
        # temp1[:, :, 2] = linearstretching(temp1[:, :, 2], b_minvalue, b_maxvalue)

    temp1 = temp1.astype(np.uint8)
    #temp2 = temp1.astype(np.uint8)  # 利用数组转换，然后再imshow时，注意数据的类型，要转换为uint8

    #print(temp1[:,:,0])
    # plt.figure(figsize=(4, 4))
    # plt.subplot(1, 2, 1)
    plt.imshow(temp1,cmap='gray')
    # plt.show()
    # plt.subplot(1, 2, 2)
    # plt.imshow(temp)

    plt.show()
#def strechimg(raster, r_band, g_band, b_band):
#  raster是传过来的图像文件用gdal 打开后的返回值，raster=gdal.open()，后三个参数分别为rgb波段对应的波段数字
#输入参数r,g,b应当是二维数组[height,width]
def strechimg(xsize,ysize,r, g, b):

    print('The image will be shown shortly')
    temp = np.empty((ysize, xsize, 3)).astype(np.float32)

    temp[:, :, 0] = r
    temp[:, :, 1] = g
    temp[:, :, 2] = b
    temp1 = temp.astype(np.float32)
    r_minvalue, r_maxvalue = minmaximg(ysize, xsize, temp1[:, :, 0])  # 三个波段分别计算2%和98%处的灰度值，并设置最大最小值
    g_minvalue, g_maxvalue = minmaximg(ysize, xsize, temp1[:, :, 1])
    b_minvalue, b_maxvalue = minmaximg(ysize, xsize, temp1[:, :, 2])
    temp1[:, :, 0] = linearstretching(temp1[:, :, 0], r_minvalue, r_maxvalue)  # 三个波段分别将最大最小值之间的灰度值拉伸到0-255
    print('band1 end')
    temp1[:, :, 1] = linearstretching(temp1[:, :, 1], g_minvalue, g_maxvalue)
    print('band2 end')
    temp1[:, :, 2] = linearstretching(temp1[:, :, 2], b_minvalue, b_maxvalue)
    print('band3 end')

    # temp1[0:46,0:92,0] = 255
    # temp1[46:92,0:92,1] =255

    temp1 = temp1.astype(np.uint8)
    #temp2 = temp1.astype(np.uint8)  # 利用数组转换，然后再imshow时，注意数据的类型，要转换为uint8

    #print(temp1[:,:,0])
    # plt.figure(figsize=(4, 4))
    # plt.subplot(1, 2, 1)
    plt.imshow(temp1[:,:,:])
    # plt.show()
    # plt.subplot(1, 2, 2)
    # plt.imshow(temp)

    plt.show()


if __name__ == "__main__":
    raster = gdal.Open('C:\Program Files\Exelis\ENVI53\classic\data\can_tmr.img')
    #raster = gdal.Open('C:\\test.jpg')
    #raster = gdal.Open('C:\sub-TM-Spot-GS.img')
    raster_array = raster.ReadAsArray()
    r = raster_array[0]
    g = raster_array[1]
    b = raster_array[2]

    #print(r)
    xsize = raster.RasterXSize  # RasterXSize是图像的行数，也就是图的高，即Y
    ysize = raster.RasterYSize  # RasterYSize是图像的列数，图像的宽度

    '''
    r_test = raster.GetRasterBand(4)  # 用GDAL的GetRasterBand时注意，读取的第一个波段编号是1，而不是0
    g_test = raster.GetRasterBand(3)
    b_test = raster.GetRasterBand(2)

    r = r_test.ReadAsArray(0, 0, xsize, ysize)
    g = g_test.ReadAsArray(0, 0, xsize, ysize)
    b = b_test.ReadAsArray(0, 0, xsize, ysize)
    '''
    temp = np.empty((ysize, xsize, 3),dtype=float)

    temp[:, :, 0] = r
    temp[:, :, 1] = g
    temp[:, :, 2] = b

    temp1 = temp.astype(np.float32)
    r_minvalue, r_maxvalue = minmaximg(ysize, xsize, temp1[:, :, 0])  # 三个波段分别计算2%和98%处的灰度值，并设置最大最小值
    g_minvalue, g_maxvalue = minmaximg(ysize, xsize, temp1[:, :, 1])
    b_minvalue, b_maxvalue = minmaximg(ysize, xsize, temp1[:, :, 2])
    temp1[:, :, 0] = linearstretching(temp1[:, :, 0], r_minvalue, r_maxvalue)  # 三个波段分别将最大最小值之间的灰度值拉伸到0-255
    temp1[:, :, 1] = linearstretching(temp1[:, :, 1], g_minvalue, g_maxvalue)
    temp1[:, :, 2] = linearstretching(temp1[:, :, 2], b_minvalue, b_maxvalue)

   #temp2 = temp1.astype(np.uint8)  # 利用数组转换，然后再imshow时，注意数据的类型，要转换为uint8
    # temp1[0:100,:,0] =255
    # temp1[0:100,:,1] =0
    # temp1[0:100,:,2] =0
    temp1 = temp1.astype(np.uint8)
    #print(temp2[:,:,0])
    # plt.figure(figsize=(4, 4))
    # plt.subplot(1, 2, 1)
    plt.imshow(temp1)
    # plt.show()
    # plt.subplot(1, 2, 2)
    # plt.imshow(temp)

    plt.show()