from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np


# 计算2% 98%处值作为最小 最大值
def minmaximg(width, heigh, img):
    img_fla = img.flatten()  # 展成一维数组
    # img_fla = np.sort(img_fla) #排序
    len = img.max()
    print('max=%d' % len)
    num = np.zeros(len + 1, dtype=np.int32)
    print('size=%d' % np.size(img_fla))
    for i in np.arange(np.size(img_fla)):
        num[img_fla[i]] = num[img_fla[i]] + 1
        # print('img_fla= %d,%d'%(i,img_fla[i]))
        # print('num %d,%d' % (i, num[img_fla[i]]))
    p = 0
    min = 0
    max = 0
    print(np.sum(num))
    for j in np.arange(len + 1):
        p = num[j] / (width * heigh) + p
        # print('/n',num[j])
        if p >= 0.02:
            min = j
            break
    p = 0

    for j in np.arange(len + 1):
        p = num[j] / (width * heigh) + p
        # print('/n,%', num[j])
        # print('/n,%', p)
        # print('/n,%', j)
        if p >= 0.98:
            max = j
            break
    return min, max


def linearstretching(img, min, max):
    img = np.where(img > min, img, min)
    img = np.where(img < max, img, max)
    img = (img - min) / (max - min) * 255
    return img
if __name__ == "__main__":
    #raster = gdal.Open('C:\Program Files\Exelis\ENVI53\classic\data\can_tmr.img')
    raster = gdal.Open('C:\\test.jpg')
    # raster = gdal.Open('C:\\2008-10-16_13-17-33_DATA.BSQ')
    raster_array = raster.ReadAsArray()
    r = raster_array[0]
    g = raster_array[1]
    b = raster_array[2]

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
    temp = np.empty((ysize, xsize, 3))

    temp[:, :, 0] = r
    temp[:, :, 1] = g
    temp[:, :, 2] = b

    temp2 = temp.astype(np.uint8)  # 利用数组转换，然后再imshow时，注意数据的类型，要转换为uint8
    temp1 = temp.astype(np.uint8)

    r_minvalue, r_maxvalue = minmaximg(ysize, xsize, temp2[:, :, 0]) #三个波段分别计算2%和98%处的灰度值，并设置最大最小值
    g_minvalue, g_maxvalue = minmaximg(ysize, xsize, temp2[:, :, 1])
    b_minvalue, b_maxvalue = minmaximg(ysize, xsize, temp2[:, :, 2])
    temp2[:, :, 0] = linearstretching(temp2[:, :, 0], r_minvalue, r_maxvalue) #三个波段分别将最大最小值之间的灰度值拉伸到0-255
    temp2[:, :, 1] = linearstretching(temp2[:, :, 1], g_minvalue, g_maxvalue)
    temp2[:, :, 2] = linearstretching(temp2[:, :, 2], b_minvalue, b_maxvalue)

    plt.figure( figsize = (4, 4) )
    plt.subplot(1,2,1)
    plt.imshow(temp2)
    #plt.show()
    plt.subplot(1,2,2)
    plt.imshow(temp1)

    plt.show()