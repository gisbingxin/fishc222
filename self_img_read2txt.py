from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from self_strech_img import minmaximg, linearstretching,strechimg,strechimg_pan
from self_read_position_excel import read_sample_position


def read_from_img(image_name,show_img = False):
    # 用于将遥感影像读入到数组，整体影像。
    # 输入值为影像完整路径，输出为raster,raster_array,xsize, ysize,band_num
    raster = gdal.Open(image_name)
    #raster = gdal.Open('e:\qwer.img')
    print('Read from img function,Reading the image named:',image_name)
    xsize = raster.RasterXSize  # RasterXSize是图像的行数，也就是图的高，即Y
    ysize = raster.RasterYSize  # RasterYSize是图像的列数，图像的宽度
    band_num = raster.RasterCount #RasterCount是图像的波段数

    # 获取rgb对应波段数值，传递给拉伸显示模块
    raster_array = raster.ReadAsArray() #raster_array.shape=(191,128,155)对应（band_num,Ysize,Xsize),
                                        # Ysize代表行数，Xsize代表列数
    #raster_array_np = np.array(raster_array)
    #print('raster_array shape in read_show img',raster_array_np.shape)
#
    if band_num >= 3:
        # r = raster.GetRasterBand(29)#raster_array[0]
        # g = raster.GetRasterBand(20)#raster_array[1]
        # b = raster.GetRasterBand(12)#raster_array[2]

        # r = r.ReadAsArray()
        # g = g.ReadAsArray()
        # b = b.ReadAsArray()
        r = raster_array[0]
        g = raster_array[1]
        b = raster_array[2]
        print('type of r in read_show_img:', type(r))

        if show_img:    #判断是否需要显示图像
            strechimg(xsize,ysize,r,g,b) #调用图像拉伸显示模块
            #strechimg(92, 92, r[0:92,0:92], g[0:92,0:92], b[0:92,0:92])  # 调用图像拉伸显示模块
    else:
        img_data = raster.GetRasterBand(1)
        img_data = img_data.ReadAsArray()
        print(np.shape(img_data))
        if show_img:
            strechimg_pan(xsize,ysize,img_data)
            #print()
    #return raster,raster_array,xsize, ysize,band_num

def sample_from_img():
    #获取采样点的数据（训练数据或测试数据），
    # 输入图像读取后的raster和波段数，输出x_data,y_data。分别是采样点位置及其数据、该点的类别数组
    pass

def sample_img2txt():
    # 用于将已读入的采样点数据（训练数据或测试数据）存储为txt格式，方便多次使用。
    # 输入为已经存为数组的采样点数据，输出为txt文件
    pass

def sample_from_txt():
    # 用于从txt文件中读取采样点数据（训练数据或测试数据）。输入为txt文件的完整路径名，输出为符合cnn需要的数据
    pass


if __name__ == '__main__':
    #本模块的目的是将采样点转化为txt。其单独运行时，只需要执行read_from_img，sample_from_img和sample_img2txt.
    #cnn模块调用时，只需要执行sample_from_txt。从txt中读取采样数据即可。（训练和测试）
    #如果用训练好的网络进行分类，则cnn模块需要读取完整影像。（可将sample_size设置为影像大小，坐标点为（1，1））
    img_name = 'F:\Python\workshop\\fishc\\aviris_oil\\aviris_subsize.img'
    read_from_img(img_name,False)
    sample_from_img()
    sample_img2txt()