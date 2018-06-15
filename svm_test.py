# Multi-class (Nonlinear) SVM Example
# ----------------------------------
#
# This function wll illustrate how to
# implement the gaussian kernel with
# multiple classes on the iris dataset.
#
# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
#
# X : (Sepal Length, Petal Width)
# Y: (I. setosa, I. virginica, I. versicolor) (3 classes)
#
# Basic idea: introduce an extra dimension to do
# one vs all classification.
#
# The prediction of a point will be the category with
# the largest margin or distance to boundary.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
from self_strech_img import minmaximg, linearstretching, strechimg, strechimg_pan
from self_read_position_excel import read_sample_position, read_sample_position_random


'''
def get_1D_sample_data(raster, band_num, class_num, num_per_class, total_per_class, sample_num,
                       excel_name, sheet_num, start_row, start_col, end_row, end_col,
                       sample_size_X=1, sample_size_Y=1, channel_1D=1, random=False):
    # 读取Excel中存储的采样点位置数据，返回值中sample_num代表类别编号
    if random:
        sample_pos = read_sample_position_random(excel_name, sheet_num, num_per_class, total_per_class, start_col,
                                                 start_row)
    else:
        sample_pos = read_sample_position(excel_name, sheet_num, start_row, start_col, end_row, end_col)

        # 该函数返回所选择的sample的位置点，是一个三维数组shape为[class_num,X_num,Y_num]

    # #sample定义为[batch，band,Ysize,Xsize],batch这里是所有类别采样点的数量
    # x_data = np.zeros([sample_num, band_num, sample_size_Y, sample_size_X], dtype=float)
    x_data = np.zeros([sample_num, band_num, channel_1D], dtype=float)
    # x_data = []
    # 用来存储各类别各采样点的图像灰度值
    y_data = np.zeros([sample_num, class_num], dtype=int)

    margin_pixel = []
    loc_origin = 0
    for i in range(0, class_num):  # i表示类的序号
        for x_i in range(loc_origin, loc_origin + num_per_class[i]):  # x_i 表示在该类内的序号，即行号
            # for y_i in range(0,2):
            x_offset = int(sample_pos[x_i, 0]) - 1  # GDAL中的ReadAsArray传入参数应当为int型
            # 而此处sample是numpy.int32，所以需要进行数据类型转换
            y_offset = int(sample_pos[x_i, 1]) - 1
            # print('x,y location:', x_locate,y_locate)
            data_from_raster = raster.ReadAsArray(x_offset, y_offset, sample_size_X, sample_size_Y)
            # print('shape of data_from_raster',np.shape(data_from_raster))
            data = np.swapaxes(data_from_raster, 0, 1)
            x_data[x_i, :, :] = data
            # y_data[num_per_class[i]*i+x_i,i] = 1             #与x_data对应的batch处，赋值为1，其余位置为0
            y_data[x_i, i] = 1
        loc_origin = loc_origin + num_per_class[i]
        # print('shape of x_data in get next batch',np.shape(x_data),np.shape(y_data))
    return x_data, y_data



if __name__ == '__main__':


    image_name = 'G:\data for manuscripts\\aviris_oil\org\\aviris_subsize.img'
    excel_name = 'G:\data for manuscripts\\aviris_oil\oil samples.xlsx'
    # train_excel_name = 'F:\Python\workshop\data\hydata\mannual_samp\Pavia_sample_manual.xlsx'

    if train or test:
        # 训练样本位置和测试样本位置存在同一个Excel中，前num_per_class是training samples
        #  从第num_per_class + 1之后的数据是test samples
        # 通常样本选择不是随机的，而是人工选择
        num_per_class = np.array([400, 200, 400, 400, 400, 400, 400])  # 训练数据中，每一类的采样点个数
        # num_per_class = np.array([6431, 18449, 1899, 2864, 1145, 4829, 1130, 3482, 747])
        total_per_class = np.array([3691, 427, 3905, 3942, 4035, 3788, 3504])

        sample_num = np.sum(num_per_class)  # class_num * num_per_class  #训练数据中，所有类采样点的总数。对应后面的batch

        start_row = 1  # 表示记录采样点数据的Excel中，数据开始的行，0表示第一行
        end_row = start_row + num_per_class - 1

        start_col = 1  # 表示记录采样点数据的Excel中，数据开始的列，0表示第一列
        end_col = 2  # 如果行列数字错误，可能出现如下错误：
        # ERROR 5: Access window out of range in RasterIO().  Requested
        # (630,100) of size 10x10 on raster of 634x478.
        sheet_num = class_num  # 表示Excel中sheet的数目，必须与类别数量一致

        show_img = False  # 用于判断是否对图像进行显示
        raster, raster_array, xsize, ysize, band_num = read_show_img(image_name, show_img)  # 读取遥感影像

        # sample_size_X = 46  #训练数据的宽
        # sample_size_Y = 80  #训练数据的高
        # class_num = 3       #训练数据的类数
        xs = tf.placeholder(tf.float32, [None, band_num, channel_1D])  #
        if train:
            # x_data, y_data = get_1D_sample_data(raster, band_num, class_num, num_per_class, total_per_class, sample_num,
            #                                     excel_name, sheet_num, start_row, start_col, end_row, end_col,
            #                                     sample_size_X, sample_size_Y, channel_1D, random_sample)
            x_data, y_data = get_1D_sample_data(raster, band_num, class_num, num_per_class, total_per_class, sample_num,
                                                excel_name, sheet_num, start_row, start_col, end_row, end_col,
                                                sample_size_X, sample_size_Y, channel_1D, random_sample)


'''
