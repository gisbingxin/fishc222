# import pandas as pd
# import pymrmr
# df=pd.read_csv('G:\data for manuscripts\mrmr\\tese_lung_s3.csv')
# pymrmr.mRMR(df,'MIQ',10)
# * First parameter is a pandas DataFrame containing the input dataset, discretised as defined in the original paper
# (for ref. see http://home.penglab.com/proj/mRMR/). The rows of the dataset are the different samples.
# The first column is the classification (target) variable for each sample.
# The remaining columns are the different variables (features) which may be selected by the algorithm.
# (see "Sample Data Sets" at http://home.penglab.com/proj/mRMR/ to download sample dataset to test this algorithm);
# * Second parameter is a string which defines the internal Feature Selection method to use (defined in the original paper):
#  possible values are "MIQ" or "MID";
# * Third parameter is an integer which defines the number of features that should be selected by the algorithm.

import numpy as np
from osgeo import gdal
from mat2tiff import writeTiff
from self_read_position_excel import read_sample_position
import pandas as pd
import xlrd


def discretise(x_data, y_data, sample_num, band_num, output_file):

    # print(band_num)
    # data0=raster.ReadAsArray(0,0,width,height) #（band_num,Ysize,Xsize)

    data = np.array(x_data, dtype=np.int32)  # 图像的数据是无符号类型，无法赋值为负数，因此现将其改为int型，再赋值负数即可

    for i in range(band_num):
        m = np.mean(data[:, i])
        s = np.std(data[:, i])
        mask1 = data[:, i] > (m + 2*s)
        mask2 = data[:, i] < (m - 2*s)
        mask3 = ((data[:, i] <= (m + 2*s)) & (data[:, i] >= (m - 2*s)))
        # mask3 = ((data[i,:,:]-(m+s)).all())# and ((m+s)-data[i,:,:] ).any())


        data[mask1, i] = 1
        data[mask2, i] = -1
        data[mask3, i] = 0
    data_with_category = np.c_[y_data, data]  # 把两个矩阵进行合并
    print(data_with_category)

    name = []
    for name_id in range(0, band_num + 1):
        txt = 'b' + str(name_id)
        name.append(txt)
    # print(name)
    name[0] = 'class'
    print(name)

    csv_content = pd.DataFrame(columns=name, data=data_with_category)
    csv_content.to_csv(output_file, sep=',', index=False)
    # writeTiff(data,width,height,band_num,'e:\\test.tif')
def from_img(file_path,excel_file,class_num,num_per_class,row_start,col_start,row_end,col_end):
    raster = gdal.Open(file_path)

    band_num = raster.RasterCount
    height = raster.RasterYSize
    width = raster.RasterXSize

    # get sample data()
    #     read_sample_position_categoryID #返回数组形状为（行，列，类别号）
    #     read sample data using gdal
    #     discretise data
    #     write to csv file

    # excel_file = 'G:\data for manuscripts\AVIRIS20100517\\all_new\ROIs\ROIs.xlsx'
    # class_num = 7
    # #num_per_class = np.array([196, 201, 195, 276, 196, 199,179])
    # num_per_class = np.array([6, 201, 195, 1,1, 1, 1])
    # row_start = 1
    # col_start = 1
    # row_end = row_start + num_per_class - 1
    # col_end = 2
    sample_num = np.sum(num_per_class)

    print(np.shape(row_end))

    # sample_pos=read_sample_position_categoryID(excel_file,class_num,row_start,col_start,row_end,col_end)

    sample_pos = read_sample_position(excel_file, class_num, row_start, col_start, row_end, col_end)

    x_data = np.zeros([sample_num, band_num], dtype=float)
    # x_data = []
    # 用来存储各类别各采样点的图像灰度值
    y_data = np.zeros(sample_num, dtype=int)  # 代表每个sample的类别编号

    loc_origin = 0
    for i in range(0, class_num):  # i表示类的序号
        for x_i in range(loc_origin, loc_origin + num_per_class[i]):  # x_i 表示在该类内的序号，即行号
            # for y_i in range(0,2):
            x_offset = int(sample_pos[x_i, 0]) - 1  # GDAL中的ReadAsArray传入参数应当为int型
            # 而此处sample是numpy.int32，所以需要进行数据类型转换
            y_offset = int(sample_pos[x_i, 1]) - 1
            # print('x,y location:', x_locate,y_locate)
            data_from_raster = raster.ReadAsArray(x_offset, y_offset, 1, 1)  # sample_size_X, sample_size_Y)
            print('shape of data_from_raster', np.shape(data_from_raster))
            # data = np.swapaxes(data_from_raster, 0, 1)
            data = np.squeeze(data_from_raster)
            x_data[x_i, :] = data
            # y_data[num_per_class[i]*i+x_i,i] = 1             #与x_data对应的batch处，赋值为1，其余位置为0
            y_data[x_i] = i
        loc_origin = loc_origin + num_per_class[i]

    return x_data,y_data,sample_num,band_num

def from_excel(excel_name):
    workbook = xlrd.open_workbook(excel_name)
    sheet1 = workbook.sheet_by_index(0) #默认读取第一个sheet的数值
    nrows=sheet1.nrows
    ncols=sheet1.ncols
    data=np.empty((nrows,ncols),dtype=np.float32)

    for row_id in range(nrows):
        data[row_id,:]=sheet1.row_values(row_id)

    x_data=data[:,1:]
    y_data=data[:,0]
    sample_num=nrows
    band_num=ncols-1
    print('from excel x shape, y shape',np.shape(x_data),np.shape(y_data))
    return x_data,y_data,sample_num,band_num

if __name__=='__main__':

    is_from_img=True
    is_from_excel=True
    x_data=[]
    y_data=[]
    sample_num=0
    band_num=0

    if is_from_img:
        file_path='G:\data for manuscripts\\aviris2010070910\\r2\\test'
        excel_file = 'G:\data for manuscripts\AVIRIS20100517\\all_new\ROIs\ROIs.xlsx'
        class_num = 7
        # num_per_class = np.array([196, 201, 195, 276, 196, 199,179])
        num_per_class = np.array([6, 201, 195, 1, 1, 1, 1])
        row_start = 1
        col_start = 1
        row_end = row_start + num_per_class - 1
        col_end = 2
        x_data,y_data,sample_num,band_num=from_img(file_path,excel_file,class_num,num_per_class,row_start,col_start,row_end,col_end)
        y_data = np.reshape(y_data, [sample_num, 1])  # 把行向量转换为列向量，以便在后期与x_data合并
        is_from_excel=False

    elif is_from_excel:
        excel_name='G:\data for manuscripts\AVIRIS20100517\\f100517t01p00r10 For CNN\\xdata_wv2.xlsx'
        x_data,y_data,sample_num,band_num=from_excel(excel_name)


    output_file='G:\data for manuscripts\AVIRIS20100517\\all_new\ROIs\discretise-20180625.csv'

    discretise(x_data,y_data,sample_num,band_num,output_file)