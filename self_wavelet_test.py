import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import pywt as wt
from wrcoef import wavedec,wrcoef
import matlab
import matlab.engine

def from_img(file_path):
    raster = gdal.Open(file_path)

    band_num = raster.RasterCount
    height = raster.RasterYSize
    width = raster.RasterXSize

    # sample_pos=read_sample_position_categoryID(excel_file,class_num,row_start,col_start,row_end,col_end)
    sample_num =3
    x_data = np.zeros([band_num], dtype=float)

    data_from_raster = raster.ReadAsArray(0, 0, 1, 1)  # sample_size_X, sample_size_Y)
    print('shape of data_from_raster', np.shape(data_from_raster))
            # data = np.swapaxes(data_from_raster, 0, 1)
    data = np.squeeze(data_from_raster)
    x_data[:]= data
    print('shape of xdata', np.shape(x_data))

    return x_data

if __name__ == '__main__':
    eng = matlab.engine.start_matlab()

    file_path='G:\data for manuscripts\\aviris2010070910\\r2\\test'
    xdata=from_img(file_path)
    #xdata=matlab.double(xdata[:].tolist())
    #a = [1, 2, 1, 2, 1, 2, 1, 5, 1, 2, 1, 2, 1, 4, 1, 3, 1, 4]


    # cC=eng.wavedec(matlab.double(a),2,'db4')
    # print(cC)
    # print(np.shape(cC))
#

    wv=wt.Wavelet('db4')
    wv_str='db4'
# #print(wt.swt_max_level(len(a)))
# #aA2,aD2,aD1\
# cA2,cD2,cD1=wt.wavedec(xdata[200],db1,level=2)
#
    level=3
    c,l=wavedec(xdata,wv,level=level)
    print('c in pywr',c)
    print('l in pywr',l)

    a2=eng.wrcoef('a',matlab.double(c[:].tolist())  #利用matlab对概要系数进行重构
                  ,matlab.double(l[:]),wv_str,level)

    a2=np.squeeze(np.array(a2._data))#从matlab返回来的概要系数类型是mlarray，需要转换为numpy.array格式
    #print(type())
    eng.quit()
    print('a2 shape,a shape',np.shape(a2),np.shape(xdata))

    d=[]
    for n in range(len(l)-2):   #利用python版本的wrcoef进行细节系数重构
        D =wrcoef(c,l,wavelet=wv,level=n+1)
        print(n)
        d.append(D)

    plt.figure()
    plt.subplot(221)
    plt.plot(xdata)
    plt.subplot(222)
    plt.plot(a2)
    plt.subplot(223)
    plt.plot(d[0])
    plt.subplot(224)
    plt.plot(d[1])
    plt.show()

# d=[]
#
# for n in range(len(l)-2):
#     D =wrcoef(c,l,wavelet=db1,level=n+1)
#     print(n)
#     d.append(D)
#
# plt.figure()
# plt.subplot(411)
# plt.plot(xdata[200])
# plt.subplot(412)
# plt.plot(cA2)
# plt.subplot(413)
# plt.plot(d[1])
# plt.subplot(414)
# plt.plot(d[0])
# # p1=plt.subplot(311)
# # p2=plt.subplot(312)
# # p3=plt.subplot(313)
# #
# # p1.plot (xdata[400])
# # p2.plot(d[0])
# # p3.plot(d[1])
#
# plt.show()