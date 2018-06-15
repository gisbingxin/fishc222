from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

#raster = gdal.Open('C:\\test.jpg')

raster = gdal.Open('C:\\can_tmr.img')
#raster = gdal.Open('C:\\2008-10-16_13-17-33_DATA.BSQ')
raster_array = raster.ReadAsArray()
'''raster_ts = tf.convert_to_tensor(raster_array)
raster_ts_reshape = tf.reshape(raster_ts,[400,640,6])
sess = tf.Session()
print(sess.run(raster_ts_reshape))
'''
r_test=raster.GetRasterBand(4)  #用GDAL的GetRasterBand时注意，读取的第一个波段编号是1，而不是0
g_test=raster.GetRasterBand(3)
b_test=raster.GetRasterBand(2)
print(type(r_test))

xsize=raster.RasterXSize  #RasterXSize是图像的行数，也就是图的高，即Y
ysize=raster.RasterYSize  #RasterYSize是图像的列数，图像的宽度
#print(raster_array.shape)
#print(raster_array.dtype)
'''将原始图像的各个波段进行分离，并分别赋值作为r,g,b波段。原始数据为3维数据，rgb则分别为二维数据。
另外,注意数据存储方式，BIP，BIL，BSQ，imshow的时候支持[x,y,band]。
r=raster_array[3]
g=raster_array[2]
b=raster_array[1]'''


r=r_test.ReadAsArray(0,0,xsize,ysize)
g=g_test.ReadAsArray(0,0,xsize,ysize)
b=b_test.ReadAsArray(0,0,xsize,ysize)


temp=np.empty((ysize,xsize,3))
print(temp.shape)

temp[:,:,0]=r
temp[:,:,1]=g
temp[:,:,2]=b

temp2=temp.astype(np.uint8) #利用数组转换，然后再imshow时，注意数据的类型，要转换为uint8


#plt.imshow(raster_array[1],cmap='Greys_r') #第二个波段，灰度显示
#plt.imshow(raster_array[2]) # 第三个波段，显示的是彩色图
plt.imshow(temp2)
plt.show()

bins = np.arange(-100, 100, 5)  # fixed bin size
r2=r.flatten()  #r.reshape(xsize*ysize) ###直方图统计的时候，应先将图像二维数组变为一维数组
plt.xlim(min(r2), max(r2))

plt.hist(r2, bins=bins, alpha=0.5)
plt.title('Random Gaussian data (fixed bin size)')
plt.xlabel('variable X (bin size = 5)')
plt.ylabel('count')

plt.show()