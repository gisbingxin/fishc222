from scipy import io
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from osgeo import osr
import self_strech_img as stretch


def readMat(mat_path,key_w,only_get_key):
    mat = io.loadmat(mat_path) #返回值是一个字典
    if only_get_key ==True:
        print(mat.items())
        return 0,0,0,0
    im_data = mat.get(key_w)

    im_data = np.array(im_data, np.float32)
    # print(np.max(im_data))
    # print(im_data.shape)
    if len(im_data.shape) ==3:
        height,width,band_num = im_data.shape   # 注意，band_num, height,width的位置，要根据具体数据存储顺序进行改变。
                                                # 如果顺序错误，则在下面显示部分出现异常显示，不报错
    else:
        band_num,(height,width) = 1,im_data.shape

    return im_data,band_num,height,width
#保存tif文件函数
def writeTiff(im_data,im_width,im_height,im_bands,save_path,im_geotrans='NONE',im_proj='NONE'):

    #print(im_data.dtype)

    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    print('max in writTIF',np.max(im_data))
    # im_data.reshape(im_bands,im_width,im_height)
    # print('im_data0',im_data[0,:,:])
        #创建文件
    dataset = gdal.GetDriverByName('GTiff').Create(save_path,im_width, im_height, im_bands ,datatype)
    if dataset!=None:
        if im_geotrans !='NONE':
            dataset.SetGeoTransform(im_geotrans)
        else:
            pass
        if im_proj !='NONE':
            dataset.SetProjection(im_proj)
        else:
            pass
    if im_bands > 1:
        for i in range(im_bands):
             dataset.GetRasterBand(i+1).WriteArray(im_data[i,:,:]) #注意im_data的数据形状，此时形状为[band,height,width]
    else:
        dataset.GetRasterBand(1).WriteArray(im_data)
    del dataset

def read_show_tiff(tiff_path):
    raster = gdal.Open(tiff_path)
    xsize = raster.RasterXSize
    ysize = raster.RasterYSize
    band_num = raster.RasterCount
    raster_array = raster.ReadAsArray()

    if band_num >=3:
        # plt.imshow(raster_array[0,:,:])
        # plt.show()
        r = raster_array[40]
        g = raster_array[20]
        b = raster_array[10]
        stretch.strechimg(xsize,ysize,r,g,b)
    elif band_num==1:
        # plt.imshow(raster_array)
        # plt.show()
        im=raster_array
        print('max in show tiff', np.max(im))
        stretch.strechimg_pan(xsize,ysize,im,False)

if __name__ == '__main__':
    mat_path = '.\hydata\Botswana_gt.mat'
    key_w = 'Botswana_gt'
    only_get_key =False #通过这个“开关”控制是否仅仅查看mat文件中的key值。
    im_data,band_num,height,width=readMat(mat_path,key_w,only_get_key)  #读取Mat文件并返回图像参数，
                                # 注意，band_num, height,width的位置，要根据具体数据存储顺序进行改变。
                                # 如果顺序错误，则在下面显示部分出现异常显示，不报错
    if only_get_key == False:
        save_path = './hydata/Botswana_gt.tif'
        geotransform = 'NONE'   #如果有地理变换和投影信息，则此处改为真实的信息，
                                # 否则以NONE为flag，在数据保存时不进行地理变换和投影的数据写入
        pro = 'NONE'
        writeTiff(im_data,width,height,band_num,geotransform,pro,save_path)

        tiff_path = save_path#'./hydata/salinas_gt.tif'

        read_show_tiff(tiff_path)

