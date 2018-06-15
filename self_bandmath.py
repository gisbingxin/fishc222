from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
#from mat2tiff import writeTiff

def writeTiff(im_data,im_width,im_height,im_bands,save_path,im_geotrans='NONE',im_proj='NONE'):

    #print(im_data.dtype)
    print('writeTiff starts')
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
            dataset.GetRasterBand(i+1).WriteArray(im_data[i,:,:])
            print(i)
    else:
        dataset.GetRasterBand(1).WriteArray(im_data)
    del dataset
    print('writeTiff ends')

def readIMG(data_path):
    print('readIMG starts')
    raster =gdal.Open(data_path)#('G:\data for manuscripts\AVIRIS20100517\\f100517t01p00r10\\'
     #                 'f100517t01p00r10rdn_b\\f100517t01p00r10rdn_b_sc01_ort_img_resized.img')

    im_width = raster.RasterXSize
    im_height = raster.RasterYSize
    im_bands = raster.RasterCount
    im_GeoTrans = raster.GetGeoTransform() #仿射矩阵，左上角像素的大地坐标和像素分辨率
                        # (左上角x, x分辨率，仿射变换，左上角y坐标，y分辨率，仿射变换)
    im_proj = raster.GetProjection() #地图投影信息，字符串表示
    im_data =raster.ReadAsArray(0,0,im_width,im_height)#raster_array.shape=(191,128,155)对应（band_num,Ysize,Xsize),
                                        # Ysize代表行数，Xsize代表列数

    del raster
    print('readIMG ends',im_data.dtype.name)
    return im_data,im_width,im_height,im_bands,im_GeoTrans,im_proj

def aviris2radiance(im_data,im_width,im_height,im_bands):
    print('aviris2radiance starts')
    #im_data = np.zeros([im_bands,im_height,im_width])
    im_data = np.asanyarray(im_data,np.float32)
    im_data[0:109,:,:] = im_data[0:109,:,:]/300.0
    print('1-110 end')
    im_data[110:159,:,:] = im_data[110:159,:,:]/600.0
    print('111-160 end')
    im_data[160:223, :, :] = im_data[160:223, :, :] / 1200.0
    print('aviris2radiance ends',np.shape(im_data),im_data.dtype.name)
    return im_data

if __name__ == '__main__':
    data_in_path ='G:\data for manuscripts\AVIRIS20100517\\f100517t01p00r10\\' \
               'f100517t01p00r10rdn_b\\f100517t01p00r10rdn_b_sc01_ort_img_resized.img'
    data_out_path = 'G:\data for manuscripts\AVIRIS20100517\\f100517t01p00r10\\' \
               'f100517t01p00r10rdn_b\\f100517t01p00r10rdn_b_sc01_ort_img_resized2radiance.tif'
    im_data,im_width,im_height,im_bands,im_GeoTrans,im_proj = readIMG(data_in_path)
    im_data = aviris2radiance(im_data,im_width,im_height,im_bands)
    writeTiff(im_data,im_width,im_height,im_bands,data_out_path,im_GeoTrans,im_proj)
