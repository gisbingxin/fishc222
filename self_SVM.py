# from sklearn.datasets import load_iris
# import pandas as pd
# from osgeo import gdal
# from self_read_position_excel import read_sample_position
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
import self_imgWR

# def from_img_train(file_path,excel_file,class_num,num_per_class,row_start,col_start,row_end,col_end):
#     raster = gdal.Open(file_path)
#
#     band_num = raster.RasterCount
#     height = raster.RasterYSize
#     width = raster.RasterXSize
#
#     # get sample data()
#     #     read_sample_position_categoryID #返回数组形状为（行，列，类别号）
#     #     read sample data using gdal
#     #     discretise data
#     #     write to csv file
#
#     # excel_file = 'G:\data for manuscripts\AVIRIS20100517\\all_new\ROIs\ROIs.xlsx'
#     # class_num = 7
#     # #num_per_class = np.array([196, 201, 195, 276, 196, 199,179])
#     # num_per_class = np.array([6, 201, 195, 1,1, 1, 1])
#     # row_start = 1
#     # col_start = 1
#     # row_end = row_start + num_per_class - 1
#     # col_end = 2
#     sample_num = np.sum(num_per_class)
#
#     print(np.shape(row_end))
#
#     # sample_pos=read_sample_position_categoryID(excel_file,class_num,row_start,col_start,row_end,col_end)
#
#     sample_pos = read_sample_position(excel_file, class_num, row_start, col_start, row_end, col_end)
#
#     x_data = np.zeros([sample_num, band_num], dtype=float)
#     # x_data = []
#     # 用来存储各类别各采样点的图像灰度值
#     y_data = np.zeros(sample_num, dtype=int)  # 代表每个sample的类别编号
#
#     loc_origin = 0
#     for i in range(0, class_num):  # i表示类的序号
#         for x_i in range(loc_origin, loc_origin + num_per_class[i]):  # x_i 表示在该类内的序号，即行号
#             # for y_i in range(0,2):
#             x_offset = int(sample_pos[x_i, 0]) - 1  # GDAL中的ReadAsArray传入参数应当为int型
#             # 而此处sample是numpy.int32，所以需要进行数据类型转换
#             y_offset = int(sample_pos[x_i, 1]) - 1
#             # print('x,y location:', x_locate,y_locate)
#             data_from_raster = raster.ReadAsArray(x_offset, y_offset, 1, 1)  # sample_size_X, sample_size_Y)
#             print('shape of data_from_raster', x_i,np.shape(data_from_raster))
#             # data = np.swapaxes(data_from_raster, 0, 1)
#             data = np.squeeze(data_from_raster)
#             x_data[x_i, :] = data
#             # y_data[num_per_class[i]*i+x_i,i] = 1             #与x_data对应的batch处，赋值为1，其余位置为0
#             y_data[x_i] = i
#         loc_origin = loc_origin + num_per_class[i]
#
#     return x_data,y_data,sample_num,band_num
#
# def from_img_app(data_path):
#     print('readIMG starts')
#     raster =gdal.Open(data_path)#('G:\data for manuscripts\AVIRIS20100517\\f100517t01p00r10\\'
#      #                 'f100517t01p00r10rdn_b\\f100517t01p00r10rdn_b_sc01_ort_img_resized.img')
#
#     im_width = raster.RasterXSize
#     im_height = raster.RasterYSize
#     im_bands = raster.RasterCount
#     im_GeoTrans = raster.GetGeoTransform() #仿射矩阵，左上角像素的大地坐标和像素分辨率
#                         # (左上角x, x分辨率，仿射变换，左上角y坐标，y分辨率，仿射变换)
#     im_proj = raster.GetProjection() #地图投影信息，字符串表示
#     # im_data =raster.ReadAsArray(0,0,im_width,im_height)#raster_array.shape=(191,128,155)对应（band_num,Ysize,Xsize),
#     #                                     # Ysize代表行数，Xsize代表列数
#     app_xs = np.zeros([im_width*im_height, im_bands,1], dtype=float)
#     #
#     # for row_i in range(im_height):
#     #     for col_i in range(im_width):
#     #         im_data = raster.ReadAsArray(col_i, row_i, 1,1)  # raster_array.shape=(191,128,155)对应（band_num,Ysize,Xsize),
#     #         # Ysize代表行数，Xsize代表列数
#     #         data = np.squeeze(im_data)
#     #         x_data[col_i*row_i+col_i, :] = data
#     # del raster
#
#     x_num = 0  # int((x_end - x_start)/sample_step_col)
#     y_num = 0  # int((y_end - y_start)/sample_step_row)
#     for y_i in range(0, im_height):
#         x_num = 0
#         for x_i in range(0, im_width):
#             data_from_raster = raster.ReadAsArray(x_i, y_i, 1, 1)
#             data = np.swapaxes(data_from_raster, 0, 1)
#             index = y_i * im_width + x_i
#             # print('y_i,x_i,index:',y_i,x_i,index)
#             app_xs[index, :,:] = data
#         #     x_num += 1
#         # y_num += 1
#     app_xs=np.squeeze(app_xs)
#     print('shape of appxs',np.shape(app_xs))
#
#     #print('readIMG ends',im_data.dtype.name)
#     return app_xs,im_width,im_height,im_bands,im_GeoTrans,im_proj
#
#
#
# def writeTiff(im_data,im_width,im_height,im_bands,save_path,im_geotrans='NONE',im_proj='NONE'):
#
#     #print(im_data.dtype)
#     print('writeTiff starts')
#     if 'int8' in im_data.dtype.name:
#         datatype = gdal.GDT_Byte
#     elif 'int16' in im_data.dtype.name:
#         datatype = gdal.GDT_UInt16
#     else:
#         datatype = gdal.GDT_Float32
#
#     print('max in writTIF',np.max(im_data))
#     # im_data.reshape(im_bands,im_width,im_height)
#     # print('im_data0',im_data[0,:,:])
#         #创建文件
#     dataset = gdal.GetDriverByName('GTiff').Create(save_path,im_width, im_height, im_bands ,datatype)
#     if dataset!=None:
#         if im_geotrans !='NONE':
#             dataset.SetGeoTransform(im_geotrans)
#         else:
#             pass
#         if im_proj !='NONE':
#             dataset.SetProjection(im_proj)
#         else:
#             pass
#     if im_bands > 1:
#         for i in range(im_bands):
#             dataset.GetRasterBand(i+1).WriteArray(im_data[i,:,:])
#             print(i)
#     else:
#         dataset.GetRasterBand(1).WriteArray(im_data)
#     del dataset
#     print('writeTiff ends')
#
# def writeEnvi(im_data,im_width,im_height,im_bands,save_path,im_geotrans='NONE',im_proj='NONE'):
#
#     #print(im_data.dtype)
#     print('writeEnvi starts')
#     if 'int8' in im_data.dtype.name:
#         datatype = gdal.GDT_Byte
#     elif 'int16' in im_data.dtype.name:
#         datatype = gdal.GDT_UInt16
#     else:
#         datatype = gdal.GDT_Float32
#
#     print('max in writENVI',np.max(im_data))
#     # im_data.reshape(im_bands,im_width,im_height)
#     # print('im_data0',im_data[0,:,:])
#         #创建文件
#     dataset = gdal.GetDriverByName('ENVI').Create(save_path,im_width, im_height, im_bands ,datatype, options=['INTERLEAVE=BIP'])
#     if dataset!=None:
#         if im_geotrans !='NONE':
#             dataset.SetGeoTransform(im_geotrans)
#         else:
#             pass
#         if im_proj !='NONE':
#             dataset.SetProjection(im_proj)
#         else:
#             pass
#     if im_bands > 1:
#         for i in range(im_bands):  #无论interval是BSQ还是BIP,都利用此进行写入数据
#             dataset.GetRasterBand(i+1).WriteArray(im_data[i,:,:])
#             print(i)
#         # for row_id in range(im_height):
#         #     for col_id in range(im_width):
#         #         for band_id in range(im_bands):
#         #             dataset.GetRasterBand(band_id+1).WriteArray(im_data[band_id,col_id,row_id])
#
#     else:
#         dataset.GetRasterBand(1).WriteArray(im_data)
#     del dataset
#     print('writeEnvi ends')

def svm_classifier(features, target):
    print("SVM classifier begins")
    # tuned_parameters = [{'gamma': [1,0.1,0.05,0.01], 'C': [1, 10, 50, 100]}]
    # clf= GridSearchCV(SVC(decision_function_shape='ovo'),tuned_parameters,cv=5)
    clf=SVC(decision_function_shape='ovo',C=700,gamma=0.000001)
    clf.fit(features,target)

    return clf

if __name__ == '__main__':

    is_train=False
    is_app=True
    pik_path = 'G:\data for manuscripts\AVIRIS20100517\\SVM\svm_clf2.pickle'

    if is_train:
        file_path = 'G:\data for manuscripts\AVIRIS20100517\\f100517t01p00r10 For CNN\\' \
                    'f100517t01p00r10rdn_b_sc01_ort_img_resized2radiance_resized2_flaashed_deletebands_eq0.img'
        excel_file = 'G:\data for manuscripts\AVIRIS20100517\\fig_thickness_resized_roated_class_resize_ROIs\ROIs.xlsx'
        class_num = 7
        num_per_class = np.array([196, 201, 195, 276, 196, 199,179])
        #num_per_class = np.array([150, 150, 150, 150, 150, 150, 150])
        # file_path = 'F:\Python\workshop\data\hydata\PaviaU.tif'
        # excel_file = 'F:\Python\workshop\data\hydata\PaviaU.xlsx'
        # class_num = 9
        # num_per_class = np.array([6000,9990,1960, 3001, 1295, 5020, 1300, 3199,939])
        row_start = 1
        col_start = 1
        row_end = row_start + num_per_class - 1
        col_end = 2
        x_data, y_data, sample_num, band_num = self_imgWR.from_img_train(file_path, excel_file, class_num, num_per_class, row_start,
                                                        col_start, row_end, col_end)
        #y_data = np.reshape(y_data, [sample_num, 1])  # 把行向量转换为列向量，以便在后期与x_data合并
        data_train, data_test, label_train, label_test = train_test_split(x_data, y_data, test_size=0.5)

        trained_model=svm_classifier(data_train,label_train)

        #trained_model = svm_classifier(x_data, y_data)
        print(accuracy_score(y_data,trained_model.predict(x_data)))
        # print('y_data',np.shape(y_data),y_data)
        # print('predi',np.shape(trained_model.predict(x_data)),trained_model.predict(x_data))


        with open(pik_path,'wb') as f:
            pickle.dump(trained_model,f)
        # is_app=False
    else:

        file_path = 'G:\data for manuscripts\AVIRIS20100517\\f100517t01p00r11 For test\\f11_radiance_flaashed_resized_deletebands_eq0.img'
        #file_path = 'G:\data for manuscripts\AVIRIS20100517\\SVM\\test.img'
        # file_path = 'G:\data for manuscripts\AVIRIS20100517\\f100517t01p00r10 For CNN\\' \
        #             'f100517t01p00r10rdn_b_sc01_ort_img_resized2radiance_resized2_flaashed_deletebands_eq0.img'
        #file_path = 'F:\Python\workshop\data\hydata\PaviaU.tif'
        im_data, im_width, im_height, im_bands, im_GeoTrans, im_proj=self_imgWR.from_img_app(file_path)

        with open(pik_path,'rb') as f:
            trained_model=pickle.load(f)

            predicted_label=[]
            total_size=np.shape(im_data)[0]

            if total_size > 2000:
                start = 0
                n = int(total_size / 2000)
                for batch_i in range(0, n):
                    predicted_tmp = trained_model.predict(im_data[start:start+2000])
                    predicted_label.extend(predicted_tmp)
                    start += 2000
                    print(start, 'shape of label', np.shape(predicted_label))
                if (n*2000) < total_size:
                    predicted_tmp = trained_model.predict(im_data[n*2000:total_size])
                    predicted_label.extend(predicted_tmp)
                    print('shape of predicted_label', np.shape(predicted_label))
            else:
                predicted_label=trained_model.predict(im_data)


            label = np.reshape(predicted_label, (im_height, im_width))
            label = label + 1
            out_tif = 'G:\data for manuscripts\AVIRIS20100517\\SVM\\f11_test1.tif'
            #out_tif='F:\Python\workshop\data\hydata\PaviaU_svm.tif'
            self_imgWR.writeTiff(label, im_width, im_height, 1, out_tif, im_GeoTrans, im_proj)
            plt.imshow(label)
            plt.show()

