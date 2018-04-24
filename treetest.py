import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])
plt.ion()

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

while True:
    plt.pause(0.05)

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # plt.ion()
    #
    # for i in range(x_num*y_num):
    #     #strechimg(app_sample_size_width, app_sample_size_height,app_xs[i, :, :, 0], app_xs[i, :, :, 1], app_xs[i, :, :, 2])
    #     try:
    #         ax.images.remove(img[0])
    #     except:
    #         pass
    #     img = plt.imshow(app_xs[i, :, :, 0])
    #
    #     fig.suptitle(str(i))
    #     plt.pause(0.1)
    #     ax.images.clear()
    # plt.ioff()

    # x=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
# plt.imshow(x)
# plt.show()
# x=np.array(x)
# xr = x.reshape(3,4)
# plt.imshow(xr)
# plt.show()
# xrr =x.swapaxes(0,1)
# plt.imshow(xrr)
# plt.show()
#
# # print(x.shape)
# # print(x)
# # print(xr)
#
#
# # raster = gdal.Open('e:\\qwer.img')
# # raster = gdal.Open('e:\qwer.img')
# # print('Reading the image named:',image_name)
# # print('geotrans is ', raster.GetGeoTransform())
# # print('projection is ',raster.GetProjection())
# # xsize = raster.RasterXSize  # RasterXSize是图像的行数，也就是图的高，即Y
# # ysize = raster.RasterYSize  # RasterYSize是图像的列数，图像的宽度
# # band_num = raster.RasterCount #RasterCount是图像的波段数
# # # 获取rgb对应波段数值，传递给拉伸显示模块
# #
# # raster_array = raster.ReadAsArray() #raster_array.shape=(191,128,155)对应（band_num,Ysize,Xsize),
# # print(np.shape(raster_array),xsize,ysize)
# # writeTiff(raster_array,xsize,ysize,band_num,'','','./hydata/t.tif')
# # mat_img = np.empty((610,340,3))#.astype(np.float32))
# # mat_img[:,:,0] = r
# # mat_img[:,:,1] = g
# # mat_img[:,:,2] = b
# # plt.imshow(mat_img)
# # plt.show()
#
# # print(mat.get('paviaU'))
#
#
#
#
#
# # import tensorflow as tf
# import numpy as np
#
# import xlrd
#
# workbook = xlrd.open_workbook('samples.xlsx')
#
# sheet_names= workbook.sheet_names()
# sample = np.empty([3,14,2],dtype=int)
# sample_temp = np.empty([14,2],dtype=int)
# x = np.empty(14,dtype=int)
# y = np.empty(2,dtype=int)
# sheet_num = 0
# for sheet_name in sheet_names:
#     sheet2 = workbook.sheet_by_name(sheet_name)
#     for i in range(1,15):
#         for j in range(3,5):
#             #print(i,j)
#             #sample_temp[i-1,j] = sheet2.cell(i, j).value
#             sample[sheet_num, i-1, j-3] = sheet2.cell(i, j).value
#         #sample[sheet_num,:,:] = sample_temp
#     sheet_num += 1
# print(sample)
#
#     # rows = sheet2.row_values(3) # 获取第四行内容
#     # cols = sheet2.col_values(1) # 获取第二列内容
#
#
# # b=tf.reshape(a,[2,3,4])
# # w_c1 = tf.Variable(tf.random_normal([24]))
# # w_c2 = tf.reshape(w_c1,[2,1,2,6])
# # w_c3 = tf.reshape(w_c1,[1,2,2,6])
# # sess = tf.Session()
# # sess.run(tf.global_variables_initializer())
# #
# # print('wc2',sess.run(w_c2))
# # print('wc3',sess.run(w_c3))
# #a=[0,1,2,3,3,3,3]
# # a=np.arange(1,10)
# # b=np.unique(a)
# # print(a)
# # print(b[2])
# # c =np.sum(a ==b[2])
# # print(c)