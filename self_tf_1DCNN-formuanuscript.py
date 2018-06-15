from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from self_strech_img import minmaximg, linearstretching, strechimg, strechimg_pan
from self_read_position_excel import read_sample_position, read_sample_position_random
from mat2tiff import writeTiff
import sklearn.metrics as skmetr


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv1d(x, W, stride=1, padding='VALID'):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.conv1d(x, W, stride, padding, data_format='NWC')


def max_pool_nx1(x, pool_size=3, stride=1, padding='valid'):
    # stride [1, x_movement, y_movement, 1]
    # return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return tf.layers.max_pooling1d(x, pool_size, stride, padding)  # padding=same时，输入输出的channel一样，
    # 当padding=valid时，输出=输入-（filter_size-1+strides-1)


# def create_cnn(xs,keep_prob,class_num,sample_size_X,sample_size_Y,band_num):

# 图像读取与现实，输入文件完整路径名，输出图像的长、宽和波段数
def read_show_img(image_name, show_img=False):
    # raster = gdal.Open('C:\\test.jpg')
    # image_name = 'C:\hyperspectral\AVIRISReflectanceSubset.dat'
    raster = gdal.Open(image_name)
    # raster = gdal.Open('e:\qwer.img')
    print('Reading the image named:', image_name)
    # print('geotrans is ', raster.GetGeoTransform())
    # print('projection is ',raster.GetProjection())
    xsize = raster.RasterXSize  # RasterXSize是图像宽
    ysize = raster.RasterYSize  # RasterYSize是图像的高
    band_num = raster.RasterCount  # RasterCount是图像的波段数
    print('in read and show x,y', xsize, ysize)

    # 获取rgb对应波段数值，传递给拉伸显示模块

    raster_array = raster.ReadAsArray()  # raster_array.shape=(191,128,155)对应（band_num,Ysize,Xsize),
    # Ysize代表行数，Xsize代表列数
    raster_array_np = np.array(raster_array)

    #
    # print('rnp',np.shape(rnp))
    print('raster_array shape in read_show img', raster_array_np.shape)
    if band_num >= 3:
        r = raster_array[0]
        g = raster_array[1]
        b = raster_array[2]
        print('type of r in read_show_img:', type(r))

        if show_img:  # 判断是否需要显示图像
            strechimg(xsize, ysize, r, g, b)  # 调用图像拉伸显示模块
            # strechimg(92, 92, r[0:92,0:92], g[0:92,0:92], b[0:92,0:92])  # 调用图像拉伸显示模块
    else:
        img_data = raster.GetRasterBand(1)
        img_data = img_data.ReadAsArray()
        # print(np.shape(img_data))
        if show_img:
            strechimg_pan(xsize, ysize, img_data)
            # print()
    return raster, raster_array, xsize, ysize, band_num


# 获取采样点的数据，输入图像读取后的raster和波段数，输出x_data,y_data。分别是采样点位置及其数据、该点的类别数组
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


# 在所有采样数据中，随机选择一组数据组成batch，其大小也就是一次学习多少个batch。
# 输入为采样位置及数据x_data、类别数据y_data，偏移量off_set和一次学习的数据量。
# 输出是随机产生的batch_size大小的一系列数据及类别（x,y)
# off_set是指每个类别的batch之间相差的数字，等于每个类别中采样点的数，比如0-9为一类，10-19为一类，那么off_set就是10
def get_next_batch(x_data, y_data, batch_size=20):
    x = []
    y = []
    idxs = []

    for _ in range(batch_size):
        idx = np.random.randint(0, sample_num)
        # 依次在每一类的sample中随机选择一个，放入x，y数组中，几种类别共同组成一个batch
        x.append(x_data[idx])
        y.append(y_data[idx])
    x = np.array(x)
    y = np.array(y)
    # print('x shape in get nextbatch',x.shape)
    # x的shape为[batch,height, width,band]
    # y的shape 为[batch,class_num]
    # 如果想对某个batch中的图进行显示，应当先将x的形状转换为[batch,band,height,width].
    # 因为在python或numpy中，数组的形状就是[batch,band,height,width]，
    # 如果不对x形状进行变换，数组会将输入的x的band参数误以为是width，造成显示错误。
    # strechimg(sample_size_X,sample_size_Y,x[0,:,:,0],x[0,:,:,1],x[0,:,:,2])
    # print(y)
    # print('x shape in get_next_batch',np.shape(x))
    return x, y


def get_1D_app_data_batch(app_data_path):
    # x表示宽，y表示高
    raster, raster_array, xsize, ysize, band_num = read_show_img(app_data_path)
    # print('xsize,ysize',xsize,ysize)
    img_height = ysize
    img_width = xsize
    sample_num_all = img_height * img_width
    channel_1D = 1
    app_xs = np.zeros([sample_num_all, band_num, channel_1D], dtype=float)  # 分别用于存储每一个batch的数据，及其行、列数
    x_num = 0  # int((x_end - x_start)/sample_step_col)
    y_num = 0  # int((y_end - y_start)/sample_step_row)
    for y_i in range(0, img_height):
        x_num = 0
        for x_i in range(0, img_width):
            data_from_raster = raster.ReadAsArray(x_i, y_i, 1, 1)
            data = np.swapaxes(data_from_raster, 0, 1)
            index = y_i * img_width + x_i
            # print('y_i,x_i,index:',y_i,x_i,index)
            app_xs[index, :, :] = data
            x_num += 1
        y_num += 1
    # print('app_xs shape in get 1D app data',np.shape(app_xs))
    return app_xs, img_width, img_height, band_num


def create_1D_cnn(xs, class_num, band_num, win_size_X, channel_1D=1):
    # 此处band_num 相当于2D中的width
    sample_size_X = band_num
    x_image = xs  # width=band_num=15
    # print('x_image shape:',np.shape(x_image))
    ## conv1 layer ##
    W_conv1 = weight_variable([win_size_X, channel_1D, 5])  # [3,1,5][filter_size,input_channels, out_channels]
    b_conv1 = bias_variable([5])
    h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)  # output size 13
    # print('h_conv1 shape:', h_conv1.shape.as_list())

    h_pool1 = max_pool_nx1(h_conv1)  # output size 11
    # print('h_pool1 shape:', h_pool1.shape.as_list())
    # conv2 layer ##
    W_conv2 = weight_variable([win_size_X, 5, 12])  # in_channel: 5, out_channel:12
    b_conv2 = bias_variable([12])
    h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2) + b_conv2)  # output size 9
    h_pool2 = max_pool_nx1(h_conv2)  # output size 7
    # print('h_pool2 shape:', h_pool2.shape.as_list())

    xx = sample_size_X - (win_size_X - 1) * 4
    yy = 1  # round(sample_size_Y/4+0.25)
    ## fc1 layer ##
    W_fc1 = weight_variable([xx * yy * 12, 24])
    b_fc1 = bias_variable([24])
    h_pool2_flat = tf.reshape(h_pool2, [-1, W_fc1.get_shape().as_list()[0]])

    # print('h_pool2_flat shape:', h_pool2_flat.shape.as_list())

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

    # fc2 layer ===out layer##
    W_fc2 = weight_variable([24, class_num])
    b_fc2 = bias_variable([class_num])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # print('create_cnn_finished')
    # print(prediction.get_shape().as_list())
    return prediction


def create_1D_cnn_simple(xs, class_num, band_num, win_size_X, channel_1D=1):
    # pass
    # print('create_cnn_simple begin')
    x_image = xs
    sample_size_X = band_num

    W_conv1 = weight_variable([win_size_X, channel_1D, 32])  #
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)  # in_size:15 output size:13

    # h_pool1 = max_pool_nx1(h_conv1)# in_size:13 output size:11

    W_conv3 = weight_variable([win_size_X, 32, 64])
    b_conv3 = bias_variable([64])
    # h_conv3 = tf.nn.relu(conv1d(h_pool1,W_conv3) + b_conv3)# in_size:11 output size:9
    h_conv3 = tf.nn.relu(conv1d(h_conv1, W_conv3) + b_conv3)  # in_size:13 output size:11

    h_pool2 = max_pool_nx1(h_conv3,pool_size=win_size_X)  # in_size:11 output size:9

    xx = sample_size_X - (win_size_X - 1) * 3
    yy = 1

    W_fc = weight_variable([xx * yy * 64, 128])
    b_fc = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2, [-1, W_fc.get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
    # prediction = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc) + b_fc)
    h_fc_drop = tf.nn.dropout(h_fc1, keep_prob)
    # fc2 layer ===out layer##
    W_fc2 = weight_variable([128, class_num])
    b_fc2 = bias_variable([class_num])
    prediction = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)

    return prediction


def create_1D_cnn_HU(xs, class_num, band_num, win_size_X=11, channel_1D=1):
    x_image = xs
    sample_size_X = band_num
    #win_size_X = 3 ## for MNF PCA 15 bands
    win_size_X=11

    W_conv1 = weight_variable([win_size_X, channel_1D, 20])  #
    b_conv1 = bias_variable([20])
    h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)  # in_size:15 output size:13

    h_pool1 = max_pool_nx1(h_conv1, pool_size=4, stride=2)  #  for MNF PCA 15 bands
    #h_pool1 = max_pool_nx1(h_conv1, pool_size=4, stride=3)  # in_size:13 output size:11
    #xx = 30
    #xx=5 # for MNF PCA 15 bands
    xx=89 #for AVIRIS 191bands
    yy = 1

    W_fc = weight_variable([xx * yy * 20, 100])
    b_fc = bias_variable([100])
    h_pool2_flat = tf.reshape(h_pool1, [-1, W_fc.get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
    # prediction = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc) + b_fc)
    h_fc_drop = tf.nn.dropout(h_fc1, keep_prob)
    # fc2 layer ===out layer##
    W_fc2 = weight_variable([100, class_num])
    b_fc2 = bias_variable([class_num])
    prediction = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)
    return prediction


def train_cnn(win_size_X, win_size_Y, cnn_model='simple'):
    # global prediction
    if cnn_model == 'simple':
        prediction = create_1D_cnn_simple(xs, class_num, band_num, win_size_X, channel_1D=1)
    elif cnn_model == 'HU':
        prediction = create_1D_cnn_HU(xs, class_num, band_num, win_size_X, channel_1D=1)
    else:
        prediction = create_1D_cnn(xs, class_num, band_num, win_size_X, channel_1D=1)
    # prediction = create_cnn(xs,class_num,sample_size_X,sample_size_Y,band_num,win_size_X,win_size_Y)


    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,1.0))
    #                                               ,reduction_indices=[1]))  # loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction + 1e-10)
                                                  , reduction_indices=[1]))  # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 计算准确率
    max_idx_p = tf.argmax(prediction, 1)
    max_idx_l = tf.argmax(ys, 1)
    correct_prediction = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(max_to_keep=3)
    # session及变量初始化
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # 开始训练
    high_acc_num = 0
    acc70 = 0
    acc80 = 0
    acc90 = 0
    eighty=True
    seventy=True
    selected_class = 400  # 每次学习量
    for step in range(25000):  # 学习的次数是，每次学习量是batch(类数*batch_size)
        # batch_xs, batch_ys = mnist.train.next_batch(60)
        batch_xs, batch_ys = get_next_batch(x_data, y_data, selected_class)
        # off_set = num_per_class,batch_size = selected_per_class
        # get_next_batch(x_data, y_data,off_set,batch_size = 20)
        # batch_size是每一个类别用于学习的采样点个数
        # print('shape of batch_ys',np.shape(batch_ys))
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        # print('cross_entropy:',sess.run(cross_entropy,feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5}))

        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1.0})
            print(step, 'batch data accuracy:', acc)
            # print(sess.run(accuracy,feed_dict={xs:x_data,ys:y_data,keep_prob:1.0}))
            if high_acc_num > 0 and acc <= 0.9:
                high_acc_num -= 1

            if 0.8 >= acc > 0.7 and seventy:
                if acc >= acc70:
                    acc70=acc
                    saver.save(sess, "G:\data for manuscripts\AVIRIS20100517\CNN\MNF_simp\simp0.ckpt", global_step=step)
            if 0.9 >= acc > 0.8 and eighty:
                seventy=False
                if acc >= acc80:
                    acc80 = acc
                    saver.save(sess, "G:\data for manuscripts\AVIRIS20100517\CNN\MNF_simp\simp1.ckpt", global_step=step)
                    # break
            if acc > 0.9:
                eighty=False
                if acc >= acc90:
                    acc90 = acc
                    saver.save(sess, "G:\data for manuscripts\AVIRIS20100517\CNN\MNF_simp\simp2.ckpt", global_step=step)
                high_acc_num += 1
                # if high_acc_num >=5:
                #      break

                # print(i,'all data accuracy:',sess.run(accuracy,feed_dict={xs:x_data,ys:y_data,keep_prob:1.0}))
                # print(i,'cross_entropy/loss',sess.run(cross_entropy,feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:1}))


def test_cnn(test_xs, win_size_X, win_size_Y, cnn_model='simple', batch_i=0):
    # global prediction
    # prediction = create_cnn(xs,class_num,sample_size_X,sample_size_Y,band_num,win_size_X,win_size_Y)
    total_size=np.shape(test_xs)[0]
    label=[]
    print('test_xs shape in test cnn:', np.shape(test_xs))
    if cnn_model == 'simple':
        prediction = create_1D_cnn_simple(xs, class_num, band_num, win_size_X, channel_1D=1)
    elif cnn_model == 'HU':
        prediction = create_1D_cnn_HU(xs, class_num, band_num, win_size_X, channel_1D=1)
    else:
        prediction = create_1D_cnn(xs, class_num, band_num, win_size_X, channel_1D=1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "G:\\data for manuscripts\\aviris_oil\\org\\simp\\simp2.ckpt-24000")
        label_position = tf.argmax(prediction, 1)

        if total_size >1000:
            start = 0
            n = int(total_size / 1000)
            for batch_i in range(0, n):
                #predicted_label.append(test_cnn(app_xs[start:start + 1000], win_size_X, win_size_Y, cnn_model, batch_i))
                predicted_label = sess.run(label_position, feed_dict={xs: test_xs[start:start + 1000], keep_prob: 1})
                label.extend(predicted_label)
                start += 1000
                print(start,'shape of label', np.shape(label))
                #print('shape of predicted_label', np.shape(predicted_label))
            predicted_label = sess.run(label_position, feed_dict={xs: test_xs[n * 1000:total_size], keep_prob: 1})
            label.extend(predicted_label)
            print('shape of predicted_label', np.shape(label))
        else:
            label = sess.run(label_position, feed_dict={xs: test_xs, keep_prob: 1})

    return label


if __name__ == '__main__':
    # image_name = 'C:\hyperspectral\AVIRISReflectanceSubset.dat'
    # image_name = 'F:\遥感相关\墨西哥AVIRIS\\f100709t01p00r11\\f100709t01p00r11rdn_b\\f100709t01p00r11rdn_b_sc01_ort_img_QUAC'
    train = False
    test = False
    random_sample = True  # 用于flag是否随机采样
    test_all = True  # 用于flag是否利用所有数据进行验证，如果是FALSE，则只对采样数据进行验证。
    #cnn_model='HU'
    cnn_model = 'simple'
    # cnn_model='Let4'

    sample_size_X = 1  # 训练数据的宽
    sample_size_Y = 1  # 训练数据的高
    class_num = 7  # 训练数据的类数
    band_num = 0
    channel_1D = 1
    x_data = []
    y_data = []
    win_size_X = 3
    win_size_Y = 1
    xs = tf.placeholder(tf.float32, [None, band_num, channel_1D])  #
    ys = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)

    image_name = 'G:\data for manuscripts\AVIRIS20100517\\f100517t01p00r10' \
                 '\\f100517t01p00r10rdn_b\\f100517t01p00r10rdn_b_sc01_ort_img_resized2radiance_resized2_flaashed'
    excel_name = 'G:\data for manuscripts\AVIRIS20100517\\fig_thickness_resized_roated_class_resize_ROIs\ROIs.xlsx'
    #train_excel_name = 'F:\Python\workshop\data\hydata\mannual_samp\Pavia_sample_manual.xlsx'

    if train or test:
        #训练样本位置和测试样本位置存在同一个Excel中，前num_per_class是training samples
        #  从第num_per_class + 1之后的数据是test samples
        # 通常样本选择不是随机的，而是人工选择
        num_per_class = np.array([200, 700, 400, 400, 200, 200])  # 训练数据中，每一类的采样点个数
        # num_per_class = np.array([6431, 18449, 1899, 2864, 1145, 4829, 1130, 3482, 747])
        total_per_class = np.array([29345, 356713, 81948, 28845, 1547, 802])

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
            # 此处返回值得shape是[batch，band,channel]
            s = list(np.shape(x_data))  # 去除掉图像的边缘之后的数据形状，取形状参数的第一个数，即数据batch个数
            # print('s shape',np.shape(s),s[0])
            sample_num = s[0]

            test = False
            train_cnn(win_size_X, win_size_Y, cnn_model)
        elif test:
            if test_all:  # 如果选择利用所有数据进行精度评价
                random_sample = False
                test_num_per_class = total_per_class - num_per_class
                sample_num = np.sum(test_num_per_class)
                #start_row = [1,1,1,1,1,1]#1# + num_per_class  # 表示记录采样点数据的Excel中，数据开始的行，0表示第一行
                start_row = [201, 701, 401, 401, 201, 201]
                end_row = start_row + test_num_per_class - 1
                print('test_num_per,start_row,end_row',test_num_per_class,start_row,end_row)
                start_col = 1  # 表示记录采样点数据的Excel中，数据开始的列，0表示第一列
                end_col = 2
                test_xs, test_ys = get_1D_sample_data(raster, band_num, class_num, test_num_per_class, total_per_class,
                                                      sample_num,
                                                      excel_name, sheet_num, start_row, start_col, end_row, end_col,
                                                      sample_size_X, sample_size_Y, channel_1D, random_sample)

            else:
                test_xs, test_ys = get_next_batch(x_data, y_data, 1200)

            predicted_label = test_cnn(test_xs, win_size_X, win_size_Y, cnn_model)
            label_test = np.array(predicted_label)
            # label_test = np.reshape(label_test,(12,16))
            # plt.imshow(label_test)
            # plt.show()
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # plt.ion()
            # for i in range(0,np.size(x_data,0)):
            #     try:
            #         ax.images.remove(img[0])
            #     except:
            #         pass
            #     fig.suptitle(str(i))
            #     img = ax.imshow(x_data[i,:,:,0])
            #     plt.pause(0.05)
            #     #fig.images.remove(img[i])
            # plt.ioff()
            print('real label:', test_ys.argmax(1))
            real_label = test_ys.argmax(1)
            # real_label =[2,3]
            # result = (predicted_label == real_label) #判断两个数组的对应元素是否相同
            # result =np.array(result)
            # print( result, np.sum(result==True),(np.sum(result==True))/1200*100)    #打印对应元素相等的数量
            conf_mtrx = skmetr.confusion_matrix(real_label, label_test)
            overall_acc = skmetr.accuracy_score(real_label, label_test)
            acc_for_each_class = skmetr.precision_score(real_label, label_test, average=None)
            aver_acc = np.mean(acc_for_each_class)
            aver_acc_score = skmetr.accuracy_score(real_label, label_test)
            kappa_co = skmetr.cohen_kappa_score(real_label, label_test)
            # plt.imshow(conf_mtrx)
            # plt.show()
            print('confusion metrics', conf_mtrx)
            print('overall accuracy:', overall_acc)
            print('accuracy for each class:', acc_for_each_class)
            print('average accuracy:', aver_acc)
            print('average accuracy score:', aver_acc_score)
            print('kappa coefficient:', kappa_co)

    else:
        part_data = False
        app_data_path = image_name  # 'F:\Python\workshop\data\hydata\Pavia_MNF'
        app_xs, x_num, y_num, band_num = get_1D_app_data_batch(app_data_path)
        print('shape of app_xs:', np.shape(app_xs)[0])
        xs = tf.placeholder(tf.float32, [None, band_num, channel_1D])  #
        # # 2l = []
        #
        #
        # total_size = np.shape(app_xs)[0]
        # predicted_label = []
        # if total_size > 1000:
        #     if cnn_model == 'simple':
        #         prediction = create_1D_cnn_simple(xs, class_num, band_num, win_size_X, channel_1D=1)
        #     elif cnn_model == 'HU':
        #         prediction = create_1D_cnn_HU(xs, class_num, band_num, win_size_X, channel_1D=1)
        #     else:
        #         prediction = create_1D_cnn(xs, class_num, band_num, win_size_X, channel_1D=1)
        #     saver = tf.train.Saver()
        #     sess = tf.Session()
        #     saver.restore(sess, "F:/Python/workshop/data/hydata/Pavia_org_1D2/simp-2.ckpt-49600")
        #     label_position = tf.argmax(prediction, 1)
        #
        #
        #     start = 0
        #     n = int(total_size / 1000)
        #     for batch_i in range(0, n):
        #         #predicted_label.append(test_cnn(app_xs[start:start + 1000], win_size_X, win_size_Y, cnn_model, batch_i))
        #         label = sess.run(label_position, feed_dict={xs: app_xs[start:start + 1000], keep_prob: 1})
        #         predicted_label.extend(label)
        #         start += 1000
        #         print(start,'shape of predicted_label', np.shape(predicted_label))
        #         #print('shape of predicted_label', np.shape(predicted_label))
        #     label = sess.run(label_position, feed_dict={xs: app_xs[n * 1000:total_size], keep_prob: 1})
        #     predicted_label.extend(label)
        #     print('shape of predicted_label', np.shape(predicted_label))
        #     #predicted_label.append(test_cnn(app_xs[n * 1000:total_size], win_size_X, win_size_Y, cnn_model, batch_i=n))
        # else:
        predicted_label = test_cnn(app_xs, win_size_X, win_size_Y, cnn_model)


        label = np.reshape(predicted_label, (y_num, x_num))
        label = label + 1
        ttt = np.zeros((y_num, x_num))
        out_tif='G:\\data for manuscripts\\AVIRIS20100517\\CNN\\org_simp\\org_simp_othertrained.tif'
        if part_data: #用在Pivia校园数据的情况，即：整个图像上，有一部分数据是没参与训练和分类的，
            # 在输出时，应当只对参与计算了的像素进行赋值
            num_per_class = np.array([3691, 427, 3905, 3942, 4035, 3788, 3504])  # 训练数据中，每一类的采样点个数
            # num_per_class = np.array([6431, 18449, 1899, 2864, 1145, 4829, 1130, 3482, 747])
            print("if part_data")
            sample_num = np.sum(num_per_class)  # class_num * num_per_class  #训练数据中，所有类采样点的总数。对应后面的batch
            start_row = np.array([1, 1, 1, 1,1, 1, 1])  # 表示记录采样点数据的Excel中，数据开始的行，0表示第一行
            end_row = start_row + num_per_class - 1

            start_col = 0  # 表示记录采样点数据的Excel中，数据开始的列，0表示第一列
            end_col = 1  # 如果行列数字错误，可能出现如下错误：
            # ERROR 5: Access window out of range in RasterIO().  Requested
            # (630,100) of size 10x10 on raster of 634x478.
            sheet_num = class_num  # 表示Excel中sheet的数目，必须与类别数量一致
            position = read_sample_position(excel_name, sheet_num, start_row, start_col, end_row, end_col)

            col_offset = int(sample_size_X / 2)  #该参数主要在二维CNN时有用。因为在get_app_data_batch时，
                         # 已经将图像边缘一圈切掉了，所以此处用offset值，将cnn处理结果图与原始图的位置进行对应。
            # 例如：某个像素在Excel记录中的位置（即原始图位置）为[3,4]，那么在处理结果图中位置应当为y=3-1-offset
            row_offset = int(sample_size_Y / 2)
            in_index = []
            for i in range(1, y_num + 1):
                for j in range(1, x_num + 1):
                    temp1 = np.where((position == [j, i]).all(1))[0] #判断图像上的位置，是不是在Excel记录中
                    if np.size(temp1) != 0:
                        #print(temp1)
                        in_index.append(temp1[0])
            a = position[in_index]
            for i in range(0, np.size(a, 0)):
                col = int(a[i][0]) - 1 - col_offset
                row = int(a[i][1]) - 1 - row_offset
                ttt[row, col] = label[row, col]
                #         in_index.append(a)
            # print(in_index)
            writeTiff(ttt, x_num, y_num, 1, out_tif)
        else:
            ttt = label
            #       print(np.shape(label))
            writeTiff(ttt, x_num, y_num, 1, out_tif)
        plt.imshow(ttt)
        plt.show()