from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from self_strech_img import minmaximg, linearstretching,strechimg,strechimg_pan
from self_read_position_excel import read_sample_position
from mat2tiff import writeTiff
#from self_img_read2txt import read_from_img

#图像读取与现实，输入文件完整路径名，输出图像的长、宽和波段数
def read_show_img(image_name,show_img = False):
    #raster = gdal.Open('C:\\test.jpg')
    #image_name = 'C:\hyperspectral\AVIRISReflectanceSubset.dat'
    raster = gdal.Open(image_name)
    #raster = gdal.Open('e:\qwer.img')
    print('Reading the image named:',image_name)
    # print('geotrans is ', raster.GetGeoTransform())
    # print('projection is ',raster.GetProjection())
    xsize = raster.RasterXSize  # RasterXSize是图像宽
    ysize = raster.RasterYSize  # RasterYSize是图像的高
    band_num = raster.RasterCount #RasterCount是图像的波段数
    print('in read and show x,y',xsize,ysize)

    # 获取rgb对应波段数值，传递给拉伸显示模块

    raster_array = raster.ReadAsArray() #raster_array.shape=(191,128,155)对应（band_num,Ysize,Xsize),
                                        # Ysize代表行数，Xsize代表列数
    raster_array_np = np.array(raster_array)

#
    #print('rnp',np.shape(rnp))
    print('raster_array shape in read_show img',raster_array_np.shape)
    if band_num >= 3:
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
        #print(np.shape(img_data))
        if show_img:
            strechimg_pan(xsize,ysize,img_data)
            #print()
    return raster,raster_array,xsize, ysize,band_num

#获取采样点的数据，输入图像读取后的raster和波段数，输出x_data,y_data。分别是采样点位置及其数据、该点的类别数组
def get_sample_data(raster,band_num,xsize,ysize,class_num,num_per_class,sample_num,
                    sample_size_X,sample_size_Y,
                    excel_name,sheet_num,start_row,start_col,end_row, end_col):
    #读取Excel中存储的采样点位置数据，返回值中sample_num代表类别编号
    sample_pos = read_sample_position(excel_name,sheet_num,start_row,start_col,end_row, end_col)
                            # 该函数返回所选择的sample的位置点，是一个三维数组shape为[class_num,X_num,Y_num]
    #print(type(sample))

    #x_data = np.empty([72,191,3,3],dtype = float)
    # #sample定义为[batch，band,Ysize,Xsize],batch这里是所有类别采样点的数量
    x_data = np.zeros([sample_num, band_num, sample_size_Y, sample_size_X], dtype=float)
    #x_data = []
            # 用来存储各类别各采样点的图像灰度值
    y_data = np.zeros([sample_num,class_num],dtype=int)

    margin_pixel=[]
    loc_origin = 0
    for i in range(0,class_num):    #i表示类的序号
        for x_i in range(loc_origin,loc_origin + num_per_class[i]):  #x_i 表示在该类内的序号，即行号
            #for y_i in range(0,2):
            x_locate = int(sample_pos[x_i,0]) #GDAL中的ReadAsArray传入参数应当为int型
                                            # 而此处sample是numpy.int32，所以需要进行数据类型转换
            y_locate = int(sample_pos[x_i,1])
            print('x,y location:', x_locate,y_locate)
            if x_locate <= int(sample_size_X/2) or y_locate <= int(sample_size_Y/2) or\
                            x_locate > (xsize-int(sample_size_X/2)) or y_locate > (ysize-int(sample_size_Y/2)):
                margin_pixel.append(x_i)
                break
            else:
                x_left_upper = x_locate - int(sample_size_X/2)  #获取sample窗口的左上角点坐标值。
                                                        # ReadAsArray中前两个参数是数据读取的起始位置，后两个参数是窗口大小。
                                                        # 为了让我们选择的点仍然是子窗口的中心点，所以进行该步处理。
                y_left_upper = y_locate - int(sample_size_Y/2)
                #print(type(x_locate))
                data = raster.ReadAsArray(x_left_upper,y_left_upper,sample_size_X,sample_size_Y)
                x_data[x_i, :, :, :] = data
                  #y_data[num_per_class[i]*i+x_i,i] = 1             #与x_data对应的batch处，赋值为1，其余位置为0
                y_data[x_i, i] = 1
        loc_origin = loc_origin + num_per_class[i]
    print('x_data shape before delete',np.shape(x_data))
    x_data=np.delete(x_data,margin_pixel,0)
    y_data=np.delete(y_data,margin_pixel,0)
    print('x_data shape in getsampledata',np.shape(x_data))
    return x_data,y_data


#在所有采样数据中，随机选择一组数据组成batch，其大小也就是一次学习多少个batch。
# 输入为采样位置及数据x_data、类别数据y_data，偏移量off_set和一次学习的数据量。
# 输出是随机产生的batch_size大小的一系列数据及类别（x,y)
# off_set是指每个类别的batch之间相差的数字，等于每个类别中采样点的数，比如0-9为一类，10-19为一类，那么off_set就是10
def get_next_batch(x_data, y_data,off_set,sample_num,batch_size = 20):
    x=[]
    y=[]
    idxs=[]
    for _ in range(batch_size):
        idx = np.random.randint(0,sample_num)
            #依次在每一类的sample中随机选择一个，放入x，y数组中，几种类别共同组成一个batch
        x.append(x_data[idx])
        y.append(y_data[idx])
    x = np.array(x)
    y = np.array(y)
    #print('x shape in get nextbatch',x.shape)
    # x的shape为[batch,height, width,band]
    # y的shape 为[batch,class_num]
    # 如果想对某个batch中的图进行显示，应当先将x的形状转换为[batch,band,height,width].
    # 因为在python或numpy中，数组的形状就是[batch,band,height,width]，
    # 如果不对x形状进行变换，数组会将输入的x的band参数误以为是width，造成显示错误。
    # strechimg(sample_size_X,sample_size_Y,x[0,:,:,0],x[0,:,:,1],x[0,:,:,2])
    # print(y)
    return x,y

# 用于获取实际应用时的图像数据（非测试数据）
# 输入实际数据的地址，数据每一个batch的长宽，以及采样间隔（移动步长）
def get_app_data_batch(app_data_path,sample_height,sample_width,sample_step_row,sample_step_col,x_start,y_start):
    # x表示宽，y表示高
    raster, raster_array, xsize, ysize, band_num = read_show_img(app_data_path)
    #print('xsize,ysize',xsize,ysize)
    img_height = ysize
    img_width = xsize
    #print('img heigth and width:',img_height,img_width)
    # x_start = 1
    # y_start = 1
    # x_margin =round((sample_width+1)/2)
    # y_margin = round((sample_height+1)/2)
    x_margin =int(sample_width/2)
    y_margin = int(sample_height/2)
    x_end = img_width  - x_margin
    y_end = img_height - y_margin

    app_xs = [] #分别用于存储每一个batch的数据，及其行、列数
    #x_num = 0#int((x_end - x_start)/sample_step_col)
    y_num = 0#int((y_end - y_start)/sample_step_row)
    for y_i in range(y_start,y_end,sample_step_row):
        x_num=0
        for x_i in range(x_start,x_end,sample_step_col):
            x_left_upper = x_i - int(sample_width / 2)
            y_left_upper = y_i - int(sample_height / 2)
            #print('xlu,ylu',x_left_upper,y_left_upper)
            data = raster.ReadAsArray(x_left_upper, y_left_upper, sample_width, sample_height)
            app_xs.append(data)
            x_num += 1
        y_num+=1
    app_xs = np.array(app_xs)
    print('shape of app_xs',np.shape(app_xs))
    app_xs = app_xs.swapaxes(1, 3)  # 将两个轴的数据对换，使[batch,band,xsize/height,ysize/width],
            # 变为[bantch,xsize/height,ysize/width,band]
    app_xs = app_xs.swapaxes(1,2)

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
    # print('new shape of app_xs',np.shape(app_xs))
    return app_xs,x_num,y_num,band_num
        #def compute_accuracy(sess,prediction,keep_prob,v_xs, v_ys):
# def compute_accuracy(v_xs, v_ys,sess):
#     global prediction
#     global xs
#     global ys
#     global keep_prob
#     #print('compute_acc')
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
#     correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
#     #print(result)
#     return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#def create_cnn(xs,keep_prob,class_num,sample_size_X,sample_size_Y,band_num):
def create_cnn(xs,class_num, sample_size_X, sample_size_Y, band_num,win_size_X,win_size_Y):

    #print('create_cnn begin')
    x_image = tf.reshape(xs, [-1, sample_size_Y, sample_size_X, band_num])
    #print('x_image shape in create_cnn',x_image.shape)  # [n_samples, 28,28,1]

    ## conv1 layer ##
    W_conv1 = weight_variable([win_size_Y, win_size_X, band_num, 64])  # patch 3x3, in size 199, out size 199*3
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

    # conv2 layer ##
    W_conv2 = weight_variable([win_size_Y, win_size_X, 64, 128])  # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

    xx = round(sample_size_X/4+0.25)
    yy = round(sample_size_Y/4+0.25)
    ## fc1 layer ##
    W_fc1 = weight_variable([xx * yy * 128, 256])
    b_fc1 = bias_variable([256])
    #h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 1 * 512])
    #print(W_fc1.get_shape().as_list()[0]])
    h_pool2_flat = tf.reshape(h_pool2,[-1,W_fc1.get_shape().as_list()[0]])
    #print(tf.shape(h_pool2_flat))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

    # fc2 layer ===out layer##
    W_fc2 = weight_variable([256, class_num])
    b_fc2 = bias_variable([class_num])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    #print('create_cnn_finished')
    #print(prediction.get_shape().as_list())
    return prediction

# 以下是比较简化了的一个cnn
def create_cnn_simple(xs,class_num, sample_size_X, sample_size_Y, band_num,win_size_X,win_size_Y):
    #pass
    #print('create_cnn_simple begin')
    x_image = tf.reshape(xs, [-1, sample_size_Y, sample_size_X, band_num])

    W_conv1 = weight_variable([win_size_Y, win_size_X, band_num, 32])  # patch 3x3, in size 199, out size 199*3
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size

    h_pool1 = max_pool_2x2(h_conv1)

    W_conv3 = weight_variable([win_size_Y, win_size_X, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool1,W_conv3) + b_conv3)

    h_pool2 = max_pool_2x2(h_conv3) #24

    xx = round(sample_size_X/4+0.25)
    yy = round(sample_size_Y/4+0.25)

    W_fc = weight_variable([xx * yy * 64, 128])
    b_fc = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2, [-1, W_fc.get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
    #prediction = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc) + b_fc)
    h_fc_drop = tf.nn.dropout(h_fc1, keep_prob)
    # fc2 layer ===out layer##
    W_fc2 = weight_variable([128, class_num])
    b_fc2 = bias_variable([class_num])
    prediction = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)

    return prediction

def train_cnn(win_size_X,win_size_Y):


    #global prediction
    #prediction = create_cnn(xs,class_num,sample_size_X,sample_size_Y,band_num,win_size_X,win_size_Y)
    prediction = create_cnn(xs, class_num, sample_size_X, sample_size_Y, band_num, win_size_X, win_size_Y)

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,1.0))
    #                                               ,reduction_indices=[1]))  # loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction + 1e-10)
                                                  ,reduction_indices=[1]))  # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 计算准确率
    max_idx_p = tf.argmax(prediction, 1)
    max_idx_l = tf.argmax(ys, 1)
    correct_prediction = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(max_to_keep=3)
    #session及变量初始化
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    #开始训练
    high_acc_num = 0
    acc70 = 0
    acc80 = 0
    acc90 = 0
    selected_per_class = 200 # 每次学习量
    for step in range(10000):  # 学习的次数是，每次学习量是batch(类数*batch_size)
        #batch_xs, batch_ys = mnist.train.next_batch(60)
        batch_xs, batch_ys= get_next_batch(x_data, y_data, num_per_class, sample_num,selected_per_class)
                                                    #off_set = num_per_class,batch_size = selected_per_class
                                                    #get_next_batch(x_data, y_data,off_set,batch_size = 20)
                                                    # batch_size是每一个类别用于学习的采样点个数
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        #print('cross_entropy:',sess.run(cross_entropy,feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5}))

        if step % 50 == 0:
            acc = sess.run(accuracy,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1.0})
            print(step, 'batch data accuracy:', acc)
            #print(sess.run(accuracy,feed_dict={xs:x_data,ys:y_data,keep_prob:1.0}))
            if high_acc_num >0 and  acc <= 0.9:
                high_acc_num -= 1

            # if 0.8 >= acc > 0.7:
            #     if acc >= acc70:
            #         acc70=acc
            #         saver.save(sess, "./Pavia_MNF/MNF_model-1.ckpt", global_step=step)
            # if 0.9 >= acc > 0.8:
            #     if acc>= acc80:
            #         acc80=acc
            #         saver.save(sess, "./Pavia_MNF/MNF_model-2.ckpt", global_step=step)
                # break
            if acc > 0.9:
                if acc>=acc90:
                    acc90=acc
                    saver.save(sess, "./Pavia_MNF/MNF_model-3.ckpt", global_step=step)
                high_acc_num += 1
                # if high_acc_num >=5:
                #      break

            #print(i,'all data accuracy:',sess.run(accuracy,feed_dict={xs:x_data,ys:y_data,keep_prob:1.0}))
            #print(i,'cross_entropy/loss',sess.run(cross_entropy,feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:1}))


def test_cnn(test_xs,win_size_X,win_size_Y):


    #global prediction
    #prediction = create_cnn(xs,class_num,sample_size_X,sample_size_Y,band_num,win_size_X,win_size_Y)
    prediction = create_cnn(xs, class_num, sample_size_X, sample_size_Y, band_num, win_size_X, win_size_Y)
    # 2l = []
    saver = tf.train.Saver()
    # new_saver = tf.train.import_meta_graph("./model/cnn.model-470.meta")
    # saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        # new_saver.restore(sess,tf.train.latest_checkpoint('./model/'))
        # new_saver.restore(sess,"./model/cnn.model-470")
        saver.restore(sess, "./Pavia_MNF/MNF_model-3.ckpt-4350")
        # all_vars = tf.trainable_variables()
        # sess.run(tf.global_variables_initializer())
        label_position = tf.argmax(prediction, 1)

        #x_in = np.array(x_data)
        label = sess.run(label_position, feed_dict={xs: test_xs, keep_prob: 1})
        '''
        for i in range(len(x_data)//199):
            x_in = x_data[i*199:(i+1)*199]
            x_in = np.array(x_in)
            label = sess.run(preject,feed_dict={X:x_in,keep_prob:1})
            l.append(label)            
        #print(label)
        '''
    return label

if __name__ == '__main__':
    # image_name = 'C:\hyperspectral\AVIRISReflectanceSubset.dat'
    # image_name = 'F:\遥感相关\墨西哥AVIRIS\\f100709t01p00r11\\f100709t01p00r11rdn_b\\f100709t01p00r11rdn_b_sc01_ort_img_QUAC'
    train = False
    test = False

    sample_size_X = 3  #训练数据的宽
    sample_size_Y = 3  #训练数据的高
    class_num = 9       #训练数据的类数
    band_num = 0
    x_data =[]
    y_data =[]
    win_size_X=3
    win_size_Y=3
    xs = tf.placeholder(tf.float32, [None, sample_size_Y,sample_size_X,band_num])   #
    ys = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)

    image_name = 'F:\Python\workshop\data\hydata\Pavia_MNF'
    excel_name = 'F:\Python\workshop\data\hydata\PaviaU.xlsx'


    if train or test:

        #image_name = 'F:\Python\workshop\\fishc\\aviris_oil\\aviris_subsize_PCA12.img'
        #image_name = 'F:\Python\workshop\\fishc\\aviris_oil\mnf\inverse_MNF data.img'
        #image_name = 'F:\Python\workshop\\fishc\\aviris_oil\mnf\mnf data'
        #raster = gdal.Open('C:\\test.jpg')
        #C:\Program Files\Exelis\ENVI53\data\qb_boulder_msi
        num_per_class = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200])  # 训练数据中，每一类的采样点个数
        # num_per_class = np.array([6431, 18449, 1899, 2864, 1145, 4829, 1130, 3482, 747])
        sample_num = np.sum(num_per_class)  # class_num * num_per_class  #训练数据中，所有类采样点的总数。对应后面的batch

        start_row = 200  # 表示记录采样点数据的Excel中，数据开始的行，0表示第一行
        end_row = start_row + num_per_class - 1

        start_col = 0  # 表示记录采样点数据的Excel中，数据开始的列，0表示第一列
        end_col = 1  # 如果行列数字错误，可能出现如下错误：
        # ERROR 5: Access window out of range in RasterIO().  Requested
        # (630,100) of size 10x10 on raster of 634x478.
        sheet_num = class_num  # 表示Excel中sheet的数目，必须与类别数量一致

        show_img = False #用于判断是否对图像进行显示
        raster, raster_array, xsize, ysize, band_num = read_show_img(image_name,show_img)    #读取遥感影像

        # sample_size_X = 46  #训练数据的宽
        # sample_size_Y = 80  #训练数据的高
        # class_num = 3       #训练数据的类数
        xs = tf.placeholder(tf.float32, [None, sample_size_Y, sample_size_X, band_num])  #


        #show
        #print('before x_data')

        x_data,y_data = get_sample_data(raster,band_num,xsize,ysize,class_num,num_per_class,sample_num,
                                        sample_size_X,sample_size_Y,
                                        excel_name,sheet_num,start_row,start_col,end_row, end_col)
                                                #此处返回值得shape是[batch，band,Xsize,Ysize]
        s=list(np.shape(x_data))#去除掉图像的边缘之后的数据形状，取形状参数的第一个数，即数据batch个数
        #print('s shape',np.shape(s),s[0])
        sample_num=s[0]
        # print(x_data.shape(0))
        x_data = x_data.swapaxes(1,3)   #将两个轴的数据对换，使[batch,band,xsize/height,ysize/width],
                                        # 变为[bantch,xsize/height,ysize/width,band]
        x_data = x_data.swapaxes(1,2)

        if train:
            test = False
            train_cnn(win_size_X, win_size_Y)
        elif test:
            test_xs, test_ys = get_next_batch(x_data, y_data, num_per_class, sample_num, 1200)
                    # print(test_xs.shape)
                    # strechimg(sample_size_X,sample_size_Y,test_xs[0,:,:,0],test_xs[0,:,:,1],test_xs[0,:,:,2])
            #test_xs = x_data
            predicted_label = test_cnn(test_xs,win_size_X,win_size_Y)

                    #predicted_label = np.array(predicted_label)
                #print('predicted_label:',predicted_label)
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
            print('real label:',test_ys.argmax(1))
            real_label = test_ys.argmax(1)
                # real_label =[2,3]
            result = (predicted_label == real_label) #判断两个数组的对应元素是否相同
            result =np.array(result)
            print( result, np.sum(result==True))    #打印对应元素相等的数量

    else:
        part_data=True
        app_data_path = image_name#'F:\Python\workshop\data\hydata\Pavia_MNF'
        app_sample_size_width =sample_size_X
        app_sample_size_height =sample_size_Y
        app_sample_step_row =1
        app_sample_step_col =1
        x_start =1
        y_start=1
        app_xs,x_num,y_num,band_num=get_app_data_batch(app_data_path, app_sample_size_height, app_sample_size_width,
                                              app_sample_step_row, app_sample_step_col,x_start,y_start)

        xs = tf.placeholder(tf.float32, [None, sample_size_Y, sample_size_X, band_num])  #

        predicted_label = test_cnn(app_xs,win_size_X,win_size_Y)
        label = np.reshape(predicted_label,(y_num,x_num))
        label = label + 1
        ttt=np.zeros((y_num,x_num))
        if part_data:
            num_per_class = np.array([6631, 18649, 2099, 3064, 1345, 5029, 1330, 3682, 947])  # 训练数据中，每一类的采样点个数
            # num_per_class = np.array([6431, 18449, 1899, 2864, 1145, 4829, 1130, 3482, 747])
            sample_num = np.sum(num_per_class)  # class_num * num_per_class  #训练数据中，所有类采样点的总数。对应后面的batch
            start_row = 1  # 表示记录采样点数据的Excel中，数据开始的行，0表示第一行
            end_row = start_row + num_per_class - 1

            start_col = 0  # 表示记录采样点数据的Excel中，数据开始的列，0表示第一列
            end_col = 1  # 如果行列数字错误，可能出现如下错误：
            # ERROR 5: Access window out of range in RasterIO().  Requested
            # (630,100) of size 10x10 on raster of 634x478.
            sheet_num = class_num  # 表示Excel中sheet的数目，必须与类别数量一致
            position = read_sample_position(excel_name, sheet_num, start_row, start_col, end_row, end_col)

            col_offset = int(sample_size_X / 2)
            row_offset = int(sample_size_Y / 2)
            in_index=[]
            for i in range(1, 100):#y_num+1):
                for j in range(1, 100):#x_num+1):
                    a = np.where((position == [i+row_offset,j+col_offset]).all(1))
                    if a != []:
                        print(a)
            #         in_index.append(a)
            # print(in_index)
 #       writeTiff(label,x_num,y_num,1,'f:\\test.tif')

 #       print(np.shape(label))
        plt.imshow(ttt)
        plt.show()
