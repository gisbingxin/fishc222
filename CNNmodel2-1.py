# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:17:51 2017

@author: ZQ
"""

import tensorflow as tf
import numpy as np

from common_fun import print_info
from read_CNN_data import read_all_train_data,read_test_data
from savetoimage import save_viaris_img_2
step_offset = 180
def rot_data(x_data,idx):
    x_0 = x_data[idx]
    x_0_90 = x_0
    x_0_180 = x_0
    x_0_270 = x_0
    for i in range(0,224):
        x_0_90[:,:,i] = np.rot90(x_0_90[:,:,i])
        x_0_180[:,:,i] = np.rot90(x_0_180[:,:,i],2)
        x_0_270[:,:,i] = np.rot90(x_0_270[:,:,i],3)
    return x_0,x_0_90,x_0_180,x_0_270
def get_next_batch_2(x_data,y_data,batch_size = 64):
    x = []
    y = []
    for _ in range(batch_size):
        idx = np.random.randint(0,step_offset)
        #print("idx:",idx)
        x.append(x_data[idx])
        y.append(y_data[idx])
        
        idx = idx + step_offset
        x.append(x_data[idx])
        y.append(y_data[idx])
        
        idx = idx + step_offset
        x.append(x_data[idx])
        y.append(y_data[idx])
        
        idx = idx + step_offset
        x.append(x_data[idx])
        y.append(y_data[idx])
        
        idx = idx + step_offset
        x.append(x_data[idx])
        y.append(y_data[idx])
    x = np.array(x)
    y = np.array(y)
    return x,y

def get_next_batch(x_data,y_data,batch_size = 100):
    x = []
    y = []
    for _ in range(batch_size // 20):
        idx = np.random.randint(0,50)
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        idx = idx + 50
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        idx = idx + 50 
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        idx = idx + 50 
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        idx = idx + 50 
        x_0,x_9,x_18,x_27 = rot_data(x_data,idx)
            
        x.append(x_0)
        x.append(x_9)
        x.append(x_18)
        x.append(x_27)
        
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        y.append(y_data[idx])
        
        '''
        x.append(x_data[idx + 50])
        y.append(y_data[idx + 50])
        
        x.append(x_data[idx + 100])
        y.append(y_data[idx + 100])
        '''
    x = np.array(x)
    y = np.array(y)
    return x,y
#定义CNN
# def create_cnn(out_size,w_alpha = 0.01,b_alpha = 0.01):
#     #x = tf.reshape(X,shape=[-1,15,15,360])
#     x = tf.reshape(X,shape=[-1,5,5,224] #5*5*224
#     #w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,360,512]))
#     w_c1 = tf.Variable(w_alpha * tf.random_normal([3,3,224,512]))
#     b_c1 = tf.Variable(b_alpha * tf.random_normal([512]))
#     conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1)) #5*5*512
#     conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #3*3*512
#     conv1 = tf.nn.dropout(conv1,keep_prob)
#     print(conv1.get_shape().as_list())
#     w_c2 = tf.Variable(w_alpha*tf.random_normal([3,3,512,1024]))
#     b_c2 = tf.Variable(b_alpha*tf.random_normal([1024]))
#     conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
#     conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#     conv2 = tf.nn.dropout(conv2,keep_prob)
#     print(conv2.get_shape().as_list())
#     w_c3 = tf.Variable(w_alpha*tf.random_normal([3,3,1024,2048]))
#     b_c3 = tf.Variable(b_alpha*tf.random_normal([2048]))
#     conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3))
#     conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#     conv3 = tf.nn.dropout(conv3,keep_prob)
#     print(conv3.get_shape().as_list())
#     #fully connect layer
#     #w_d = tf.Variable(w_alpha*tf.random_normal([2*2*2048,2048]))
#     w_d = tf.Variable(w_alpha*tf.random_normal([1*1*2048,2048]))
#     b_d = tf.Variable(b_alpha*tf.random_normal([2048]))
#     dense = tf.reshape(conv3,[-1,w_d.get_shape().as_list()[0]])
#     print(dense.get_shape().as_list())
#     dense = tf.nn.relu(tf.add(tf.matmul(dense,w_d),b_d))
#     dense = tf.nn.dropout(dense,keep_prob)
#     print(dense.get_shape().as_list())
#
#     w_out = tf.Variable(w_alpha*tf.random_normal([2048,out_size]))
#     b_out = tf.Variable(b_alpha*tf.random_normal([out_size]))
#     out = tf.add(tf.matmul(dense,w_out),b_out)
#     print(out.get_shape().as_list())
#
#     return out

def create_cnn_3muti3(out_size,w_alpha = 0.01,b_alpha = 0.01):
    #x = tf.reshape(X,shape=[-1,15,15,360])
    x = tf.reshape(X,shape=[-1,3,3,224])
    #w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,360,512]))
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,224,256]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([256]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1))
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1 = tf.nn.dropout(conv1,keep_prob)
    print(conv1.get_shape().as_list())
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3,3,256,512]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([512]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2 = tf.nn.dropout(conv2,keep_prob)
    print(conv2.get_shape().as_list())
    
    #fully connect layer
    w_d = tf.Variable(w_alpha*tf.random_normal([1*1*512,1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv2,[-1,w_d.get_shape().as_list()[0]])
    print(dense.get_shape().as_list())
    dense = tf.nn.relu(tf.add(tf.matmul(dense,w_d),b_d))
    dense = tf.nn.dropout(dense,keep_prob)
    print(dense.get_shape().as_list())
    
    w_out = tf.Variable(w_alpha*tf.random_normal([1024,out_size]))
    b_out = tf.Variable(b_alpha*tf.random_normal([out_size]))
    out = tf.add(tf.matmul(dense,w_out),b_out)
    print(out.get_shape().as_list())
    
    return out

def train_cnn(x_all_data,y_all_data):
    #output = create_cnn(3)
    output = create_cnn_3muti3(5)
    
    print_info("created cnn ...")
    print_info("start train ...")
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=output))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)
    
    max_idx_p = tf.argmax(output,1)
    max_idx_l = tf.argmax(Y,1)    
    
    correct_pred = tf.equal(max_idx_p,max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    
    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        step = 0
        while True: #一直学习，知道精度达到90%以上
            batch_x,batch_y = get_next_batch_2(x_all_data,y_all_data)
            _,loss_ = sess.run([optimizer,loss],feed_dict = {X:batch_x,Y:batch_y,keep_prob:0.75})
            print(step,loss_)
            '''
            out = sess.run([output],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            print("out:",out)
            y = sess.run([Y],{X:batch_x,Y:batch_y,keep_prob:0.75})
            print("y:",y)
            max_p = sess.run([max_idx_p],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            max_l = sess.run([max_idx_l],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            print("max_p:",max_p)
            print("max_l",max_l)
            break
            '''
            if step % 10 == 0 and step != 0:
                batch_x_test,batch_y_test = get_next_batch_2(x_all_data,y_all_data)
                acc = sess.run(accuracy,feed_dict={X:batch_x_test,Y:batch_y_test,keep_prob:1.0})
                print(step,acc)
                if acc > 0.7:
                    saver.save(sess,"./model/cnn.model",global_step=step)
                if acc > 0.8:
                    saver.save(sess,"./model/cnn.model",global_step=step)
                    #break
                if acc > 0.9:
                    saver.save(sess,"./model/cnn.model",global_step=step)
                    break
            step += 1
def test_cnn(x_data):
    
    output = create_cnn_3muti3(5)
    #2l = []
    saver = tf.train.Saver()
    #new_saver = tf.train.import_meta_graph("./model/cnn.model-470.meta") 
    #saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess: 
        
        #new_saver.restore(sess,tf.train.latest_checkpoint('./model/')) 
        #new_saver.restore(sess,"./model/cnn.model-470")                      
        saver.restore(sess,"./model_3/cnn.model-12180")
        #all_vars = tf.trainable_variables()
        #sess.run(tf.global_variables_initializer())
        preject = tf.argmax(output,1)
        
        x_in = np.array(x_data)
        label = sess.run(preject,feed_dict={X:[x_in],keep_prob:1})
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
    #
    train = 0                                        
    if train == 1:
        x_data,y_data = read_all_train_data()
        #X = tf.placeholder(tf.float32,[84,15,15,360])
        #Y = tf.placeholder(tf.float32,[84,3])
        
        X = tf.placeholder(tf.float32,[None,5,5,224])
        Y = tf.placeholder(tf.float32,[None,5])
        
        keep_prob = tf.placeholder(tf.float32)
        train_cnn(x_data,y_data)
    if train == 0:
        #x_data = read_test_data('E:/Imaging/CNNTest/test_108_aviris_3.txt')
        x_data = read_test_data('E:/Imaging/ROI_AVIRIS_test/test_roi_6_frm_envi.txt',3)
        #x_in = x_data[:24]
        #x_in = np.array(x_in)
        tf.reset_default_graph()  
        X = tf.placeholder(tf.float32,[None,3,3,224])
        keep_prob = tf.placeholder(tf.float32)
        l = test_cnn(x_data)
        '''
        a = l
        a = np.array(a)
        a = a.reshape(1,10000)

        save_viaris_img_2(a,'cnn1450.jpg',100,100)
        '''