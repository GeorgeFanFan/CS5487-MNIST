import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import csv
import math
import pandas as pd

from tensorflow.python.ops import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



#读取数据
#filename_queue = tf.data.Dataset(["digits4000_digits_vec.csv", "digits4000_digits_labels.csv","digits4000_trainset.csv", "digits4000_testset.csv"],shuffle=False)
filename1 = np.loadtxt("digits4000_digits_vec.txt")#4000*784
filename1 = filename1[0:2000,:]
print(filename1.shape)

filename11 = np.loadtxt("digits4000_digits_vec.txt")#4000*784
filename11 = filename11[2000:4000,:]
print(filename11.shape)

filename2 = np.loadtxt("digits4000_digits_labels.txt")#4000*1
filename2 = filename2.reshape(4000,1)
temp = np.zeros([4000,9])
print(type(filename2))#np.array
filename2 = np.hstack((filename2,temp))
print(filename2.shape)
for i in range (4000):
    if (filename2[i,0] == 0):
        filename2[i,0] = 1
    elif(filename2[i,0] == 1):
        filename2[i,1]=1
        filename2[i,0]=0
    elif (filename2[i, 0] == 2):
        filename2[i, 2] = 1
        filename2[i,0] = 0
    elif (filename2[i, 0] == 3):
        filename2[i, 3] = 1
        filename2[i,0] = 0
    elif (filename2[i, 0] == 4):
        filename2[i, 4] = 1
        filename2[i,0] = 0
    elif (filename2[i, 0] == 5):
        filename2[i, 5] = 1
        filename2[i,0] = 0
    elif (filename2[i, 0] == 6):
        filename2[i, 6] = 1
        filename2[i,0] = 0
    elif (filename2[i, 0] == 7):
        filename2[i, 7] = 1
        filename2[i,0] = 0
    elif (filename2[i, 0] == 8):
        filename2[i, 8] = 1
        filename2[i,0] = 0
    elif (filename2[i, 0] == 9):
        filename2[i, 9] = 1
        filename2[i,0] = 0
filename2 = filename2[0:2000,:]
print(filename2.shape)
print(filename2)

filename22 = np.loadtxt("digits4000_digits_labels.txt")#4000*1
filename22 = filename22.reshape(4000,1)
temp = np.zeros([4000,9])
print(type(filename22))#np.array
filename22 = np.column_stack((filename22,temp))
print(filename22.shape)
for i in range (4000):
    if (filename22[i, 0] == 0):
        filename22[i, 0] = 1
    elif(filename22[i,0] == 1):
        filename22[i,1]=1
        filename22[i,0]=0
    elif (filename22[i, 0] == 2):
        filename22[i, 2] = 1
        filename22[i,0] = 0
    elif (filename22[i, 0] == 3):
        filename22[i, 3] = 1
        filename22[i,0] = 0
    elif (filename22[i, 0] == 4):
        filename22[i, 4] = 1
        filename22[i,0] = 0
    elif (filename22[i, 0] == 5):
        filename22[i, 5] = 1
        filename22[i,0] = 0
    elif (filename22[i, 0] == 6):
        filename22[i, 6] = 1
        filename22[i,0] = 0
    elif (filename22[i, 0] == 7):
        filename22[i, 7] = 1
        filename22[i,0] = 0
    elif (filename22[i, 0] == 8):
        filename22[i, 8] = 1
        filename22[i,0] = 0
    elif (filename22[i, 0] == 9):
        filename22[i, 9] = 1
        filename22[i,0] = 0
filename22 = filename22[2000:4000,:]
#print(filename22)

filename_train = np.hstack((filename1,filename2))
filename_test = np.hstack((filename11,filename22))
print(filename_train.shape)
print(filename_test.shape)

#print(filename)
#dataset = tf.data.TextLineDataset(filename)
#reader = tf.TextLineReader()
dataset1 = tf.data.Dataset.from_tensor_slices({'image':filename_train[:,0:784],'label':filename_train[:,784:794]})
dataset1 = dataset1.repeat()
dataset1 = dataset1.batch(50)
iterator1 = dataset1.make_one_shot_iterator()
one_element1 = iterator1.get_next()
image1 = one_element1['image']
image1 = tf.cast(image1,tf.float32)
label1 = one_element1['label']
label1 = tf.cast(label1,tf.float32)
#one_element1 = tf.cast(one_element1, tf.float32)

'''dataset11 = tf.data.Dataset.from_tensor_slices(filename11)
dataset11 = dataset11.batch(50).repeat(50)
iterator11 = dataset11.make_one_shot_iterator()
one_element11 = iterator11.get_next()
one_element11 = tf.cast(one_element11, tf.float32)'''

dataset2 = tf.data.Dataset.from_tensor_slices({'image':filename_test[:,0:784],'label':filename_test[:,784:794]})
#print(len(dataset2))
dataset2 = dataset2.repeat()
dataset2 = dataset2.batch(50)
iterator2 = dataset2.make_one_shot_iterator()
one_element2 = iterator2.get_next()
image2 = one_element2['image']
image2 = tf.cast(image2,tf.float32)
label2 = one_element2['label']
label2 = tf.cast(label2,tf.float32)

#one_element2 = tf.cast(one_element2, tf.float32)
#print(len(one_element2))

'''dataset22 = tf.data.Dataset.from_tensor_slices(filename22)
dataset22 = dataset22.batch(50).repeat(50)
iterator22 = dataset22.make_one_shot_iterator()
one_element22 = iterator22.get_next()
one_element22 = tf.cast(one_element22, tf.float32)'''

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None,784], name='x-input')
    y = tf.placeholder(tf.float32, [None,10], name='y-input')


#卷积层，最大池化层，全连接层

#1.定义权重(w)和偏差量(b)
def Weight(shape):
    initial = tf.truncated_normal(shape,stddev=0.1) #从截断的正态分布中输出随机值，标准差0.1
    return tf.Variable(initial)

def Bias(shape):
    initial = tf.constant(0.1,shape=shape)  #偏差量设为常数0.1
    return tf.Variable(initial)

#2.定义卷积层
def Conv2d(x,w):#图像和卷积核的卷积操作
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME") #Tensorflow的卷积函数，strides长度固定四，步长为1

#3.定义最大池化层,池化即下采样，规模一般为2*2
def Maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") #[1, height, width, 1]

#4.reshape image，为了神经网络的layer可以使用image数据，要将其转换为4d的张量(Number, width, height, channels)
x_image = tf.reshape(x, [-1, 28, 28,1])

#5.搭建第一个卷积层，并最大池化，原图像28*28
#步骤：使用32个5*5的filter，并通过最大池化
w_conv1 = Weight([5, 5, 1, 32])
b_conv1 = Bias([32])

h_conv1 = tf.nn.relu(Conv2d(x_image, w_conv1) + b_conv1) #公式：w*x+b; tf.nn.relu是将矩阵内小于0的数置为0 ，28*28*32
h_pool1 = Maxpool(h_conv1) #最大池化，把卷积得出的结果输进最大池化 ， 14*14*32，32张图

#6.搭建第二个卷积层，并最大池化
#步骤：使用32个5*5的filter，并通过最大池化
w_conv2 = Weight([5,5,32,64])
b_conv2 = Bias([64])

h_conv2 = tf.nn.relu(Conv2d(h_pool1, w_conv2) + b_conv2) #14*14*64
h_pool2 = Maxpool(h_conv2) #7*7*64，64张图

#7.搭建全连接层(Fully-connected layer),步骤和卷积层很像的
#全连接层需要把输入拉成一个列项向量， 即需要将上一层的输出，展开成1d的神经层
w_fc1 = Weight([7*7*64, 1024])
b_fc1 = Bias([1024])

h_pool2_flatten = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, w_fc1) + b_fc1)

#8.防止过拟合，添加Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob)

#9.输出，输出一个线性结果
w_fc2 = Weight([1024,10])
b_fc2 = Bias([10])

tf.summary.histogram('weight', w_fc2)
#tf.summary.scalar('weight_Scalar', w_fc2)
tf.summary.histogram('bias',b_fc2)
#tf.summary.scalar('bias_Scalar',b_fc2)

h_shuchu = tf.matmul(h_fc1_dropout, w_fc2) + b_fc2


#10.训练， 评估
#tf.reduce_mean求平均值，此处未设维度，返回值应该是一个数；接着求labels和log函数的交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits= h_shuchu))

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(y, 1), tf.argmax(h_shuchu, 1)), tf.float32))

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

#创建会话，开始执行函数
with tf.Session() as sess:
    #train_writer = tf.summary.FileWriter('/train', sess.graph)
    summary_writer = tf.summary.FileWriter('E:\Python Pregramme Asset\MNIST1\mnist_logs\log', sess.graph)
    tf.global_variables_initializer().run()  # 初始化变量

    #print(type(np_one_element1))
    #print(len(np_one_element1))
    #print(len(np_one_element2))
    for i in range(500):
        np_one_element1 = sess.run(image1)
        np_one_element2 = sess.run(label1)
        #print(np_one_element1)
        np_one_element11 = sess.run(image2)
        np_one_element22 = sess.run(label2)

        #batch1 = ds1.make_one_shot_iterator().get_next()
        #batch2 = ds2.make_one_shot_iterator().get_next()
        if i%100 == 0:
            train_accuracy1 = accuracy.eval(feed_dict={x:np_one_element1, y:np_one_element2,keep_prob:0.5})
            print('setup {},the train accuracy: {}'.format(i, train_accuracy1))
        #rmse = math.sqrt(metrics.mean_squared_error())
        #print(rmse)
        #summary_writer.add_summary(train_accuracy1)


        train_step.run(feed_dict={x: np_one_element1, y: np_one_element2,keep_prob:0.5})
        #summary, _ = sess.run([merged, train_step], feed_dict={x: np_one_element1, y: np_one_element2, keep_prob: 0.5})
        #summary_writer.add_summary(summary, i)
    test_accuracy1 = accuracy.eval(feed_dict={x:np_one_element11, y:np_one_element22,keep_prob:1.})


    print("The test accuracy :{}".format(test_accuracy1))
    for i in range(500):
        np_one_element1 = sess.run(image1)
        np_one_element2 = sess.run(label1)
        #print(np_one_element1)
        np_one_element11 = sess.run(image2)
        np_one_element22 = sess.run(label2)

        #batch1 = ds1.make_one_shot_iterator().get_next()
        #batch2 = ds2.make_one_shot_iterator().get_next()
        if i%100 == 0:
            train_accuracy2 = accuracy.eval(feed_dict={x:np_one_element11, y:np_one_element22,keep_prob:0.5})
            print('setup {},the train accuracy: {}'.format(i, train_accuracy2))

        #rmse = math.sqrt(metrics.mean_squared_error())
        #print(rmse)
        train_step.run(feed_dict={x: np_one_element11, y: np_one_element22,keep_prob:0.5})
        summary, _ = sess.run([merged, train_step], feed_dict={x: np_one_element11, y: np_one_element22, keep_prob: 0.5})
        summary_writer.add_summary(summary, i)
    test_accuracy2 = accuracy.eval(feed_dict={x:np_one_element1, y:np_one_element2,keep_prob:1.})

    summary_writer.close()

    print("The test accuracy :{}".format(test_accuracy2))
    test_accuracy = test_accuracy1+test_accuracy2
    print("The eventual test accuracy :{}".format((test_accuracy)/2))

