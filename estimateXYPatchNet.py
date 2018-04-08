import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import array
import time
import random
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

numX=252
numY=252
RES=0.2
numBatchSize=50000
D_input = 20
D_label=2

f=open('E:\\research\\data\\20180314_cnn\\5lineSimulated.bin','rb')
a=array.array("f")
a.fromfile(f, int(74420000/4))
c=np.array(a)
line_inputs=np.reshape(c,[int(c.size/D_input), D_input])
norm=np.sum(line_inputs, 1)
norm.shape=[line_inputs.shape[0],1]
line_inputs/=norm
print(line_inputs.shape)

def normalize(PMTSignals):
    temp=PMTSignals
    m=PMTSignals.shape[0]
    n=PMTSignals.shape[1]
    total=np.sum(PMTSignals, axis=1)
    total.shape=(m, 1)
    PMTSignals[:,:]=PMTSignals/total 
    return

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, D_input])
y = tf.placeholder(tf.float32, shape=[None, 45])
##############################################################################################################

W_12 = weight_variable([D_input, 30])
b_2 = bias_variable([30])
h2=tf.nn.relu(tf.matmul(x, W_12)+b_2)
h2 = tf.nn.dropout(h2, keep_prob)

W_56 = weight_variable([30, 12])
b_6 = bias_variable([12])
ye=tf.matmul(h2, W_56)+b_6
ye = tf.nn.dropout(ye, keep_prob)

##############################################################################################################

sess = tf.Session()
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
idx=np.zeros([line_inputs.shape[0]], dtype=int)
saver=tf.train.Saver()

for i in range(0,12):
    for j in range(0,12):
        temp='group_x'+np.str(i)+'_y'+np.str(j)
        vars()[temp]=[]

########################

saver.restore(sess, "E:\\research\\data\\20180314_cnn\model_GX\\mod.ckpt")
print("Model GX restored.")

numBatch=line_inputs.shape[0]//numBatchSize

idx=np.zeros([line_inputs.shape[0]], dtype=int)
for i in range(0, numBatch):
    dataFeed=line_inputs[numBatchSize*i:numBatchSize*(i+1),:]
    output=ye.eval(feed_dict={x: dataFeed, keep_prob: 1.0}, session=sess)
    idx[numBatchSize*i:numBatchSize*(i+1)]=np.argmax(output,1)

dataFeed=line_inputs[numBatchSize*numBatch:]
output=ye.eval(feed_dict={x: dataFeed, keep_prob: 1.0}, session=sess)
idx[numBatchSize*numBatch:]=np.argmax(output,1)

########################

saver.restore(sess, "E:\\research\\data\\20180314_cnn\model_GY\\mod.ckpt")
print("Model GY restored.")

numBatch=line_inputs.shape[0]//numBatchSize

idy=np.zeros([line_inputs.shape[0]], dtype=int)
for i in range(0, numBatch):
    dataFeed=line_inputs[numBatchSize*i:numBatchSize*(i+1),:]
    output=ye.eval(feed_dict={x: dataFeed, keep_prob: 1.0}, session=sess)
    idy[numBatchSize*i:numBatchSize*(i+1)]=np.argmax(output,1)

dataFeed=line_inputs[numBatchSize*numBatch:]
output=ye.eval(feed_dict={x: dataFeed, keep_prob: 1.0}, session=sess)
idy[numBatchSize*numBatch:]=np.argmax(output,1)

########################

for i in range(0,line_inputs.shape[0]):
    temp='group_x'+np.str(idx[i])+'_y'+np.str(idy[i])
    vars()[temp].append(line_inputs[i,:])

for i in range(0,12):
    for j in range(0,12):
        temp='group_x'+np.str(i)+'_y'+np.str(j)
        vars()[temp]=np.array(vars()[temp])

##############################################################################################################

tf.reset_default_graph()

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, D_input])
y = tf.placeholder(tf.float32, shape=[None, 45])

for i in range(0,12):
    for j in range(0,12):
        temp1='WL'+'_x'+np.str(i)+'_y'+np.str(j)+'_12'
        vars()[temp1]=weight_variable([D_input, 60])
        temp2='bL'+'_x'+np.str(i)+'_y'+np.str(j)+'_2'
        vars()[temp2]=bias_variable([60])
        temp3='hL'+'_x'+np.str(i)+'_y'+np.str(j)+'_2'
        vars()[temp3]=tf.nn.relu(tf.matmul(x, vars()[temp1])+vars()[temp2])
        vars()[temp3]=tf.nn.dropout(vars()[temp3], keep_prob)


        temp4='WL'+'_x'+np.str(i)+'_y'+np.str(j)+'_23'
        vars()[temp4]=weight_variable([60, 60])
        temp5='bL'+'_x'+np.str(i)+'_y'+np.str(j)+'_3'
        vars()[temp5]=bias_variable([60])
        temp6='hL'+'_x'+np.str(i)+'_y'+np.str(j)+'_3'
        vars()[temp6]=tf.nn.relu(tf.matmul(vars()[temp3], vars()[temp4])+vars()[temp5])
        vars()[temp6]=tf.nn.dropout(vars()[temp6], keep_prob)


        temp7='WL'+'_x'+np.str(i)+'_y'+np.str(j)+'_34'
        vars()[temp7]=weight_variable([60, 45])
        temp8='bL'+'_x'+np.str(i)+'_y'+np.str(j)+'_4'
        vars()[temp8]=bias_variable([45])
        temp9='hL'+'_x'+np.str(i)+'_y'+np.str(j)+'_4'
        vars()[temp9]=tf.matmul(vars()[temp6], vars()[temp7])+vars()[temp8]
        vars()[temp9]=tf.nn.dropout(vars()[temp9], keep_prob)

        temp10='loss_L'+'_x'+np.str(i)+'_y'+np.str(j)
        vars()[temp10] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=vars()[temp9]))
        temp11='train_step_L'+'_x'+np.str(i)+'_y'+np.str(j)
        vars()[temp11]=tf.train.AdamOptimizer(learning_rate = tf.Variable(0.0001)).minimize(vars()[temp10])
        temp12='correct_prediction_L'+'_x'+np.str(i)+'_y'+np.str(j)
        vars()[temp12] = tf.equal(tf.argmax(y, 1), tf.argmax(vars()[temp9], 1))
        temp13='accuracy_L'+'_x'+np.str(i)+'_y'+np.str(j)
        vars()[temp13] = tf.reduce_mean(tf.cast(vars()[temp12], tf.float32))

sess = tf.Session()
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "E:\\research\\data\\20180314_cnn\model_LX2\\mod.ckpt")
print("Model LX restored.")

########################

idx=np.zeros([line_inputs.shape[0]], dtype=int)

off_setG=0

for i in range(0,12):
    if i == 0:            
        off_set=0
    elif i == 11:
        off_set=numX-45
    else:
        off_set=21*i-12
    for j in range(0,12):
        temp='group_x'+np.str(i)+'_y'+np.str(j)
        dataSeg=vars()[temp]
        if(dataSeg.shape[0]>0):
            temp='hL'+'_x'+np.str(i)+'_y'+np.str(j)+'_4'
            
            numBatch=dataSeg.shape[0]//numBatchSize
            residual=dataSeg.shape[0]%numBatchSize
            for k in range(0, numBatch):
                dataFeed=dataSeg[numBatchSize*k:numBatchSize*(k+1),:]
                output=vars()[temp].eval(feed_dict={x: dataFeed, keep_prob: 1.0}, session=sess)
                idx[off_setG+numBatchSize*k:off_setG+numBatchSize*(k+1)]=off_set+np.argmax(output,1)

            dataFeed=dataSeg[numBatchSize*numBatch:]
            output=vars()[temp].eval(feed_dict={x: dataFeed, keep_prob: 1.0}, session=sess)
            idx[off_setG+numBatchSize*numBatch:off_setG+numBatchSize*numBatch+residual]=off_set+np.argmax(output,1)

        off_setG+=dataSeg.shape[0]

########################

saver.restore(sess, "E:\\research\\data\\20180314_cnn\model_LY2\\mod.ckpt")
print("Model LY restored.")

idy=np.zeros([line_inputs.shape[0]], dtype=int)

off_setG=0

for i in range(0,12):
    for j in range(0,12):
        if j == 0:            
            off_set=0
        elif j == 11:
            off_set=numY-45
        else:
            off_set=21*j-12
        temp='group_x'+np.str(i)+'_y'+np.str(j)
        dataSeg=vars()[temp]
        if(dataSeg.shape[0]>0):
            temp='hL'+'_x'+np.str(i)+'_y'+np.str(j)+'_4'
            
            numBatch=dataSeg.shape[0]//numBatchSize
            residual=dataSeg.shape[0]%numBatchSize
            for k in range(0, numBatch):
                dataFeed=dataSeg[numBatchSize*k:numBatchSize*(k+1),:]
                output=vars()[temp].eval(feed_dict={x: dataFeed, keep_prob: 1.0}, session=sess)
                idy[off_setG+numBatchSize*k:off_setG+numBatchSize*(k+1)]=off_set+np.argmax(output,1)

            dataFeed=dataSeg[numBatchSize*numBatch:]
            output=vars()[temp].eval(feed_dict={x: dataFeed, keep_prob: 1.0}, session=sess)
            idy[off_setG+numBatchSize*numBatch:off_setG+numBatchSize*numBatch+residual]=off_set+np.argmax(output,1)

        off_setG+=dataSeg.shape[0]

##############################################################################################################

image=np.zeros([numX, numY])

for i in range(0,line_inputs.shape[0]):
    image[idx[i],idy[i]]+=1

f,a=plt.subplots(1)

a.imshow(image, cmap='gray_r', extent=[0, numX*RES, 0, numY*RES])
f.show()
