import numpy as np
import matplotlib.pyplot as plt
import array
import time
import random
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

numX=252
numY=252
numStep=0.2
numPMT=20
numBatch=30000000
numTest=50
numSave=10000
numPerTrainBatch=10000
numPerTestBatch=40000

iTrain=np.arange(0,numPerTrainBatch)
iTest=np.arange(0,numPerTestBatch)

f=open('E:\\research\\data\\20170802\\MDRFSimulated_252X252_NumPhoton200000_NumSiPMs20_CsI_Barrier1_QE0.2_lambertianRef0.9_gelIndex1.500_WholeLargeRodSpace1_.bin','rb')
a=array.array("f")
a.fromfile(f, numX*numY*numPMT)
b=np.array(a)
b=np.reshape(b,[numX,numY,numPMT])
norm=np.sum(b,2)
norm.shape=[b.shape[0],b.shape[1],1]
b/=norm

totalNumPhotons=16352
QE=0.2
bSample=b*totalNumPhotons*QE
PMTSignals=np.zeros(numPMT)

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

#PMTSignals = np.zeros([numStep,numPMT])

###############################################################

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, numPMT])
y = tf.placeholder(tf.float32, shape=[None, 12])
adamStep = tf.placeholder(tf.float32)

W_12 = weight_variable([20, 40])
b_2 = bias_variable([40])
h2=tf.nn.relu(tf.matmul(x, W_12)+b_2)
h2 = tf.nn.dropout(h2, keep_prob)


W_23 = weight_variable([40, 60])
b_3 = bias_variable([60])
h3=tf.nn.relu(tf.matmul(h2, W_23)+b_3)
h3 = tf.nn.dropout(h3, keep_prob)


##W_34 = weight_variable([120, 252])
##b_4 = bias_variable([252])
##h4=tf.nn.relu(tf.matmul(h3, W_34)+b_4)
##h4 = tf.nn.dropout(h4, keep_prob)


##W_45 = weight_variable([20, 2])
##b_5 = bias_variable([2])
##h5=tf.nn.relu(tf.matmul(h4, W_45)+b_5)
##h5 = tf.nn.dropout(h5, keep_prob)


W_56 = weight_variable([60, 12])
b_6 = bias_variable([12])
ye=tf.matmul(h3, W_56)+b_6
ye = tf.nn.dropout(ye, keep_prob)


adamStepLocal = 4.970451067745531e-010 #0.0000001 #0.00000001
#loss = np.sum(np.square(ye-y))
#yex=tf.log(tf.nn.softmax(ye))
#loss=tf.reduce_mean(tf.reduce_sum(tf.multiply(-yex,y),1))
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=ye))
#loss = tf.losses.mean_squared_error(labels = y, predictions = ye)
#train_step = tf.train.GradientDescentOptimizer(adamStep).minimize(loss)
train_step=tf.train.AdamOptimizer(learning_rate = tf.Variable(0.00001)).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ye, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
###############################################################

PMTSignals_train=np.zeros([numPerTrainBatch,numPMT])
PMTSignals_test=np.zeros([numPerTestBatch,numPMT])
labels_train=np.zeros([numPerTrainBatch,12])
labels_test=np.zeros([numPerTestBatch,12])

sess = tf.Session()
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

percentage = 0

###################################
saver=tf.train.Saver()
#saver.restore(sess, "E:\\research\\data\\20180314_cnn\model_temp\\mod.ckpt")
#print("Model restored.")
###################################

distLast=0.01

for idxBatch in range(0,numBatch):
    #PMTSignals_train.shape=[numPerTrainBatch,numPMT]
##    r_x=np.random.random_sample(numPerTrainBatch)*(numX-1)
##    r_y=np.random.random_sample(numPerTrainBatch)*(numY-1)
##    idxf=np.floor(r_x).astype(int)
##    idxc=np.ceil(r_x).astype(int)
##    idyf=np.floor(r_y).astype(int)
##    idyc=np.ceil(r_y).astype(int)
##    vec_train=np.transpose(np.transpose(bSample[idxf,idyf])*((idxc-r_x)*(idyc-r_y))+np.transpose(bSample[idxf,idyc])*((idxc-r_x)*(r_y-idyf))+np.transpose(bSample[idxc,idyf])*((r_x-idxf)*(idyc-r_y))+np.transpose(bSample[idxc,idyc])*((r_x-idxf)*(r_y-idyf)))
    labels_train=np.zeros([numPerTrainBatch,12])
    idx=np.random.randint(numX,size=[numPerTrainBatch,])
    idy=np.random.randint(numY,size=[numPerTrainBatch,])
    PMTSignals_train=b[idx,idy]
    labels_train[iTrain, idy//21]=1
    
    train_step.run(feed_dict={x: PMTSignals_train, y: labels_train, keep_prob: 0.85, adamStep: adamStepLocal}, session = sess)

    if (idxBatch!=0 and idxBatch%(numTest)==0):
        labels_test=np.zeros([numPerTestBatch,12])
        idx_test=np.random.randint(numX,size=[numPerTestBatch,])
        idy_test=np.random.randint(numY,size=[numPerTestBatch,])
        PMTSignals_test[:]=b[idx_test, idy_test]
        labels_test[iTest, idy_test//21]=1
        
        dist=accuracy.eval(feed_dict={x: PMTSignals_test, y: labels_test, keep_prob: 1.0}, session=sess)
        
        percentage = np.abs(dist-distLast)/distLast
        distLast=dist
        
        if(percentage>0.0005):
            adamStepLocal =adamStepLocal*0.5
        elif(percentage<0.0001):
            adamStepLocal =adamStepLocal*1.5
        print('step %d, dist %f, percentage %f' % (idxBatch, dist, percentage))
        
    if((idxBatch!=0)&(idxBatch%(numSave)==0)):
        print("saved\n")
        saver.save(sess, "E:\\research\\data\\20180314_cnn\model_temp\\mod.ckpt")
