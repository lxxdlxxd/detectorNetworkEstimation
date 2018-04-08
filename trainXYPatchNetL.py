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
y = tf.placeholder(tf.float32, shape=[None, 45])
###############################################################

##start array of neural networks

###############################################################
for i in range(0,12):
    for j in range(0,12):
        temp1='WL'+'_x'+np.str(i)+'_y'+np.str(j)+'_12'
        vars()[temp1]=weight_variable([numPMT, 60])
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
###############################################################
PMTSignals_train=np.zeros([numPerTrainBatch,numPMT])
PMTSignals_test=np.zeros([numPerTestBatch,numPMT])
labels_train=np.zeros([numPerTrainBatch,45])
labels_test=np.zeros([numPerTestBatch,45])

sess = tf.Session()
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

percentage = 0

###################################
saver=tf.train.Saver()
saver.restore(sess, "E:\\research\\data\\20180314_cnn\model_LX\\mod.ckpt")
print("Model restored.")
###################################

accuracyLast=0.01

for idxBatch in range(0,numBatch):
    accuracySum=0
    for i in range(0,12):
        for j in range(0,12):
            labels_train=np.zeros([numPerTrainBatch,45])
            rx=np.random.randint(45,size=[numPerTrainBatch,])
            if i == 0:            
                idx=rx
            elif i==11:
                idx=rx+numX-45
            else:
                idx=rx+21*i-12
                
            ry=np.random.randint(45,size=[numPerTrainBatch,])
            if j == 0:            
                idy=ry
            elif j==11:
                idy=ry+numY-45
            else:
                idy=ry+21*j-12
                
            PMTSignals_train=b[idx,idy]
            labels_train[iTrain, rx]=1
            temp='train_step_L'+'_x'+np.str(i)+'_y'+np.str(j)
            vars()[temp].run(feed_dict={x: PMTSignals_train, y: labels_train, keep_prob: 0.85}, session = sess)

            if (idxBatch!=0 and idxBatch%(numTest)==0):
                labels_test=np.zeros([numPerTestBatch,45])
                rx=np.random.randint(45,size=[numPerTestBatch,])
                if i == 0:            
                    idx_test=rx
                elif i==11:
                    idx_test=rx+numX-45
                else:
                    idx_test=rx+21*i-12

                ry=np.random.randint(45,size=[numPerTestBatch,])
                if j == 0:            
                    idy_test=ry
                elif j == 11:
                    idy_test=ry+numY-45
                else:
                    idy_test=ry+21*j-12

                PMTSignals_test=b[idx_test, idy_test]
                labels_test[iTest, rx]=1
                temp='accuracy_L'+'_x'+np.str(i)+'_y'+np.str(j)
                accuracy=vars()[temp].eval(feed_dict={x: PMTSignals_test, y: labels_test, keep_prob: 1.0}, session=sess)
                accuracySum+=accuracy

    if (idxBatch!=0 and idxBatch%(numTest)==0):
        accuracy=accuracySum/144          
        percentage = np.abs(accuracy-accuracyLast)/accuracyLast
        accuracyLast=accuracy
            
        print('step %d, dist %f, percentage %f' % (idxBatch, accuracy, percentage))
        
    if((idxBatch!=0)&(idxBatch%(numSave)==0)):
        print("saved\n")
        saver.save(sess, "E:\\research\\data\\20180314_cnn\model_temp\\mod.ckpt")
