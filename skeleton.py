from code2 import *
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Derain(object):

	path = ""

    def __init__(self, data_dir,checkpoint_dir='./checkpoints/'):
        """
            data_directory : path like /home/kushagr/NNFL_Project/rain/training/
            	includes the dataset folder with '/'
            Initialize all your variables here
        """
        with tf.device("/cpu:0"):
			# with open('rainWale.pkl', 'rb') as f:
			# 	loadedImages = pickle.load(f)
			# Ximgs=np.array(loadedImages)

			# with open('binaRainWale.pkl', 'rb') as f:
			# 	loadedImages = pickle.load(f)
			# Yimgs=np.array(loadedImages)

			images = []
			for i in range(1,701):
				s=data_dir + str(i)+'.jpg'
				images.append(plt.imread(s))

			inputImage=[]
			outputImage=[]
			for image in images:
				imgReal=image[:,:int(image.shape[1]/2),:]
				imgFin=image[:,-int(image.shape[1]/2):,:]
				i=cv2.resize(imgReal,(256,256))
				j=cv2.resize(imgFin,(256,256))
				inputImage.append(j)
				outputImage.append(i)

			self.Ximgs = np.array(inputImage)
			self.Yimgs = np.array(outputImage)

    def train(self, training_steps=10):
        """
            Trains the model on data given in path/train.csv
            	which conatins the RGB values of each pixel of the image  

            No return expected
        """
        self.G=Generator()
		self.X=tf.placeholder(tf.float32,shape=[None,256,256,3])
		self.Y=tf.placeholder(tf.float32,shape=[None,256,256,3])
		# X=tf.reshape(Ximgs,[-1,256,256,3])
		self.conv1Out=G.addConvLayer(X,3,64)
		self.conv2Out=G.addConvLayer(conv1Out,64,64)
		self.conv3Out=G.addConvLayer(conv2Out,64,64)
		self.conv4Out=G.addConvLayer(conv3Out,64,64)
		self.conv5Out=G.addConvLayer(conv4Out,64,32)
		self.conv6Out=G.addConvLayer(conv5Out,32,1)
		self.deconv1Out=G.addDeConvLayer(conv6Out,1,32)
		self.deconv2Out=G.addDeConvLayer(deconv1Out,32,64) + conv4Out
		self.deconv3Out=G.addDeConvLayer(deconv2Out,64,64)
		self.deconv4Out=G.addDeConvLayer(deconv3Out,64,64) + conv2Out
		self.deconv5Out=G.addDeConvLayer(deconv4Out,64,64)
		self.GX=G.addDeConvLayer(deconv5Out,64,3) + X

		##########################################################


		#This defines the discriminator, with input as Y(original image without rain)

		self.D=Generator()
		self.Dconv1Out=D.addConvLayer(Y,3,48,strideX=2, strideY=2,filterSize=4,BN=False)
		self.Dconv2Out=D.addConvLayer(Dconv1Out,48,96,strideX=2, strideY=2,filterSize=4)
		self.Dconv3Out=D.addConvLayer(Dconv2Out,96,192,strideX=2, strideY=2,filterSize=4)
		self.Dconv4Out=D.addConvLayer(Dconv3Out,192,384,strideX=1, strideY=1,filterSize=4)
		self.Dconv5Out=D.addConvLayer(Dconv4Out,384,1,strideX=1, strideY=1,filterSize=4,BN=False,PRelu=False)
		self.DY = D.addDeepNet(Dconv5Out)

		Le = tf.reduce_mean(tf.squared_difference(GX,Y))
		La_Y = -tf.reduce_mean(tf.log(DY))
		DX_ = D.forward(GX)
		La_X_ = -tf.reduce_mean(tf.log(1-DX_))
		La = La_Y + La_X_

		# for i in range(100):



		#Le = tf.reduce_mean(tf.squared_difference(GX,Y))
		#La = 

		#finding D of original image
		# D.setInput(Y)
		# Dori = tf.nn.sigmoid(D.forward(Y))
		# Lp = (tf.reduce_mean(tf.squared_difference(DY,Dori)))
		L = 0.0066 * La + Le 

    def save_model(self, step):

        # file_name = params['name']
        # pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk
            You can use pickle or Session.save in TensorFlow
            no return expected
        """


    def load_model(self, **params):
    	# file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-trained instance of Segment class
        """
