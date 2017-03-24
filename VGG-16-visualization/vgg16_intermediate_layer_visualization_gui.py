import sys
from PyQt4 import QtCore, QtGui, uic
import ctypes

qtCreatorFile = "vgg16_visualization_layout.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

import cv2,ctypes
import time
import numpy as np
import winsound
import matplotlib.pyplot as plt
from keras import backend as K


from keras.models import load_model
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from keras.preprocessing import image as image_utils

class MyApp(QtGui.QMainWindow, Ui_MainWindow):
	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		Ui_MainWindow.__init__(self)
		self.setupUi(self)
		self.loadModel.clicked.connect(self.browseModel)
		self.modelSummary.clicked.connect(self.modSummary)
		self.layerButton.clicked.connect(self.layerConfiguration)
		self.inputImage.clicked.connect(self.browseInputImage)
		self.computeOutput.clicked.connect(self.predictOutput)
		self.outputFeatureMaps.clicked.connect(self.featureMaps)
		self.specificFeatureMap.clicked.connect(self.individualFeatureMap)

		self.model=None
		self.fname = None
		self.layer = None
		self.image=None
		
	def browseModel(self):
		self.model=None
		self.fname = None
		self.layer = None
		self.filePath = QtGui.QFileDialog.getOpenFileName(self,'*.')
#		print('filepath : ', self.filePath)
		self.fname = str(self.filePath)
		print self.fname
		print type(self.fname)
		
		self.console_output2.setText(self.fname)
		
		print '-----------Loading the Model-----------------------'
		self.model=load_model(self.fname)
		print '-----------Loaded successfully---------------------'
		ctypes.windll.user32.MessageBoxA(0, "Model Loaded Successfuly", "Message", 1)
#		self.model.summary()
		from keras.utils.visualize_util import plot
		graph = plot(self.model,to_file='model-summary.png', show_shapes=True)
	def modSummary(self):
		print'-------------model summary--------------------------'
		self.model.summary()
		#model_img = QtGui.QPixmap('model-summary.png')
		#self.displayOutput.setScaledContents(True)
		#self.displayOutput.setPixmap(model_img)
		img=cv2.imread('model-summary.png')
		cv2.imshow('model',img)
		cv2.waitKey(0)
		
	def layerConfiguration(self):
		self.console_output2.setText(' ')

		self.layer = self.layerNumber.text()
		#print int(self.layer)
		#self.model.layers[int(self.layer)].get_config()
		
		def get_featuremaps(model, layer_idx, X_batch):
			get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
			activations = get_activations([X_batch,0])
			return activations
				

		self.activations = get_featuremaps(self.model, int(self.layer), self.image)
		#plt.imshow(self.activations[0][0][63],cmap = 'gray')
		output_shape = np.shape(self.activations[0])
		featuremap_size = np.shape(self.activations[0][0][0])
		num_of_featuremaps = (np.shape(self.activations[0][0]))[0]
		print num_of_featuremaps
		layer_info=self.model.layers[int(self.layer)].get_config()
		layer_name=layer_info['name']
		input_shape=self.model.layers[int(self.layer)].input_shape
		layer_param=self.model.layers[int(self.layer)].count_params()
		output_text = ('Layer Name : ' + layer_name + '\n'  
							+ 'Layer Number : ' + self.layer + '\n'
							+ 'Input Shape : ' + str(input_shape) + '\n'
							+ 'output shape :' + str(output_shape)+ '\n'
							+ 'num of feature maps :' + str(num_of_featuremaps)
							+ '\n'+ 'size of feature maps :'+ str(featuremap_size)+ '\n'
							+ 'Num of Parameters :' + str(layer_param)+'\n')
		self.console_output2.setText( output_text)

#%%     
		if len(output_shape)==2:
			fig=plt.figure(figsize=(16,16))
			plt.imshow(self.activations[0].T,cmap='gray')
			plt.savefig("featuremaps-layer-{}".format(self.layer) + '.png')

		else:
			fig=plt.figure(figsize=(16,16))
			subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
			for i in range(int(num_of_featuremaps)):
				ax = fig.add_subplot(subplot_num, subplot_num, i+1)
				#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
				ax.imshow(self.activations[0][0][i],cmap='gray')
				plt.xticks(np.array([]))
				plt.yticks(np.array([]))
				plt.tight_layout()
			plt
			fig.savefig("featuremaps-layer-{}".format(self.layer) + '.png')
		
		
	def browseInputImage(self):
		self.imageName = None
		self.image=None
		self.console_output2.setText('')
		self.imagePath = QtGui.QFileDialog.getOpenFileName(self,'*.')
#		print('filepath : ', self.filePath)
		self.imageName = str(self.imagePath)
		print self.imageName
		print type(self.imageName)
		image = QtGui.QPixmap(self.imagePath)
		#self.displayInput.setScaledContents(True)
		self.displayInput.setPixmap(image)
		
		self.image = image_utils.load_img(self.imageName, target_size=(224, 224))
		self.image = image_utils.img_to_array(self.image)
		
    # our image is now represented by a NumPy array of shape (3, 224, 224),
    # but we need to expand the dimensions to be (1, 3, 224, 224) so we can
    # pass it through the network -- we'll also preprocess the image by
    # subtracting the mean RGB pixel intensity from the ImageNet dataset
		self.image = np.expand_dims(self.image, axis=0)
		self.image = preprocess_input(self.image)
		image_preprocessed=np.rollaxis(self.image,1,4)
		#cv2.imshow('img',image_preprocessed[0,:,:,:])
		#cv2.waitKey(0)
		cv2.imwrite('image_preprocessed.jpg',image_preprocessed[0,:,:,:])
		img_preprocessed = QtGui.QPixmap('image_preprocessed.png')
		#self.displayPreprocessedInput.setScaledContents(True)
		self.displayPreprocessedInput.setPixmap(img_preprocessed)
		
	def predictOutput(self):	
		orig = cv2.imread(self.imageName)
		# classify the image
		print("[INFO] classifying image...")
		preds = self.model.predict(self.image)
		preds_class = self.model.predict_classes(self.image)
		#top4class_index = [i for i in np.argsort(preds[0])[-4:]]
		#top4class_prob = [preds[0][i] for i in np.argsort(preds[0])[-4:]]
		#print top4class_index
		#print top4class_prob
		
		preds_prob=preds[0][preds_class]
		(inID, label) = decode_predictions(preds)[0]
		
		# display the predictions to our screen
		output_text = ('Class Label :{}'.format(preds_class) + '\n'
		             + 'class Prob : {}'.format(preds_prob) + '\n'
					 + 'Class Name : {}'.format(label) + '\n'
					 + 'Imagenet Id : {}'.format(inID))
		self.console_output.setText(output_text)
		
		
		print("ImageNet ID: {}, Label: {}".format(inID, label))
		#cv2.putText(orig, "Label: {}".format(label), (10, 30),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

		#cv2.imshow("Classification", orig)
		#cv2.waitKey(0)	
 
	def featureMaps(self):
		output_img = QtGui.QPixmap("featuremaps-layer-{}".format(self.layer) + '.png')
		self.displayOutput.setScaledContents(True)
		self.displayOutput.setPixmap(output_img)
		#output_img = cv2.imread("featuremaps-layer-{}".format(self.layer) + '.png')
		
	def individualFeatureMap(self):
		self.map_num = self.featureMapNumber.text()
		fig=plt.figure(figsize=(16,16))
		plt.imshow(self.activations[0][0][int(self.map_num)],cmap = 'gray')
		plt.savefig("featuremaps-layer-{}-".format(self.layer) + 'map_num-{}'.format(self.map_num)+ '.png')
		featuremap_img = QtGui.QPixmap("featuremaps-layer-{}-".format(self.layer) + 'map_num-{}'.format(self.map_num)+ '.png')
		self.displayOutput.setScaledContents(True)
		self.displayOutput.setPixmap(featuremap_img)
if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())