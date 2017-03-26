# convolution-layer-visualization-VGG-16

A GUI Tool created on PyQt4 for visualizing the intermediate layer of VGG-16 CNN trained on Imagenet.

This tool can also be used to visualize intermediate layer of any other CNN network just by changing the way the input is fed to the respective network

![image](https://github.com/anujshah1003/convolution-layer-visualization-VGG-16/tree/master/images/VGG-16-gui.PNG)

Note that for this tool the network model along with its weight is saved in single hdf5 file. it can be in h5 format also.
whatever extension the load_model of keras reads:

    from keras.models import load_model
    model = load_model('network_name.hdf5')  
               (or)
    model = load_model('network_name.h5')
    
  you can get the VGG-16 imagenet model for the keras github https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py
  
  However in order to save the model in hdf5 or h5 format call the VGG-16 model in vgg16.py using
  
      model = VGG16(include_top=True, weights='imagenet')
      model.save(VGG16.hdf5)
                (or)
      model.save(vgg16.h5)
  # Dependencies for this tool
      1.PyQt4
      2.designer-qt4 (if you have anaconda installed you will by default have PyQt and designer-qt)
       The designer-qt can be found (path_to_anaconda\Library\bin\designer-qt4.exe)  
      3. Keras-1.2.2 
     
      Note: You can use any backend - theano or tensorflow. for theano the input shape shoud be (1,3,224,224) 
      and for tensorflow the input shape should be (1,224,224,3)
      
Not only pre-trained imagenet you can use this tool for visualizing intermediate layer of your own models too, jusr by changing the way you feed the input in browseInputImage() function in VGG-16-visualization/vgg16_intermediate_layer_visualization_gui.py

# Running the Tool
 Run the vgg16_intermediate_layer_visualization_gui.py in VGG-16-Visualization directory
