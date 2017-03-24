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
  # Dependencies for this tool
      1.PyQt4
      2.designer-qt4 (if you have anaconda installed you will by default have PyQt and designer-qt)
       The designer-qt can be found (path_to_anaconda\Library\bin\designer-qt4.exe)  
      3. Keras-1.2.2 
     
      Note: You can use any backend - theano or tensorflow. for theano the input shape shoud be (1,3,224,224) 
      and for tensorflow the input shape should be (1,224,224,3)

