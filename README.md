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
