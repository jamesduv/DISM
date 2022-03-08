# DISM

Discretization Independent Surrogate Modeling

Introduction
------------
This repository provides supporting code for the manuscript: "Methods for Discretization Independent Surrogate Modeling over Complex Geometries," which may be found at: . Tensorflow implementations for the three methods compared are provided, including:

1. DV-MLP   : Design variable multi-layer perceptron. Simple feed-forward neural network, where pointwise spatial coordinates and design variables are fed through the main network.
2. DV-Hnet  : Design variable hypernetwork. The weights and biases of the main network are generated by a simple feed-forward hypernetwork as a single, large vector. The hypernetwork consumes design variables, the main network consumes pointwise spatial quantities.
3. NIDS     : Non-linear independent dual system. A partial design variable hypernetwork, where only the weights and biases of the main network output layer are generated by a feed-forward hypernetwork.

To do
------------
1. Include python version and minimal enironment file.
2. Include data?
3. Include example training code.
   

Contents
----------------

The repo should contain the following files:  

-----------------------------------
    DISM
    ├── dense_networks.py        
    ├── keras_hypernetworks.py      
    ├── nids_keras_networks.py   
    ├── tf_common.py 
    └── README.md
-----------------------------------

Test


**Functionality**

File | Description 
--- | ---|
dense_networks.py | class definition for dense neural network, constructed using Keras sequential API. Used as standalone model for DV-MLP, used as subcomponents for DV-Hnet and NIDS.
keras_hypernetworks.py | class definition for DV-Hnet, subclasses ```tf.keras.Model.```
nids_keras_networks.py | class definition for NIDS, subclasses ```tf.keras.Model.```
tf_common.py | common tensorflow functions

**Data files**

File | Description
--- | ---|



