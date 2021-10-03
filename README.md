# Fire Detection

**Abstract**

We propose a method based on a deep learning approach using custom CNN models that's been implemented for fire detection. The dataset for training the model is procured from Kaggle and GitHub and consists of more than 1000 images which are pre classified as fire and not fire. This data will be used to train the models. Different models will be used to compare the results and the best one is selected. One of these models is a basic CNN architecture which contains three convolutional and max pooling layers with rectified linear unit as the activation function and two hidden dense layers, these layers are tweaked and the number of neurons in each layer is tweaked and any performance difference is noted. The second type of models are customized InceptionV3 model which consists of global average pooling convolution layers and dense layers on top of InceptionV3. To balance the efficiency and accuracy, the model is fine-tuned considering the nature of the target problem and fire data. We are going to use three different datasets for training our models.

### Data preprocessing and augmentation

Different models – in-house CNN, Inception V3 will be trained on our image dataset that we procured from Kaggle and GitHub. Initially, the procured data will be preprocessed before it is given as an input to our neural networks. The procured data will first be split into three sets – the training set, the validation set and the testing set. A total of 1650 images which are labelled as &quot;fire&quot; and &quot;notFire&quot; are taken into consideration.

The training set is further expanded with the help of ImageDataGenerator class from the Keras API that comes along with TensorFlow. The images are randomly flipped and rotated and are then added back to the training set as a completely new image. All of the image data which lie in the range of 0 to 255 are normalized so that they fit within the range of 0 and 1. This will allow us to get better results from our neural networks. All the images are resized to a newer constant size since different inputs with varying sizes can&#39;t be given to a neural network. The images are resized to (224, 224) pixels.

### Model Building

Different models – in-house CNN, Inception V3 will be trained on the available image data that is preprocessed. The models are built with the help of TensorFlow library.

### CNN

Three convolutional and max pooling layers with rectified linear unit as the activation function are used as the convolutional layers where the convolutional layers will perform feature extraction and the max pooling layers will help in decreasing the memory usage of the model and also help in avoiding overfitting the training data. Using ReLU as the activation function helps in making the data more non-linear. The data that comes out of the convolutional layers are flattened after which it is fed to the Dense layers. There are two hidden Dense Layers and one output layer. The Dense layers consists of 2048 and 1024 neurons respectively and have a dropout function which will drop the neurons and helps in the prevention of overfitting the training data. The dropout function for the input layer and the two hidden layers is 0.2, 0.25 and 0.2 respectively. The function will randomly drop 20%, 25% and 20% of the neurons of their preceding layers respectively. The hidden layers have ReLU as the activation function and the output layer has the softmax activation function. The loss function is categorical crossentropy. The optimizer for the loss function is the Adam optimizer with a learning rate of 0.0001. 

### Inception v3

The model has inceptionv3 architecture as the base model and has a global average pooling layer for the output from the base model, there are dense hidden layers with 2048 and 1024 neurons respectively with ReLU as their activation functions. The output layer consists of two neurons and has softmax as it&#39;s activation function. The two hidden layers are succeeded by a dropout function with the values 0.25 and 0.2. The optimizer for this model is rmsprop and the loss function is categorical crossentropy.

### Video capturing

This model uses the same architecture as used in the previous inception v3 model which will load the weights that we previously trained. The video fed is then captured as frames, then the captured frame is converted to RGB image format. Then the image is resized into 224x224.

### Comparative Study

In the CNN model, the model is performing better when the number of neurons increase. It&#39;s important to increase the dropout function value along with the increase in neurons since there is a higher chance of the neurons overfitting with the training data and giving poor results with the testing data. The CNN has given the best accuracy of 93.5% in the experiments performed, it was trained for 50 epochs.

### Results

The models have given us very promising results with over 96% accuracy and the model is detecting fire, the screenshots of which is given in the document. It does so with considerable accuracy wherein it doesn&#39;t detect small fires like the ones caused by a dia or a stove top but it definitely detects fire with higher intensity and volume. 

### References

https://khan-muhammad.github.io/public/papers/Neurocomputing\_Fire.pdf

https://www.ijstr.org/final-print/apr2020/Fire-Detection-Using-Cnn-Approach.pdf