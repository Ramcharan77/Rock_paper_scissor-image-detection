
# import the libraries as shown below

from keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from keras.models import Model
from keras.applications.vgg19 import VGG19
from glob import glob
import matplotlib.pyplot as plt
import scipy

# re-size all the images to this
IMAGE_SIZE = [300, 300]

train_path = 'Dataset/train'
valid_path = 'Dataset/test'


# Import the Vgg 19 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

vgg19 = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# don't train existing weights
for layer in vgg19.layers:
    layer.trainable = False



# useful for getting number of output classes
folders = glob('Dataset/train/*')


folders


# our layers - you can add more if you want
x = Flatten()(vgg19.output)

prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg19.input, outputs=prediction)


model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# Use the Image Data Generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Dataset/train',
                                                 target_size = (300, 300),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


training_set


test_set = test_datagen.flow_from_directory('Dataset/test',
                                            target_size = (300, 300),
                                            batch_size = 32,
                                            class_mode = 'categorical')


from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss",patience=5, verbose=True)


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=4,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=[early_stop],verbose=True
)

# save it as a h5 file
from keras.models import load_model
model.save('model_vgg19.h5')

y_pred = model.predict(test_set)

y_pred