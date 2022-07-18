# Object Detection Cereal Boxes
Cereal Model Object Detection

This code uses TensorFlow lite and Machine Learning model to detect Cereal Boxes and outputs the probabilities of what Cereal box it is.

The App takes the images as bitmap from your device's back camera, process it using the cereal_model.tflite model and outputs the probabilities.
You can see the camera preview in the activity and can see the probabilities in the bottom sheet when you swipe up.


Here is the example code to show this.

val model = CerealModel.newInstance(context)
[Create the model]


val image = TensorImage.fromBitmap(bitmap)
[Creates inputs for reference]


val outputs = model.process(image)

val probability = outputs.probabilityAsCategoryList
[Runs model inference and gets result]


model.close()
[Releases model resources if no longer used]

Next Step - 
Train the model to further output the image location, height and width to create bounding box around the image we are detecting.

References - 

TensorFlow Lite

https://www.tensorflow.org/lite/android/tutorials/object_detection

Using ML model with TensorFlow lite

https://towardsdatascience.com/machine-learning-with-android-11-whats-new-e8c829e9452
