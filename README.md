# Dog-Breed-identification
End_to_end Multi_class Dog Breed Classification
This notebook built an end_to_end multi_class image classifier using TesorFlow 2.12 and TensorFlow Hub.

1. Problem
Identifying the breed of a dog given an image of a dog.

When I'm sitting at the cafe and I take a photo of a dog, I want to know what breed of dog it is.

2. Data
The data we're using is from Kaggel's dog breed identification competition.

https://www.kaggle.com/c/dog-breed-identification

3. Evaluation
The evaluation is a file with prediction probabilities for each dog breed for each test image.

https://www.kaggle.com/competitions/dog-breed-identification/overview/evaluation

4. Features
Some informaation about the data:

We're dealing with images (unstructured data) so it's
probably best we use deep learning/ transfer learning.

There are 120 breeds of dogs (this means there are 120 different classes).

There are around 10,000+ images in the training set (these images have lables).

There are around 10,000+ images in the test set (these images have no lables, because we'll want to predict them).

# Get our workspace ready
Import TensorFlow 2.x
Import TensorFlow Hub
Make sure we're using a GPU
