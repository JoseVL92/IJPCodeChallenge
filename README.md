# IJPCodeChallenge

## Theoretical Questions
### 1. Explain the difference between object detection and image classification.
**Object Detection** and **Image Classification** differ in their goals and outputs. 
* In **Image Classification** the goal is to assign a single label to an entire image. It outputs a class label that describes the content of the image, answering the question, _"What is in this image?"_
* In **Object Detection** the goal is to identify and locate objects within an image, answering two questions: _"What objects are in this image?"_ and _"Where are they?"_

### 2. What is transfer learning, and how can it be useful in object detection tasks?
**Transfer Learning** is a technique where a model that has been trained on a large dataset is reused for a more specific task. It allows to bring in knowledge it has already learned from solving one problem and applies it to a different but related problem.

Building an object detection model from scratch usually requires massive datasets and lots of computing power. Transfer learning can make this much easier as it:

* **Saves Time**: Reduces training time since the model is already partially trained.
* **Requires Less Data**: You can achieve good results with less training data.
* **Has Better Performance**: The model can start with a good understanding of general object detection, leading to better accuracy in the new task.
* **Gives Resource Efficiency**: Reduces the need for powerful computational resources.


### 3. Describe the architecture of YOLO (You Only Look Once) and its advantages in real-time object detection.
**YOLO** frames object detection as a single regression problem that predicts both bounding boxes and class probabilities directly from the input image in one pass through the network

_Architecture Breakdown:_
1. **Input Image**: YOLO takes an entire image as input (typically resized to 448x448 pixels)
2. **Grid Division**: The image is divided into an SxS grid. Each grid cell is responsible for detecting objects whose center falls within that cell.
3. **CNN Backbone**: The core of YOLO is a CNN that extracts features from the input image. This CNN consists of multiple convolutional and pooling layers. Early versions of YOLO used a custom architecture inspired by GoogLeNet, while later versions use more advanced backbones such as CSPDarknet53.
4. **Bounding Box Prediction**: Each grid cell predicts B bounding boxes, where each bounding box contains 5 parameters: (x, y, width, height, confidence_score).
5. **Class Prediction**: Each grid cell also predicts class probabilities for the object it detects.
6. **Non-Maximum Suppression (NMS)**: After predictions, overlapping boxes for the same object are filtered using Non-Maximum Suppression to keep only the most confident predictions.

### 4. What is a Generative Adversarial Network (GAN), and how could it be used in image manipulation tasks?

A **Generative Adversarial Network (GAN)** is a type of deep learning model that consists of two neural networks: a _generator_ and a _discriminator_. These two networks are trained together in a process called adversarial training.

During training, the generator tries to produce data that is convincing enough to fool the discriminator, while the discriminator is simultaneously trying to improve its ability to tell the real from the fake. This creates a competitive dynamic:
* The generator improves its techniques to create more realistic images
* The discriminator enhances its capability to identify fakes

This back-and-forth continues until the generator produces images that the discriminator can no longer distinguish from real images.

#### Applications of GANs in Image Manipulation Tasks
* **Image Generation**: GANs can create entirely new images from scratch.
* **Image-to-Image Translation**: GANs can transform images from one domain to another.
* **Super Resolution**: GANs can enhance the resolution of images, turning low-resolution images into high-resolution ones while maintaining detail.
* **Image Inpainting**: GANs can fill in missing parts of an image, like repairing damaged photographs
* **Data Augmentation**: In scenarios where data is scarce (like medical imaging), GANs can generate additional training samples to improve the performance of machine learning models.