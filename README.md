# Image Captioning with LSTM

Image Captioning model generates natural captions for input images

## Dataset
The model is trained on [Flickr8k Dataset](https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b) which contains 8000 images. 6000 for training, 1000 for testing and validating.

Other dataset which larger observation like Flickr30k can also be used to train the model

After downloading, there are 2 folder **Flicker8k_Dataset** and **Flicker8k_Text**:
- Flicker8k_Dataset contains images with different id as its name
- Flicker8k_Text contains:
  - Flickr_8k.testImages, Flickr_8k.devImages, Flickr_8k.trainImages, Flickr_8k.devImages consist of image id for test, train and validation.
  - Flickr8k.token contain 5 different caption for each image in the *Flicker8k_Dataset*

Image | Caption
--- | ---
<img src="images_test/girl.png" width="300"> | - A child in a pink dress is climbing up a set of stairs in an entry way.<br>- A girl going into a wooden building .<br>- A little girl climbing into a wooden playhouse .<br>- A little girl climbing the stairs to her playhouse .<br>- A little girl in a pink dress going into a wooden cabin 

## Requirements
- tensorflow
- nltk
- numpy
- matplotlib
- pandas

These requirements can be easily installed by: `pip install -r requirements.txt`

## Model
<div align="center">
  <img src="model.png"><br><br>
</div>

## Performance

The model has been trained for 10 epoches on 6000 training samples of Flickr8k Dataset. It acheives a `BLEU-1 = ~0.56` with 1000 testing samples with both **Greedy Search** and **Beam Search**

----------------------------------

## Architecture

**Input** of our model is the *image* and **Output** is a text correspoding to provided image
 
 As we have input is image, we can think about using CNN to extract the features from it, along with output is text, we immediately think about RNN which handles sequences and used Long Short-Term Memory in this case.

 For the training, for each 1 image, we have 5 captions combined together. So we have to pre-process these 2 input seperately before fit into our LSTM model. 
 
 With the captions, in general, most of Machine Learning or Deep Learning model does not handly text input like `'man', 'hawk', 'woman'..` directly and have to encode into number form. Each word will be encoded into a vector with fixed length (also call *word embedding*). For this project, I use Pre-trained GLOVE Model to vectorize words, each vector has shape of `(1,200)`. Then we fit word vector into **RNN/LSTM** model to handle sequential data and predict which word is next in the sequence. The output vector of model is `(1,256)`
 
 With the images, similarly with text, we also use a pre-trained model with larget datasset (Imagenet) to extract features from images into a *featuring vector*. There are a lot of pre-trained model outhere likes: ResNet, VGG16, Inception,.. In this model, I chose to use the InceptionV3. As InceptionV3 requires input as shape `(299,299)`, we need to resize our image into that. The output vector is `(1,256)`
 
 ### Text processing
 
 Before vectorizing words into vector, we need to clean the captions with following steps:
 - Convert uppercase to lowercase, "Hello" -> "hello"
 - Remove special characters like "%", "$", "#"
 - Remove alphanumeric characters like hey199 

Afterthat, we add 2 token `"startseq"` and `"endseq"` to denote the start and end of the caption. For example: “startseq a girl going into a wooden building endseq“. The *endseq* is used to know whether it is the end of the caption while testing.

We see there are around 9000 different words out of 40000 captions. However, we don't care much for words that appear only a few times, because it looks like noise and is not good for our model's learning and prediction, so we keep only the words that appear more than **10 times**. among all the captions. After removing the words that appear less than 10 times, we are left with 1651 words. 

### Model Architecture

As the image of model above, the left hand side is the input for Text, and the right hand side is for images. After preprocessing captions, embedded words to vector, extract features from images. We concatnate these 2 inputs 
