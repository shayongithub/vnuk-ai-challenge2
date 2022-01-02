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
<img src="images_test/girl.jpg" width="300"> | - A child in a pink dress is climbing up a set of stairs in an entry way.<br>- A girl going into a wooden building .<br>- A little girl climbing into a wooden playhouse .<br>- A little girl climbing the stairs to her playhouse .<br>- A little girl in a pink dress going into a wooden cabin 
## Model
