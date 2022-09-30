# Fruit-Classification-Challenge
The dataset contains 21 kinds of fruit. We firstly built up a baseline network for fruit classification, then we compared several classical convolution neural networks and set up our own suitable network with fine-tuned parameters to improve and reach satisfactory accuracy.

## 1. Dataset:
![image](https://user-images.githubusercontent.com/58734009/193273552-96427e93-1889-4acd-b85e-e173e364a68b.png)

## 2. Data Augmentation:
* random crop: from 256x256 to 224x224
* Color Jitter:  ```transforms.ColorJitter(brightness=0.3)```
* Random Horizontal Flip: 50% randomly
* ToTensor(): transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
* Normalize: transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))


## 3. Baseline Network (GoogleNet):

<img src="https://user-images.githubusercontent.com/58734009/193274232-432df277-51e2-420a-b50b-4fee0fef8446.png" width="500" height="400"><img src="https://user-images.githubusercontent.com/58734009/193274299-ea2e2427-04fb-4d0f-858b-aa913230f2a2.png" width="500" height="400">

## 4. Modified Network structure:

![image](https://user-images.githubusercontent.com/58734009/193276990-4b9f0404-f24a-4c2e-9185-2c28f67a6273.png)

## 5. Results:

![image](https://user-images.githubusercontent.com/58734009/193276825-b1f7d232-4fe9-4d5a-a539-b87757e20a97.png)

![image](https://user-images.githubusercontent.com/58734009/193277027-b8c37ce8-b19f-4446-86f9-4496c90bb309.png)

## 6. Future Work
* Dynamic learning rate scheduler
* Structure update: e.g. advanced connection in DenseNet
* Other methods: transformer, MLP-Mixer

