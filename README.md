# Deep-Learning Project

## Detect Covid-19 from X-rays

In this repository is the reproducibility of paper : Deep-COVID: Predicting COVID-19 from chest X-ray images using deep transfer learning. (https://arxiv.org/pdf/2004.09363.pdf)

### Dataset
The dataset contains 2084 training and 3100 test images. Two datasets were used to form the one used in our experiments.
 1.  Covid-Chestxray-Dataset. 
 This dataset was filtered by a certified radiologist and 184 images (which showed clear signs of COVID-19) were kept. For the test set, 100 images were used, and 84 of them were used for the training set. 
2. ChexPert dataset was used to enhance this dataset with more non-COVID x-ray images. This dataset has 224,316 chest radiographs, divided into 14 sub-categories (no-finding, Edema, Pneumonia, etc.). There are 2000 non-COVID images in the training set and 3000 non-COVID images in the testing set. 

* Data augmentation is used to increase the number of COVID images 5 times, both in training and test set, so the total number of COVID images is 920 (420 training images and 500 test images). 
* Imbalanced data: we sampled the training set so that we have the same amount of COVID and non-COVID images.  

 ### Models
The pre-trained models are used as feature extractors and only the last layer of each models is modified to fit the number of classes we have - COVID and non-COVID.
* ResNet18
* ResNet50
* SqueezeNet
* DesneNet-121

### Run the code
The script gets arguments from the user, such as input batch size for training, number of epochs to train, number of workers to train, learning rate, momentum  and the path of the dataset. There is a default value for all of these arguments, but if you can specify your own argument too.


```
python resnet18.py --dataset_path ./data/ --batch_size 20 --epoch 10 --num_workers 4 --learning_rate 0.001
python resnet50.py --dataset_path ./data/ --batch_size 20 --epoch 10 --num_workers 4 --learning_rate 0.001
python squeezenet.py --dataset_path ./data/ --batch_size 20 --epoch 10 --num_workers 4 --learning_rate 0.001
python densenet121.py --dataset_path ./data/ --batch_size 20 --epoch 10 --num_workers 4 --learning_rate 0.001
```
### Analysis
Given the path for the test samples, the inference code provides the predicted scores (probabilities) and predicted labels of the samples. Also, you can experiment with different cut-off threshold and check  the sensitivity and specificity metrics. Histogram of the predicted probabilities, the confusion matrix, and ROC curve are also auto-generated.

```
python Inference.py --test_covid_path ./data/val/covid/ --test_non_covid_path ./data/val/non/ --trained_model_path ./results/resnet18/epoch10/covid_resnet18_epoch10.pt --cut_off_threshold 0.2 --batch_size 20 --num_workers 0
```

