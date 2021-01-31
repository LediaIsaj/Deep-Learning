import imgaug as ia
import imgaug.augmenters as iaa
import imageio
from PIL import Image
import os

ia.seed(1)

seq = iaa.Sequential([
    iaa.Fliplr(0.3),
    iaa.GaussianBlur(sigma=(0.0, 1.5))
], random_order=True)


train_covid_path = os.path.expanduser('./data/train/covid/')
for filename in os.listdir(train_covid_path):
    if filename.endswith((".jpg", ".jpeg", "png")):
        print(filename)

        for i in range(4):
            img = imageio.imread(train_covid_path+filename)
            img_aug = seq(images=img)
            # ia.imgaug.imshow(img_aug)

            im = Image.fromarray(img_aug)
            num = str(i)
            im.save(train_covid_path + 'img_aug_' + num + filename + '.png')


val_covid_path = os.path.expanduser('./data/val/covid/')
for filename in os.listdir(val_covid_path):
    if filename.endswith((".jpg", ".jpeg", "png")):
        print(filename)

        for i in range(4):
            img = imageio.imread(val_covid_path+filename)
            img_aug = seq(images=img)
            # ia.imgaug.imshow(img_aug)

            im = Image.fromarray(img_aug)
            num = str(i)
            im.save(val_covid_path + 'img_aug_' + num + filename + '.png')