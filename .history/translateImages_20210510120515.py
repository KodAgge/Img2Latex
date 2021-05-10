import inkml2img
import os
# inkml2img.inkml2img('2013ALL_inkml_data/200923-1556-49.inkml','./2013ALL_inkml_data_image/200923-1556-49.png')

mode = 'train' # EK) If we're transforming training, test or validation images.

if mode == 'train':
    img_folder = r'./data/CROHME DATA/TRAIN_CROHME_8836'
elif mode == 'validation':
    img_folder = r'./data/CROHME DATA/VAL_CROHME_671'
elif mode == 'test':
    img_folder = r'./data/CROHME DATA/TEST_CROHME_2133'
else:
    raise ValueError('Value of variable "mode" not recognized, check spelling :)') 


# img_folder = directory + '\TEST_CROHME_2133'
i = 0
for entry in os.scandir(img_folder):
    if (entry.path.endswith(".inkml") or entry.path.endswith(".inmkl")) and entry.is_file():
        print(entry.path)

        img_name = 'Image' + str(i) +'.png'
        inkml2img.inkml2img(entry.path, r'./data/CROHME DATA/' + mode + '_transformed/' + img_name)
        
        i+= 1
        input('Done')