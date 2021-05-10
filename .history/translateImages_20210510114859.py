import inkml2img
import os
# inkml2img.inkml2img('2013ALL_inkml_data/200923-1556-49.inkml','./2013ALL_inkml_data_image/200923-1556-49.png')

test_chrome = r'.\data\CROHME DATA\TRAIN_CROHME_8836'
# test_chrome = directory + '\TEST_CROHME_2133'
i = 0
for entry in os.scandir(test_chrome):
    if (entry.path.endswith(".inkml")
            or entry.path.endswith(".inmkl")) and entry.is_file():
        print(entry.path)

        inkml2img.inkml2img(entry.path, r'.\data\CROHME DATA\TrainTransformed\Image' + str(i) +'.png')
        i+= 1
        input()