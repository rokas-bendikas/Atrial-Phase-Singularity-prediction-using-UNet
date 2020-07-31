from model import *
from data_pre import *
import numpy as np
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

save_dir = "/media/rokas/HDD/Undergraduate_Project/UNet_keras/unet/results/BCE/500e/"


data_gen_args = dict(rotation_range=0,
                    width_shift_range=0,
                    height_shift_range=0,
                    shear_range=0,
                    zoom_range=0,
                    horizontal_flip=False,
                    fill_mode='nearest')


testGene = dataGenerator(train_path="/media/rokas/HDD/Undergraduate_Project/UNet_keras/unet/data/LGE/test",csv_dir='LGE',target_size=(512,512),threshold = False)

valGene = dataGenerator(train_path="/media/rokas/HDD/Undergraduate_Project/UNet_keras/unet/data/LGE/test",csv_dir='PS',target_size=(512,512),threshold = False)

model = unet(pretrained_weights = '/media/rokas/HDD/Undergraduate_Project/UNet_keras/unet/Models/unet_LGE_bce_500_epochs_SGD.hdf5')

results = model.predict(testGene,verbose=1)
saveResult(save_dir,results)

# evaluate the model
accscores = []
lossscores = []
sensitivity_sc = []
specificity_sc = []

with open(save_dir + "data.csv", mode='w') as csv_file:

    fieldnames = ['acc','loss','sensitivity','specificity']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for X,Y in zip(results,valGene):

        scores = model.evaluate(X.reshape(1,512,512,1),Y.reshape(1,512,512,1),batch_size=1, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %.2f" % (model.metrics_names[0], scores[0]))
        print("%s: %.2f" % (model.metrics_names[2], scores[2]))
        print("%s: %.2f" % (model.metrics_names[3], scores[3]))
        
        accscores.append(scores[1] * 100)
        lossscores.append(scores[0])
        sensitivity_sc.append(scores[2])
        specificity_sc.append(scores[3])
        
            
        writer.writerow({'acc': (scores[1] * 100) , 'loss': (scores[0]), 'sensitivity':(scores[2]), 'specificity':(scores[3])})

    writer.writerow({'acc': np.mean(accscores) , 'loss': np.mean(lossscores), 'sensitivity': np.mean(sensitivity_sc), 'specificity': np.mean(specificity_sc)})
    writer.writerow({'acc': np.std(accscores) , 'loss': np.std(lossscores), 'sensitivity': np.std(sensitivity_sc), 'specificity': np.std(specificity_sc)})
    
print("\n\n%s: %.2f%% (+/- %.2f%%)" % (model.metrics_names[1], np.mean(accscores), np.std(accscores)))
print("%s: %.2f (+/- %.2f)" % (model.metrics_names[0], np.mean(lossscores), np.std(lossscores)))
print("%s: %.2f (+/- %.2f)" % (model.metrics_names[2], np.mean(sensitivity_sc), np.std(sensitivity_sc)))
print("%s: %.2f (+/- %.2f)" % (model.metrics_names[3], np.mean(specificity_sc), np.std(specificity_sc)))



