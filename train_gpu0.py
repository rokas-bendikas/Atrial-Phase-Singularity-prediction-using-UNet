from model import *
from data_pre import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



data_gen_args_train = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
                
data_gen_args_val = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene_train = trainGenerator(4,'/media/rokas/HDD/Undergraduate_Project/UNet_keras/unet/data/LGE/train','LGE','PS',data_gen_args_train,save_to_dir = None,target_size=(512,512),seed=13,show_data=False,threshold = False)

myGene_val = trainGenerator(3,'/media/rokas/HDD/Undergraduate_Project/UNet_keras/unet/data/LGE/validate','LGE','PS',data_gen_args_val,save_to_dir = None,target_size=(512,512),show_data=False, seed=4,threshold = False)

model = unet(loss='kld')
model_checkpoint = ModelCheckpoint('unet_LGE_kld_5000e.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

model.fit_generator(myGene_train,steps_per_epoch=39,epochs=5000,callbacks=[model_checkpoint],validation_data=myGene_val,validation_steps=2,shuffle=True,use_multiprocessing=True, max_queue_size = 50)

