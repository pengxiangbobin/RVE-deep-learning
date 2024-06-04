from model import *
from preprocess_data import *

train_gen_args = dict(rescale=1/255, vertical_flip=True, horizontal_flip=True)
val_gen_args = dict(rescale=1/255)
           
my_train, my_val = trainGenerator(16, 4, 'g:/RVE_pre/train',
                                  'g:/RVE_pre/val',
                                    'image','mask', train_gen_args, val_gen_args)

model_checkpoint = ModelCheckpoint('g:/RVE_pre/xxx.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
#tensorboard = TensorBoard(log_dir='g:/RVE_pre/log', histogram_freq=1, embeddings_freq=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0)
callbacks_list = [model_checkpoint, reduce_lr]

model = unet()
#two-stage training
#model = load_model('g:/RVE_pre/xxx.hdf5')

history = model.fit(my_train,
                    validation_data=my_val,
                    steps_per_epoch=50,
                    validation_steps=25,
                    epochs=100,
                    callbacks=callbacks_list, 
                    verbose=1)
