from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

def trainGenerator(batch_size_train, batch_size_val, train_path, val_path,
                    image_folder, mask_folder, train_aug_dict, val_aug_dict,
                   target_size = (128,128), seed = 15):
    
    train_datagen = ImageDataGenerator(**train_aug_dict)
    val_datagen = ImageDataGenerator(**val_aug_dict)

    train_image_generator = train_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None, 
        target_size = target_size,
        batch_size = batch_size_train,
        seed = seed
        )
    train_mask_generator = train_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        target_size = target_size,
        batch_size = batch_size_train,
        seed = seed
        )
    train_generator = zip(train_image_generator, train_mask_generator)

    val_image_generator = val_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        target_size = target_size,
        batch_size = batch_size_val
        )
    val_mask_generator = val_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        target_size = target_size,
        batch_size = batch_size_val
        )
    val_generator = zip(val_image_generator, val_mask_generator)

    return train_generator, val_generator