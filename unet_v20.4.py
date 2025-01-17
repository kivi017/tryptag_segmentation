"""
Major modification to U-Net architecture
Input layer is of size 320x320
This avoids interpolation errors that came up in the previous architecture which image rescaling done to fit to the 512x512 size
Also reduces memory requirement for traning the model

The tensors are now of float32 size to further reduce the memory requirement
This might help to increase the batch size given for training, thereby enabling the use of larger learning rates and therefore reduced training time

v20.3
implementing the Generalized Dice Loss based on the paper by Sudre et al.
Implementation done by Ellen Seifert

v20.4
Modification of loss function
Jadon et al. has proposed using log(cosh(dice_loss)) as a means to get a smoother loss landscape
This, they argue facilities better training with SGD like methods, and also strikes a better balance between precision and recall
We are trying a modification of this where we use log(cosh(generalized_dice_loss))

"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from sklearn.model_selection import train_test_split
import csv
import sys
import json

class LogCoshGeneralizedDiceLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        include_background=True,
        w_type="SQUARE",
        reduction="sum_over_batch_size",
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        name="generalized_dice_loss",
    ):
        super().__init__(name=name, reduction=reduction)
        self.include_background = include_background
        self.w_type = w_type
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

    def w_func(self, grnd):
        if self.w_type == "SIMPLE":
            return tf.math.reciprocal(grnd)
        elif self.w_type == "SQUARE":
            return tf.math.reciprocal(grnd*grnd)
        else:
            return tf.ones_like(grnd)

    def call(self, y_true, y_pred):
        #y_true = tf.transpose(y_true, perm=[0, 3, 1, 2])
        #y_pred = tf.transpose(y_pred, perm=[0, 3, 1, 2])
        #print(y_true.shape)
        #print(y_pred.shape)
        y_pred = tf.squeeze(y_pred, axis=-1)
        intersection = tf.reduce_sum(y_true * y_pred, axis=range(2, len(y_pred.shape)))
        ground_o = tf.reduce_sum(y_true, axis=range(2, len(y_true.shape)))
        pred_o = tf.reduce_sum(y_pred, axis=range(2, len(y_pred.shape)))

        denominator = ground_o + pred_o

        w = self.w_func(tf.cast(ground_o, tf.float32))
        infs = tf.math.is_inf(w)
        w = tf.where(infs, 0.0, w)
        max_values = tf.reduce_max(w, axis=1, keepdims=True)
        w = w + tf.cast(infs, tf.float32) * max_values

        final_reduce_dim = 1
        numer = 2.0 * tf.reduce_sum(intersection * w, axis=final_reduce_dim, keepdims=True) + self.smooth_nr
        denom = tf.reduce_sum(denominator * w, axis=final_reduce_dim, keepdims=True) + self.smooth_dr
        f = 1.0 - (numer / denom)
        f = f[..., -1]
        
        loss = tf.math.log((tf.exp(f) + tf.exp(-f)) / 2.0)

        return loss

def log_cosh_gen_dice_loss(y_true, y_pred):
    x = GeneralizedDiceLoss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)

    return x

# Downsample Block
def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x

def build_unet_model():
    inputs = layers.Input(shape=(320, 320, 1))

    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    bottleneck = double_conv_block(p4, 1024)

    u6 = upsample_block(bottleneck, f4, 512)

    u7 = upsample_block(u6, f3, 256)

    u8 = upsample_block(u7, f2, 128)

    u9 = upsample_block(u8, f1, 64)

    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)

    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    
    return unet_model

def load_dataset(images_file, masks_file, split_ratio=0.2):
    """
    Takes as arguments the paths to the images and masks files
    These files are stored in the .npy format
    """
    
    images = np.load(images_file)
    masks = np.load(masks_file)

    split_size = int(len(images)*split_ratio)
    
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=17)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=17)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=17)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=17)


    return train_x, valid_x, test_x, train_y, valid_y, test_y


def main(no_of_epochs, learning_rate, batch_size, model_version, data_source='default'):
    MODEL_VERSION = model_version
    epochs = int(no_of_epochs)
    learning_rate = float(learning_rate)
    BATCH_SIZE = int(batch_size)
    data_source = data_source
    
        
    if (data_source == 'ilastik') or (data_source == 'tryptag') or (data_source == 'combined'):
        if os.path.exists('constants.json'):
            json_file = open('constants.json')
            json_data = json.load(json_file)
            train_x_path = json_data[data_source]["training"]["images"]
            train_y_path = json_data[data_source]["training"]["masks"]
            valid_x_path = json_data[data_source]["validation"]["images"]
            valid_y_path = json_data[data_source]["validation"]["masks"]
            test_x_path = json_data[data_source]["testing"]["images"]
            test_y_path = json_data[data_source]["testing"]["masks"]
            file_paths = [train_x_path, train_y_path, valid_x_path, valid_y_path, test_x_path, test_y_path]
            if all(os.path.isfile(file_path) for file_path in file_paths):
                print("All the data files specified in constants.json exists")
            else:
                print("Error: Not all files exists")
                return
            train_x = np.load(train_x_path)
            train_y = np.load(train_y_path)
            valid_x = np.load(valid_x_path)
            valid_y = np.load(valid_y_path)
            test_x  = np.load(test_x_path)
            test_y  = np.load(test_y_path)
            json_file.close()
        else:
            print("Error: The constants.json file does not exists")
    else:
        images_path = '/home/gopan/kiran/CellSegmentation/UNet_v20/phase_channel_data_500samples_float32.npy'
        masks_path = '/home/gopan/kiran/CellSegmentation/UNet_v20/ilastic_segmentation_data_500samples_float32.npy'
        file_paths = [images_path, masks_path]
        if all(os.path.isfile(file_path) for file_path in file_paths):
            print("All the data files exists")
        else:
            print("Error: Not all files exists")
            return
        train_x, valid_x, test_x, train_y, valid_y, test_y = load_dataset(images_file = images_path, masks_file = masks_path, split_ratio=0.2)
    
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    
    print("\t Cell Segmentation : Model : "+str(MODEL_VERSION))
    print("\t No of epochs : "+str(epochs))
    print("\t Learning Rate : "+str(learning_rate))
    print("\t Batch Size : "+str(BATCH_SIZE))
    
    
    if os.path.isdir("./unet_"+MODEL_VERSION):
        print("Model version folder exists")
    else:
        os.mkdir("./unet_"+MODEL_VERSION)
        
    if os.path.isdir("./unet_" + str(MODEL_VERSION) + "/N_EPOCHS_"+ str(epochs)):
        print("Epochs sub-folder exists")
    else:
        os.mkdir("./unet_" + str(MODEL_VERSION) + "/N_EPOCHS_"+ str(epochs))
        
    if os.path.isdir("./unet_" + str(MODEL_VERSION) + "/N_EPOCHS_"+ str(epochs) + "/LearningRate_" + str(learning_rate)):
        print("Learning Rate subfolder exists")
    else:
        os.mkdir("./unet_" + str(MODEL_VERSION) + "/N_EPOCHS_"+ str(epochs) + "/LearningRate_" + str(learning_rate))
        
    if os.path.isdir("./unet_" + str(MODEL_VERSION) + "/N_EPOCHS_"+ str(epochs) + "/LearningRate_" + str(learning_rate)+ "/BatchSize_" + str(BATCH_SIZE)):
        print("Batch Size subfolder exists")
    else:
        os.mkdir("./unet_" + str(MODEL_VERSION) + "/N_EPOCHS_"+ str(epochs) + "/LearningRate_" + str(learning_rate)+ "/BatchSize_" + str(BATCH_SIZE))
        
    
    if os.path.isfile("./unet_"+MODEL_VERSION+"/"+MODEL_VERSION+"_Results.csv"):
        print("Resuts CSV exists")
    else:
        f = open("./unet_"+MODEL_VERSION+"/"+MODEL_VERSION+"_Results.csv", 'w')
        writer = csv.writer(f)
        csv_header = ["No_of_EPOCHS", "Learning_Rate", "Batch_Size", "Training_Size", "Validation_Size", 
                      "Test_Size", "Test_Loss", "Test_Binary_IoU"]
        writer.writerow(csv_header)
        f.close()
    
    path_to_folder = "./unet_" + str(MODEL_VERSION) + "/N_EPOCHS_"+ str(epochs) + "/LearningRate_" + str(learning_rate)+ "/BatchSize_" + str(BATCH_SIZE)
    
    
    unet_model = build_unet_model()
    metrics = [tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)]
    callbacks = [
                ModelCheckpoint(path_to_folder + '/unet_model_' + MODEL_VERSION + '.weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
                CSVLogger(path_to_folder + "/data_Unet_model_" + MODEL_VERSION + ".csv"),
                TensorBoard(log_dir= path_to_folder + "/unet_" + MODEL_VERSION + "/logs"),
                EarlyStopping(patience=50, start_from_epoch=300)
            ]
    unet_model.compile(optimizer=Adam(learning_rate = learning_rate), loss=LogCoshGeneralizedDiceLoss(),
                               metrics=metrics)
    model_results = unet_model.fit(tf.stack(train_x), tf.stack(train_y), 
                                           batch_size=BATCH_SIZE, 
                                           epochs=epochs, 
                                           callbacks=callbacks, 
                                           validation_data=(tf.stack(valid_x), tf.stack(valid_y))
                                           )
    eval_loss, binary_io_u = unet_model.evaluate(tf.stack(test_x), tf.stack(test_y))
    f = open("./unet_"+MODEL_VERSION+"/"+MODEL_VERSION+"_Results.csv", 'a')
    writer = csv.writer(f)
    writer.writerow([epochs, learning_rate, BATCH_SIZE, len(train_x), len(valid_x), len(test_x), eval_loss, binary_io_u])
    f.close()

if __name__ == "__main__":
    print("No: of arguments : ",len(sys.argv))
    if len(sys.argv) < 4 :
        raise SyntaxError("Insufficient Arguments")
    elif len(sys.argv) == 5:
        main(no_of_epochs=sys.argv[1], learning_rate=sys.argv[2], batch_size=sys.argv[3], model_version=sys.argv[4], data_source='default')
    elif len(sys.argv) == 6:
        main(no_of_epochs=sys.argv[1], learning_rate=sys.argv[2], batch_size=sys.argv[3], model_version=sys.argv[4], data_source=sys.argv[5])
    else:
        raise SyntaxError("Additional Arguments provided")
