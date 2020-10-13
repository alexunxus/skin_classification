from Module.dataloader import DataLoader, SlideDataSet
from Module.util import *
from Module.model import MyResNet, return_resnet, build_optimizer, preproc_resnet, multi_category_focal_loss1, scheduler
from Module.config import get_cfg_defaults
from Module.augment import PathoAugmentation
import json
import os
import pandas as pd
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import numpy as np
import argparse
from Module.config import get_cfg_defaults

parser = argparse.ArgumentParser(description='Train with config file')
parser.add_argument('--config', default="Module.config",
                    help='config file path')

args = parser.parse_args()

if __name__ == "__main__":

    cfg = get_cfg_defaults()
    # assign GPU
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    if cfg.SYSTEM.USE_HOROVOD:
        # Horovod: initialize Horovod.
        hvd.init()
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    is_hvd_0 = not cfg.SYSTEM.USE_HOROVOD or hvd.rank() == 0

    slide_dir=cfg.DATASET.SLIDE_DIR
    
    # prepare label dictionary and frequency dictionary
    train_histogram = {}
    train_slides=cfg.DATASET.TRAIN_SLIDE
    valid_histogram = {}
    valid_slides=cfg.DATASET.VALID_SLIDE
    class_map  = get_class_map(cfg.DATASET.CLASS_MAP)
    with open(cfg.DATASET.JSON_PATH) as f:
        data = json.load(f)
        for key, val in data.items():
            if type(data[key]) is list:
                targets = data[key][0]['targets']
            elif type(data[key]) is dict:
                targets = data[key]["targets"]
            else:
                print("Indeciphorable json file!")
                raise ValueError
            if key+".ndpi" in train_slides:
                collect_histogram(targets, train_histogram, interest=cfg.DATASET.INT_TO_CLASS)
            if key+".ndpi" in valid_slides:
                collect_histogram(targets, valid_histogram, interest=cfg.DATASET.INT_TO_CLASS)
    if is_hvd_0:
        print(train_histogram.keys())
        print(valid_histogram.keys())
        print("Histogram: train valid")
        train_keys , valid_keys = list(train_histogram.keys()), list(valid_histogram.keys())
        train_keys.sort()
        valid_keys.sort()
        assert train_keys == valid_keys
        for key in train_histogram.keys():
            print("{:}: {:4d} {:4d}".format(key, train_histogram[key], valid_histogram[key]) )
        print("Sum:", sum(train_histogram.values()), sum(valid_histogram.values()))
        print("======================")
        
    # frequency of each set of slide
    upsample = 1 if cfg.DATASET.INPUT_SHAPE[0] >=1024 else 4
    train_frequency = get_frequency_dict(train_histogram, upsample=upsample)
    valid_frequency = get_frequency_dict(valid_histogram, upsample=upsample)
    if is_hvd_0:
        for key in train_histogram.keys():
            print("{:}: {:4d} {:4d}".format(key, train_frequency[key], valid_frequency[key]) )
        print("======================")
        
    # show total number of patches for each classes
    train_total_sample = [train_histogram[key]*train_frequency[key] for key in train_frequency.keys()]
    valid_total_sample = [valid_histogram[key]*valid_frequency[key] for key in valid_frequency.keys()]
    if is_hvd_0:
        print([train_histogram[key]*train_frequency[key] for key in train_frequency.keys()])
        print([valid_histogram[key]*valid_frequency[key] for key in valid_frequency.keys()])
        print("======================")

    # prepare loader
    if is_hvd_0:
        print("==========Prepare train loader=================")
    train_datasets =[SlideDataSet(slide_path=cfg.DATASET.SLIDE_DIR,
                                    slide_name=cfg.DATASET.TRAIN_SLIDE[i],
                                    label_path=cfg.DATASET.JSON_PATH,
                                    frequency_dict=train_frequency,
                                    class_map=class_map,
                                    patch_size=cfg.DATASET.INPUT_SHAPE[:2],
                                    interest = cfg.DATASET.INT_TO_CLASS,
                                    num_worker=10,
                                    preproc=preproc_resnet if cfg.DATASET.PREPROC else None,
                                    augment=PathoAugmentation.augmentation if cfg.DATASET.AUGMENT else None,
                                    save_bbox=True, 
                                    multiscale=cfg.MODEL.MULTISCALE) for i in range(len(cfg.DATASET.TRAIN_SLIDE))]
    train_loader = DataLoader(datasets=train_datasets, 
                              batch_size=cfg.MODEL.BATCH_SIZE)
    
    if is_hvd_0: 
        print("==========Prepare validating loader=============")
        valid_datasets=[SlideDataSet(slide_path=cfg.DATASET.SLIDE_DIR,
                                     slide_name=cfg.DATASET.VALID_SLIDE[i],
                                     label_path=cfg.DATASET.JSON_PATH,
                                     frequency_dict=valid_frequency,
                                     class_map=class_map,
                                     patch_size=cfg.DATASET.INPUT_SHAPE[:2],
                                     interest = cfg.DATASET.INT_TO_CLASS,
                                     preproc=preproc_resnet if cfg.DATASET.PREPROC else None,
                                     augment=None,
                                     shuffle=False,
                                     num_worker=10,
                                     save_bbox = True,
                                     multiscale=cfg.MODEL.MULTISCALE) for i in range(len(cfg.DATASET.VALID_SLIDE))]
        valid_loader = DataLoader(datasets=valid_datasets, 
                                  batch_size=cfg.MODEL.BATCH_SIZE)
        # prepare validating imgs and labels
        # print("==========Fetch validating data and labels========")
        # x_valid, y_valid = valid_loader.pack_data()

    # prepare resnet
    model = return_resnet(cfg.MODEL.BACKBONE, classNum=len(class_map), in_shape=cfg.DATASET.INPUT_SHAPE)

    if cfg.SYSTEM.USE_HOROVOD:
        # Horovod: adjust learning rate based on number of GPUs.
        scaled_lr = cfg.MODEL.LEARNING_RATE * np.sqrt(hvd.size())
        # prepare optimizer
        optim = build_optimizer(cfg.MODEL.OPTIMIZER, scaled_lr)
    else:
        # use lr in config
        optim = build_optimizer(cfg.MODEL.OPTIMIZER, cfg.MODEL.LEARNING_RATE)

    model.build(input_shape=(1,)+cfg.DATASET.INPUT_SHAPE)
    model.compile(loss=[multi_category_focal_loss1(alpha=cfg.MODEL.ALPHA, gamma=2)],
                  metrics=["accuracy"], 
                  optimizer=optim)
    
    # callbacks
    checkpoint_dir=cfg.MODEL.CHECKPOINT_DIR
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # prefix : R-101-v1_512_
    prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.INPUT_SHAPE[0]}_E{cfg.MODEL.EPOCHS}_cls{len(class_map)}"
    if cfg.DATASET.AUGMENT: 
        prefix+="_AUG"
    if cfg.DATASET.PREPROC: 
        prefix+="_PREPROC"
    if cfg.MODEL.MULTISCALE:
        prefix+="_MULTISCALE"
    checkpoint_path =os.path.join(checkpoint_dir, prefix + ".h5")
    # checkpoint_path = checkpoint_dir + prefix + ".ckpt"
    
    # loading checkpoints:
    if os.path.isfile(checkpoint_path):
        if is_hvd_0:
            print("==============Loading model weights===============")
        model.load_weights(checkpoint_path)
    
    callbackCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                            monitor='val_loss',
                                                            save_best_only=True,
                                                            save_weights_only= True,
                                                            mode='auto')

    callbackEarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         min_delta=0,
                                                         patience=10,
                                                         verbose=0,
                                                         mode='auto')

    callbackLRscheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    callbacks = [callbackEarlyStop, callbackLRscheduler]

    model.summary()

    if cfg.SYSTEM.USE_HOROVOD:
        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(callbackCheckpoint)

        # Horovod: write logs on worker 0.
        verbose = 1 if hvd.rank() == 0 else 0
    else:
        callbacks.append(callbackEarlyStop)

    # training model
    model.fit( 
        train_loader,
        epochs=cfg.MODEL.EPOCHS,
        batch_size=cfg.MODEL.BATCH_SIZE,
        steps_per_epoch=len(train_loader),
        #validation_data=(x_valid, y_valid),
        validation_data=valid_loader if is_hvd_0 else None,
        callbacks=[callbackCheckpoint, callbackEarlyStop],
        workers=4, use_multiprocessing=True,
        verbose=(1 if is_hvd_0 else 0)
            )
    
    if model.history is not None and is_hvd_0:
        print(model.history.history.keys())
        train_loss = model.history.history['loss']
        valid_loss = model.history.history['val_loss']
        train_acc =  model.history.history['accuracy']
        valid_acc =  model.history.history['val_accuracy']
        result_df =  pd.DataFrame({"train_loss":train_loss,
                                   "valid_loss":valid_loss,
                                   "train_acc":train_acc,
                                   "valid_acc":valid_acc
                                  })
        if is_hvd_0:
            epoch_taken = np.argmin(valid_loss)
            print(f"Minimal validating loss: {valid_loss[epoch_taken]}")
            print(f"Train Accuracy: {train_acc[epoch_taken]}, Valid Accuracy:{valid_acc[epoch_taken]}")
            result_dir = cfg.MODEL.RESULT_DIR
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            result_df.to_csv(os.path.join(result_dir,prefix+".csv"), index=False)
    
    