from Module.dataloader import DataLoader
from Module.util import *
from Module.model import MyResNet, return_resnet, build_optimizer, preproc_resnet, multi_category_focal_loss1
from Module.config import get_cfg_defaults
from Module.augment import PathoAugmentation
import json
import os
import pandas as pd
import tensorflow as tf




if __name__ == "__main__":
    cfg = get_cfg_defaults()
    # assign GPU
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device

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
            targets = data[key][0]['targets']
            if key+".ndpi" in train_slides:
                collect_histogram(targets, train_histogram)
            if key+".ndpi" in valid_slides:
                collect_histogram(targets, valid_histogram)
    print("Histogram: train valid")
    for key in train_histogram.keys():
        print("{:}: {:04d} {:04d}".format(key, train_histogram[key], valid_histogram[key]) )
    print("Sum:", sum(train_histogram.values()), sum(valid_histogram.values()))
    print("======================")
    
    # frequency of each set of slide
    train_frequency = get_frequency_dict(train_histogram)
    valid_frequency = get_frequency_dict(valid_histogram)
    for key in train_histogram.keys():
        print("{:}: {:04d} {:04d}".format(key, train_frequency[key], valid_frequency[key]) )
    print("======================")
    
    # show total number of patches for each classes
    train_total_sample = [train_histogram[key]*train_frequency[key] for key in train_frequency.keys()]
    valid_total_sample = [valid_histogram[key]*valid_frequency[key] for key in valid_frequency.keys()]
    print([train_histogram[key]*train_frequency[key] for key in train_frequency.keys()])
    print([valid_histogram[key]*valid_frequency[key] for key in valid_frequency.keys()])
    print("======================")

    # prepare loader
    print("Prepare train loader: ")
    train_loader = DataLoader(datasets_dir=cfg.DATASET.SLIDE_DIR, 
                              valid_slides=cfg.DATASET.TRAIN_SLIDE,
                              label_path=cfg.DATASET.JSON_PATH,
                              frequency_dict=train_frequency,
                              class_map=class_map,
                              preproc_fn=preproc_resnet if cfg.DATASET.PREPROC else None,
                              augment_fn=PathoAugmentation.augmentation if cfg.DATASET.AUGMENT else None,
                              batch_size=cfg.MODEL.BATCH_SIZE, 
                              num_slide=cfg.DATASET.NUM_SLIDE_HOLD)
    
    print("Prepare validating loader: ")
    valid_loader = DataLoader(datasets_dir=cfg.DATASET.SLIDE_DIR, 
                              valid_slides=cfg.DATASET.VALID_SLIDE,
                              label_path=cfg.DATASET.JSON_PATH,
                              frequency_dict=valid_frequency,
                              class_map=class_map,
                              preproc_fn=preproc_resnet if cfg.DATASET.PREPROC else None,
                              batch_size=cfg.MODEL.BATCH_SIZE, 
                              num_slide=cfg.DATASET.NUM_SLIDE_HOLD)

    # prepare validating imgs and labels
    print("Fetch validating data and labels")
    x_valid, y_valid = valid_loader.pack_data()

    # prepare resnet
    model = return_resnet(cfg.MODEL.BACKBONE, classNum=len(class_map), in_shape=cfg.DATASET.INPUT_SHAPE)

    # prepare optimizer
    optim = build_optimizer(cfg.MODEL.OPTIMIZER, cfg.MODEL.LEARNING_RATE)

    model.build(input_shape=(1,)+cfg.DATASET.INPUT_SHAPE)
    model.compile(loss=[multi_category_focal_loss1(alpha=cfg.MODEL.ALPHA, gamma=2)],
                  metrics=["accuracy"], 
                  optimizer=optim)
    
    # callbacks
    checkpoint_dir=cfg.MODEL.CHECKPOINT_DIR
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    prefix = str(cfg.DATASET.INPUT_SHAPE[0])+"loss_acc_S"+str(cfg.DATASET.NUM_SLIDE_HOLD)+"B"+str(cfg.MODEL.EPOCHS)
    if cfg.DATASET.AUGMENT: 
        prefix+="_AUG"
    if cfg.DATASET.PREPROC: 
        prefix+="_PREPROC"
    # checkpoint+path = checkpoint_dir + prefix + ".h5"
    checkpoint_path = checkpoint_dir + prefix + ".ckpt"
    
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
    
    model.summary()

    # training model
    model.fit( 
        train_loader,
        epochs=cfg.MODEL.EPOCHS,
        batch_size=cfg.MODEL.BATCH_SIZE,
        steps_per_epoch=len(train_loader),
        validation_data=(x_valid, y_valid),
        callbacks=[callbackCheckpoint],
        workers=4, use_multiprocessing=True,
            )
    

    if model.history is not None:
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
        result_dir = cfg.MODEL.RESULT_DIR
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_df.to_csv(os.path.join(result_dir,prefix+".csv"), index=False)
    else:
        print("Training failed: no history found!")  
    
    