# Skin inflammatory pattern recognition project
## Training
Training config is in Module/config.py, the config file is written in yacs cnf node format.
To read the config, one can call **get_cfg_defaults()**.

To start training the model, one should type
```
    python3 train.py
```

The result checkpoint file and accuracy/loss information will be stored in checkpoint directory.
During Training, the bbox information will be stored in directory bbox for convenience for future retreival(will not
generate new bbox in runtime).

## Inference
Inference configuration is placed at inference_configs/config.yaml  
One can change to other checkpoint .h file by modifying the config.yaml file.

### Get probability map
To start inferencing the model, type

```
    python3 inference.py
```

The inference result(a chunk of png files) will be stored in result/inference_CSV. If one wants to 
plot and save the ROC, AUC curves as well as confusion matrix, run inferenceLab.ipynb by loading the
corresponding weight(stored in checkpoint/*.h)

![image info](/result/inference_CSV/2019-10-30\ 01.59.42/binary_map/alpha_thres-0.100.png)

### Recolor the image mask
Then recolor the images by  
`./fast_recolor.sh`
The recolor scheme is stored in recolor.py --> color_table(a dictinonary)


or you can manually decide which slide to recolor
```
    python3 recolor.py \
	--mask_dir result/inference_CSV/ \
	--slide_name SLIDE_NAME"
```

![image info](./result/inference_CSV/2019-10-30\ 01.59.42/alpha_thres-0.000.png)

### Post database
move the recolored image folders to */mnt/ai_result/research/A19001_NCKU_SKIN/* and the go to this folder.
```
python3 postdb.py
```

### Automatic inference pipeline
The above pipeline is summarized in a shell script, **pipeline_recolor.sh**

## Fetch contour result from research.aetherai.com
go to label/ directory and run
```
python3 label.py 
```

The latest contour result will be stored in label/label.json file.

## Decision tree 
The decision tree in the second stage of the project is written in directory decision_tree/. Go to
the above directory and run

```
python3 decision_tree.py
```

the final decision tree is printed on screen. If one wants to store the algorithm tree, go back to
skin project root directory and run BDTlab.ipynb. 

# Skin inflammatory cell nuclear segmentation project
## Fetching region of interest(ROI)
The ROI information is stored in label/label.py, ROI is a class with ID=220, one can get the ROI patches
into the folder roi/ by running
```
python3 contour.py
```

## Inference by mmdetection
The mmdetection model will first predict the contours of images in roi/ folder, and the contour
results will be stored in folder roi_result/ .
