import json
import numpy as np
import os

# import package from other folder
import sys
sys.path.append('../torch_training/')

from pytorch_model.config import get_cfg_defaults
from pytorch_model.util import open_json, create_statistic_table, \
     print_sampling_method, balance_contour, get_contour_frequency, get_bbox_one_slide

if __name__ == '__main__':

    import cProfile
    from contextlib import contextmanager

    @contextmanager
    def profiler():
        profiler = cProfile.Profile()
        profiler.enable()
        yield profiler
        profiler.disable()

    cfg = get_cfg_defaults()
    slide_dir    = cfg.DATASET.SLIDE_DIR 
    train_slides = cfg.DATASET.TRAIN_SLIDE
    test_slides  = cfg.DATASET.TEST_SLIDE 
    extension    = cfg.DATASET.EXTENSION 
    class_map    = cfg.DATASET.CLASS_MAP
    id2name      = {item[0]: item[3] for item in cfg.DATASET.CLASS_MAP}
    id2cls       = {item[0]:item[1] for item in cfg.DATASET.CLASS_MAP}
    bbox_dir     = './bbox_0121/'

    os.makedirs(bbox_dir, exist_ok=True)

    js,  keys = open_json(cfg.DATASET.LABEL_PATH)
    for slides in [train_slides, test_slides]:
        statistics = create_statistic_table(slides, js, class_map, extension)
        contour_freq = get_contour_frequency(statistics)
        sample_freq = balance_contour(contour_freq)
        print_sampling_method(contour_freq, sample_freq, class_map)

        with profiler() as pr:
            for slide in slides:
                bboxes = get_bbox_one_slide(slide_dir=slide_dir, 
                                            sample_slide_name=slide, 
                                            sample_freq=sample_freq, 
                                            label_path='../label/label_0118.json', 
                                            id2cls=id2cls,
                                            patch_size=cfg.DATASET.PATCH_SIZE,
                                            extension=extension,
                                            robust=True,
                                            )
                # print(bboxes)
                np.save(os.path.join(bbox_dir, f'{slide.split(extension)[0]}'), bboxes)
        pr.print_stats(sort='cumtime')


