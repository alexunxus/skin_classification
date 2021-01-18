import os

if __name__=="__main__":
    src_dir     = '/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20191106/'
    train_dir   = '../train/ '
    test_dir    = '../test/'
    test_slides =   ["2019-10-30 02.01.19.ndpi",
    "2019-10-30 02.02.21.ndpi",
    "2019-10-30 02.05.46.ndpi", ] 

    for file in os.listdir(src_dir):
        if file not in test_slides:
            if os.path.isfile(os.path.join(train_dir, file)):
                print('skipping...')
                continue
            os.symlink(os.path.join(src_dir, file), os.path.join(train_dir, file), target_is_directory=False)
        else:
            target = os.path.join(test_dir, file)
            if os.path.isfile(target):
                print('skipping...')
                continue
            print(f'Test slide {file} collect4ed')
            os.symlink(os.path.join(src_dir, file), target, target_is_directory=False)
    
    extra_train_dir   = '/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20210107/'
    extra_train_slide = ['19-D02531.ndpi', '19-D01869.ndpi', '20-000077.ndpi']
    
    for file in extra_train_slide:
        target = os.path.join(train_dir, file)
        if not os.path.isfile(extra_train_dir+file):
            raise ValueError(f"{extra_train_dir+file} does not exist!")
        if os.path.isfile(train_dir+file):
            print('skipping...')
            continue
        os.symlink(src=extra_train_dir+file, dst=target, target_is_directory=False)
