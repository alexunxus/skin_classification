import os

def create_soft_link(src_dir, target_dir, file):
    target_path = os.path.join(target_dir, file)
    if os.path.isfile(target_path):
        print('skipping...')
        return
    os.symlink(src_dir+file, target_path, target_is_directory=False)

if __name__=="__main__":
    src_dir     = '/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20191106/'
    train_dir   = '../train/'
    test_dir    = '../test/'
    test_slides =   [
        "2019-10-30 02.01.19.ndpi",
        "2019-10-30 02.02.21.ndpi",
        "2019-10-30 02.05.46.ndpi", 
    ] 

    for file in os.listdir(src_dir):
        if file not in test_slides:
            create_soft_link(src_dir, train_dir, file)
        else:
            create_soft_link(src_dir, test_dir, file)

    extra_train_dir   = '/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20210107/'
    extra_train_slide = ['19-D02531.ndpi', '19-D01869.ndpi']
    
    for file in extra_train_slide:
        create_soft_link(extra_train_dir, train_dir, file)