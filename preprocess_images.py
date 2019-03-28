import os
from multiprocessing.pool import Pool

import click
import numpy as np
import PIL
import cv2

white_list_extensions = ['jpg', 'jpeg', 'JPEG', 'tif']

def convert(file_name, crop_size):
    image = PIL.Image.open(file_name, mode='r')
    assert len(np.shape(image)) == 3, 'Shape of image {} unexpected'.format(file_name)
    width, height = im.size
    converted = None
    if width / float(height) >= 1.3:
        cols_thres = np.where(np.max(np.max(np.asarray(image), axis=2), axis=0) > 35)[0]
        if len(cols_thres) > crop_size//2:
            min_x, max_x = cols_thres[0], cols_thres[-1]
        else:
            min_x, max_x = 0, -1
        converted = image.crop((min_x, 0, max_x, h))
    else:
        converted = image
    # TODO: Perhaps switch to Bibuc or LANCZOS?
    converted = converted.resize((crop_size, crop_size), resample=Image.BILINEAR)
    enhanced_image = contrast_enhance(np.asarray(converted), radius=crop_size//2)
    return Image.fromarray(enhanced_image.astype(np.uint8))

def contrast_enhance(image, radius):
    '''Subtract local average color and map local average to 50% gray
    Parameters
    ==========
    image: array of shape (height, width, 3)
    radius: int
        for square images a good choice is size/2
    Returns
    =======
    contrast enhanced image as array of shape (height, width, 3)
    Reference
    =========
    B. Graham, "Kaggle diabetic retinopathy detection competition report",
        University of Warwick, Tech. Rep., 2015
    https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/kaggleDiabeticRetinopathy/preprocessImages.py
    '''
    radius = int(radius)
    b = np.zeros(image.shape)
    cv2.circle(b, (radius, radius), int(radius * 0.9), (1, 1, 1), -1, 8, 0)
    blurred = cv2.GaussianBlur(image, (0, 0), radius / 30)
    return cv2.addWeighted(image, 4, blurred, -4, 128)*b + 128*(1 - b)

def get_convert_fname(file_name, extension, directory, convert_directory):
    source_extension = file_name.split('.')[-1]
    return file_name.replace(source_extension, extension).replace(directory, convert_directory)

def create_dirs(paths):
    for p in paths:
        try:
            os.makedirs(p)
        except OSError:
            pass

def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, \
        extension, enhance_contrast, ignore_grayscale = arg
    convert_fname = get_convert_fname(fname, extension, directory,
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size, enhance_contrast, ignore_grayscale)
        if img is not None:
            save(img, convert_fname)

def save(img, fname):
    img.save(fname, quality=97)

@click.command()
@click.option('--source_dir', default='data/train', show_default=True,
              help="Directory with original images.")
@click.option('--target_dir', default='data/train_res',
              show_default=True,
              help="Where to save converted images.")
@click.option('--crop_size', default=512, show_default=True,
              help="Size of converted images.")
@click.option('--extension', default='jpeg', show_default=True,
              help="Filetype of converted images.")
@click.option('--n_proc', default=1, show_default=True,
              help="Number of processes for parallelization.")
def main(source_dir, target_dir, crop_size, extension, n_proc):
    '''Image conversion: crop, resize, (enhance colour contrast) and save.'''
    try:
        os.mkdir(target_dir)
    except OSError:
        pass
    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(source_dir)
                 for f in fn if f.split('.')[-1] in white_list_extensions]

    assert filenames, 'No valid filenames.'
    print('Resizing images in {} to {}, this takes a while.'.format(source_dir, target_dir))

    n = len(filenames)
    batchsize = 500
    batches = n // batchsize + 1
    pool = Pool(n_proc)
    args = []
    for f in filenames:
        args.append((convert, (source_dir, target_dir, f, crop_size,
                               extension, enhance_contrast, ignore_grayscale)))
    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])
    pool.close()
    print('done')

if __name__ == '__main__':
    main()
