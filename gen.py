"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import os

import tarfile
import wget

from common import *
from synthgen import *

import random

## Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1  # no. of times to use the same image
SECS_PER_IMG = 5  # max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH, 'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText.h5'


def get_data():
    """
    Download the image,depth and segmentation data:
    Returns, the h5 database.
    """
    if not osp.exists(DB_FNAME):
        try:
            colorprint(Color.BLUE, '\tdownloading data (56 M) from: ' + DATA_URL, bold=True)
            print()
            sys.stdout.flush()
            out_fname = 'data.tar.gz'
            wget.download(DATA_URL, out=out_fname)
            tar = tarfile.open(out_fname)
            tar.extractall()
            tar.close()
            os.remove(out_fname)
            colorprint(Color.BLUE, '\n\tdata saved at:' + DB_FNAME, bold=True)
            sys.stdout.flush()
        except:
            print(colorize(Color.RED, 'Data not found and have problems downloading.', bold=True))
            sys.stdout.flush()
            sys.exit(-1)
    # open the h5 file and return:
    return h5py.File(DB_FNAME, 'r')


def add_res_to_db(imgname, res, db):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    ninstance = len(res)
    for i in range(ninstance):
        dname = "%s_%d" % (imgname, i)
        db['data'].create_dataset(dname, data=res[i]['img'])
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        # db['data'][dname].attrs['txt'] = res[i]['txt']
        L = res[i]['txt']
        L = [n.encode("ascii", "ignore") for n in L]
        db['data'][dname].attrs['txt'] = L


def main(viz=False, use8000=False, max_placement_num=1):
    # open databases:
    print(colorize(Color.BLUE, 'getting data..', bold=True))
    if use8000:
        depth_db = h5py.File('data/8000/depth.h5')
        seg_db = h5py.File('data/8000/seg.h5')['mask']
    else:
        db = get_data()
        depth_db = db['depth']
        seg_db = db['seg']

    print(colorize(Color.BLUE, '\t-> done', bold=True))

    # open the output h5 file:
    out_db = h5py.File(OUT_FILE, 'w')
    out_db.create_group('/data')
    print(colorize(Color.GREEN, 'Storing the output in: ' + OUT_FILE, bold=True))

    # get the names of the image files in the dataset:
    imnames = sorted(depth_db.keys())
    N = len(imnames)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 0, min(NUM_IMG, N)

    RV3 = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)
    for i in range(start_idx, end_idx):
        # imname = imnames[i]
        imname = random.choice(imnames)
        try:
            # get the image:
            if use8000:
                img = Image.open('data/8000/bg_img/' + imname).convert('RGB')
            else:
                img = Image.fromarray(db['image'][imname][:])
            # get the pre-computed depth:
            #  there are 2 estimates of depth (represented as 2 "channels")
            #  here we are using the second one (in some cases it might be
            #  useful to use the other one):
            depth = depth_db[imname][:].T
            depth = depth[:, :, 1]
            # get segmentation:
            seg = seg_db[imname][:].astype('float32')
            area = seg_db[imname].attrs['area']
            label = seg_db[imname].attrs['label']

            # re-size uniformly:
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz, Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))

            print(colorize(Color.RED, '%d of %d' % (i, end_idx - 1), bold=True))
            res = RV3.render_text(img, depth, seg, area, label,
                                  ninstance=INSTANCE_PER_IMAGE, viz=viz, max_placement_num=max_placement_num)
            if len(res) > 0:
                # non-empty : successful in placing text:
                add_res_to_db(imname, res, out_db)
            # visualize the output:
            # if viz:
            #     if 'q' in input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
            #         break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue
    db.close()
    out_db.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    parser.add_argument('--use8000', action='store_true', dest='use8000', default=False,
                        help='use the 8000 images dataset')
    args = parser.parse_args()
    main(args.viz, args.use8000)
