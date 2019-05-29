from synthgen import *
import random
from PIL import Image

import cv2

# Define some configuration variables:
SECS_PER_IMG = 5  # max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'


def generator(instance_per_img=5):
    # open databases:
    depth_db = h5py.File('data/8000/depth.h5')
    seg_db = h5py.File('data/8000/seg.h5')['mask']

    # get the names of the image files in the dataset:
    imnames = list(depth_db.keys())

    RV3 = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)
    while True:
        imname = random.choice(imnames)
        try:
            # get the image:
            img = Image.open('data/8000/bg_img/' + imname).convert('RGB')

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

            res = RV3.render_text(img, depth, seg, area, label,
                                  ninstance=instance_per_img)
            if len(res) > 0:
                # non-empty : successful in placing text:
                for r in res:
                    yield r

        except KeyboardInterrupt:
            break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue


if __name__ == '__main__':
    gen = generator()

    for g in gen:
        print(g['txt'])
        # cv2.imshow('img', g['img'])