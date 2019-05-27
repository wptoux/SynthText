# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import pygame
from pygame import freetype
from text_utils import FontState
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    pygame.init()

    ys = np.arange(8, 200, dtype='float')
    A = np.c_[ys, np.ones_like(ys)]

    xs = []
    models = {}  # linear model

    FS = FontState()
    # plt.figure()
    # plt.hold(True)
    for i in range(len(FS.fonts)):
        print(i)
        font = freetype.Font(FS.fonts[i], size=50)
        h = []
        for y in ys:
            h.append(font.get_sized_glyph_height(y))
        h = np.array(h)
        m, _, _, _ = np.linalg.lstsq(A, h, rcond=0)
        models[font.name] = m
        xs.append(h)

    pickle.dump(models, open('data/models/font_px2pt.cp', 'wb'))
# plt.plot(xs,ys[i])
# plt.show()
