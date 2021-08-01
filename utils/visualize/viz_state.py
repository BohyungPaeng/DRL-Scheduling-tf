import cv2, os
import numpy as np
from collections import defaultdict

class VizState(object):
    def __init__(self, file_path, imshow=False, colormap=None):
        self.file_path = file_path
        self.epi = 0
        self.cnt = 0
        self.imshow = imshow
        if colormap is None:
            self.colormap = defaultdict(cv2.COLORMAP_TWILIGHT_SHIFTED)
        else:
            self.colormap = dict()
            colormap_list = [cv2.COLORMAP_SPRING, cv2.COLORMAP_AUTUMN, cv2.COLORMAP_WINTER,  cv2.COLORMAP_COOL,
                             cv2.COLORMAP_BONE, cv2.COLORMAP_MAGMA]
            for i in range(len(colormap)):
                temp_colormap = colormap_list[i%len(colormap_list)]
                idx_list = colormap[i]
                for j in idx_list: self.colormap[j]=temp_colormap

    def new_episode(self):
        self.epi += 1
        self.cnt = 0

    def viz_img_2d(self, state, prod=10):
        if type(state) is not np.ndarray: mat2d = np.array(state)
        if len(mat2d.shape)==1: mat2d = np.reshape(mat2d, (prod,-1))
        elif len(mat2d.shape)==3: pass
        mat2d = np.array(mat2d * 255, dtype=np.uint8)
        h, w = mat2d.shape
        img = None
        for i in range(w):
            img_line = cv2.applyColorMap(mat2d[:,i], self.colormap[i])
            if img is None: img = img_line
            else: img = np.concatenate((img, img_line),axis=1)
        img = cv2.resize(img, (w*10, h*10))
        # img = cv2.resize(mat2d, (w * 10, h * 10))
        if self.imshow:
            cv2.imshow('',img)
            cv2.waitKey(100)
        else:
            cv2.imwrite(os.path.join(self.file_path, 'epi{}_{}.png'.format(self.epi, self.cnt)), img)
            self.cnt+=1

    def viz_img_3d(self, state, col_num=5):
        if type(state) is not np.ndarray: state = np.array(state)
        h, w, c = state.shape
        line_space = 2
        full_w = (w+line_space)*col_num
        full_h = (h+line_space)*(c//col_num+1)
        canvas = np.ones((full_h, full_w), dtype=np.uint8) * 125
        for ch in range(c):
            top = (h+line_space) * (ch//col_num)
            left = (w+line_space) * (ch%col_num)
            mat2d = np.array(state[:,:,ch] * 255, dtype=np.uint8)
            canvas[top:top+h,left:left+w] = mat2d
        img = cv2.resize(canvas, (full_w*5, full_h*5))
        if self.imshow:
            cv2.imshow('',img)
            cv2.waitKey(100)
        else:
            cv2.imwrite(os.path.join(self.file_path, 'epi{}_{}.png'.format(self.epi, self.cnt)), img)
            self.cnt+=1


