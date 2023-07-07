import cv2, os
import numpy as np
from collections import defaultdict

class VizState(object):
    def __init__(self, file_path, imshow=False, colormap=None):
        self.file_path = file_path
        self.env_idx = 0
        self.cnt = 0
        self.imshow = imshow
        if colormap is None:
            self.colormap = defaultdict(lambda:cv2.COLORMAP_TWILIGHT_SHIFTED)
        else:
            self.colormap = dict()
            colormap_list = [cv2.COLORMAP_SPRING, cv2.COLORMAP_AUTUMN, cv2.COLORMAP_WINTER,  cv2.COLORMAP_COOL,
                             cv2.COLORMAP_BONE, cv2.COLORMAP_MAGMA]
            for i in range(len(colormap)):
                temp_colormap = colormap_list[i%len(colormap_list)]
                idx_list = colormap[i]
                for j in idx_list: self.colormap[j]=temp_colormap

        self.epoch = 0
        self.state_list = list()
        self.info_list = list()
    def tsne(self):
        """
        Cluster state trajectories with various perspectives
        model validation mode: Label as indices of validation problem (color)
        Training curriculum mode : Epoch can be label. How to visualize the Q-values?
        """
        X = np.vstack([s.flatten() for s in self.state_list])
        labels = np.hstack([i[3] for i in self.info_list])
        """ Information type for each index of self.info_list
        0 : training epoch
        1 : environment index (validation seed index, normally 300~329)
        2 : Time-step 
        index 3 : Q-value
        """
        import sklearn
        from sklearn.manifold import TSNE
        tsne_rslt = TSNE(perplexity=30).fit_transform(X)

        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        # Create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(tsne_rslt[:, 0], tsne_rslt[:, 1], lw=0, s=40, c=labels)
        plt.colorbar(sc,ax=ax)
        title_txt = 'tsne{}_{}.png'.format(self.epoch, self.env_idx)
        plt.title(title_txt)
        # Add the labels for each digit.
        # txts = []
        # for i in range(10):
        #     # Position of each label.
        #     xtext, ytext = np.median(x[colors == i, :], axis=0)
        #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
        #     txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        #     txts.append(txt)
        plt.savefig(os.path.join(self.file_path, title_txt))
        f.clear()
        plt.close('all')

        """Save tsne rslt as csv file"""
        import pandas as pd
        rslt_df = pd.DataFrame(np.concatenate((tsne_rslt,np.stack(self.info_list)), axis=1))
        rslt_df.columns=['tx','ty','epoch','env','cnt','Q']
        rslt_df.to_csv(os.path.join(self.file_path, 'tsne_rslt.csv'), mode='a', index=False, header=False)
        # for fast training
        self.state_list.clear()
        self.info_list.clear()

        # with open(os.path.join(self.file_path, 'tsne_rslt.csv'), 'a') as f:

    def new_episode(self, episode):
        if self.epoch != episode:
            self.epoch = episode
            self.env_idx = 0
        else:
            self.env_idx += 1
        self.cnt = 0

    def viz_img_2d(self, observe, prod=10):
        state = observe['state']
        if type(state) is not np.ndarray: mat2d = np.array(state)
        if len(mat2d.shape)==1: mat2d = np.reshape(mat2d[:-2], (prod,-1))
        elif len(mat2d.shape)==3: pass
        """Save state information"""
        self.cnt += 1
        q_value = observe['max_q']
        self.info_list.append([self.epoch, self.env_idx, self.cnt, q_value])
        self.state_list.append(mat2d)
        """Visuzlize states"""
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
            if self.env_idx != 0: return #only save img for env_idx = 0
            if self.epoch == 500 or self.epoch % 5000 == 0:
                cv2.imwrite(os.path.join(self.file_path, 'epoch{}_epi{}_{}.png'.format(self.epoch, self.env_idx, self.cnt)), img)

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
            cv2.imwrite(os.path.join(self.file_path, 'epi{}_{}.png'.format(self.env_idx, self.cnt)), img)
            self.cnt+=1


