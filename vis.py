import json
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict
from scipy.ndimage import gaussian_filter1d

class visualize_mmdetection():
    def __init__(self, path):
        self.log = open(path)
        self.dict_list = list()
        self.loss_rpn_bbox = list()
        self.loss_rpn_cls = list()
        self.loss_bbox = list()
        self.loss_cls = list()
        self.loss_obj = list()
        self.loss = list()
        self.acc = list()

    def load_data(self):
        for line in self.log:
            info = json.loads(line)
            self.dict_list.append(info)

        for i in range(1, len(self.dict_list)):
            for value, key in dict(self.dict_list[i]).items():
                # ------------find key for every iter-------------------#
                if(dict(self.dict_list[i])['mode']=='val'):
                    continue
                if(dict(self.dict_list[i])['iter']!=1):
                    continue
                loss_bbox_value = dict(self.dict_list[i])['loss_bbox']
                loss_cls_value = dict(self.dict_list[i])['loss_cls']
                oss_obj_value = dict(self.dict_list[i])['loss_obj']
                loss_value = dict(self.dict_list[i])['loss']
                # -------------list append------------------------------#

                self.loss_bbox.append(loss_bbox_value)
                self.loss_cls.append(loss_cls_value)
                self.loss_obj.append(oss_obj_value)
                self.loss.append(loss_value)
 
                # -------------clear repeated value---------------------#
        self.loss_rpn_cls = list(OrderedDict.fromkeys(self.loss_rpn_cls))
        self.loss_rpn_bbox = list(OrderedDict.fromkeys(self.loss_rpn_bbox))
        self.loss_bbox = list(OrderedDict.fromkeys(self.loss_bbox))
        self.loss_cls = list(OrderedDict.fromkeys(self.loss_cls))
        self.loss_obj = list(OrderedDict.fromkeys(self.loss_obj))
        self.loss = list(OrderedDict.fromkeys(self.loss))
        self.acc = list(OrderedDict.fromkeys(self.acc))

    def show_chart(self):
        
        plt.rcParams.update({'font.size': 25})

        plt.figure(figsize=(10, 10))
        self.loss_cls = gaussian_filter1d(self.loss_cls, sigma=2)
        self.loss_bbox = gaussian_filter1d(self.loss_bbox, sigma=2)
        self.loss_obj = gaussian_filter1d(self.loss_obj, sigma=2)
        self.loss = gaussian_filter1d(self.loss, sigma=2)
        # plt.subplot(141, title='loss_cls', ylabel='loss')
        # plt.plot(self.loss_cls)
        # plt.subplot(142, title='loss_bbox', ylabel='loss')
        # plt.plot(self.loss_bbox)
        # plt.subplot(143, title='loss_obj', ylabel='loss')
        # plt.plot(self.loss_obj)
        plt.subplot(111, title='total loss', ylabel='loss')
        plt.plot(self.loss)


        plt.suptitle("Loss", fontsize=30)
        plt.savefig(('/home/tongxin/mmdetection/result_one.png'), dpi=750)

if __name__ == '__main__':
    x = visualize_mmdetection(sys.argv[1])
    x.load_data()
    x.show_chart()
