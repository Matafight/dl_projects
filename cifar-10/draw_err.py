#_*_coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as  plt
import os
'''
传入误差的数组，画出误差随着epoch变化的曲线，需要传入训练的超参数，比如learning_rate,weight_decay等,这部分用字典的形式传入
将曲线存入文件，文件名就命名为超参数,
应该将训练误差的曲线和验证误差的曲线都画在一个图上
err_arr 应该是个list，list的每个元素也是list
'''


class ErrorPlot():
    def __init__(self,err_arr,**kwargs):
        self.err_arr = err_arr
        self.tar_dir = './err_plots'
        if not os.path.exists(self.tar_dir):
            os.mkdir(self.tar_dir)
        self.plot_name=''
        for i,key in enumerate(kwargs):
            key_value_name = ('_' if i>0 else '')+str(key)+'-'+str(kwargs[key])
            self.plot_name += key_value_name

    def plot(self):
        for err in self.err_arr:
            plt.plot(range(len(err)),err)
        plt.savefig(self.tar_dir+'/'+self.plot_name+'.png')
        # plt.show()



if __name__=='__main__':
    err_arr = [[1,2,3,4,5]]
    para ={
        'a':1,
        'b':23
    }

    solu = ErrorPlot(err_arr,**para)
    solu.plot()
