#_*_coding:utf-8_*_

##读取完整的训练数据并训练，并给出在测试集上的预测结果
## 也可以先只用90%的训练集训练的模型 给出在测试集上的预测结果

import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import os
from draw_err import ErrorPlot
import argparse
from model import resnet18
from dataloader import my_data_loader

def load_existsing_model(path):
    model = resnet18()
    model.load_state_dict(torch.load(path))
    return model


def pred_on_test(model,test_data_loader):
    model.eval()
    model.cuda()
    pred_on_test=[]
    img_ids = []
    for x,y,img_path in test_data_loader:
        x = x.cuda()
        img_id = [os.path.basename(path_item)[:-4] for path_item in img_path]
        img_ids.extend(img_id)
        with torch.no_grad():
            pred_y = model(x)
            ##只要获得最大的那个值的索引就可以了。
            pred_label=torch.argmax(pred_y.detach().cpu(),dim=1)
            pred_on_test.extend(list(pred_label.numpy()))
    return pred_on_test,img_ids


def convert_idx2class(classname2idx,pred_idx):
    inversed_classname2idx = {}
    for key in classname2idx.keys():
        val = classname2idx[key]
        inversed_classname2idx[val] = key
    classnames = []
    for idx in pred_idx:
        classnames.append(inversed_classname2idx[idx])
    return classnames


if __name__=='__main__':
    # path = './checkpoint/num_epoches-15_learning_rate-0.0001_learning_rate_decay-0.5.pickle'
    path = './checkpoint/num_epoches-100_learning_rate-0.0001_learning_rate_decay-0.5.pickle'
    _, _, test_data_loader, classname_idx_map = my_data_loader(512)
    model = load_existsing_model(path)
    pred_on_test,img_ids = pred_on_test(model,test_data_loader)

    classnames = convert_idx2class(classname_idx_map,pred_on_test)

    ## 读取sampleSubmission
    import pandas as pd
    sample_sub = pd.read_csv('sampleSubmission.csv')
    sample_sub['label']=classnames
    sample_sub['id']=img_ids
    sample_sub.to_csv('my_fourth_submission.csv',index=False)


