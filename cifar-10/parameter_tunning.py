import torch.nn as nn
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import os
from draw_err import ErrorPlot
import argparse
from sklearn.metrics import accuracy_score
from model import resnet18
from dataloader import my_data_loader
import copy


def early_stopping(valid_loss, early_stop_epoch):
    '''
    Args:
        valid_loss : list， 验证集每个epoch的损失
        early_stop_epoch: int，连续多少个epoch 验证集上的误差都没有降低就停止训练

    Returns:
        返回bool类型，表示是否需要停止训练，True or False
    '''
    if len(valid_loss) <= 1:
        return False

    min_loss = min(valid_loss)
    for ls in valid_loss[:-early_stop_epoch-1:-1]:
        if ls <= min_loss:
            return False
    return True


def checkpoint_tmp(model, save_name, **kwargs):
    tar_dir = './checkpoint'
    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)
    # 保存模型参数
    torch.save(model.state_dict(), tar_dir+'/' + save_name + '.pickle')


def train(num_epoch, learning_rate, save_name,batch_size, early_stopping_epoch=10, cuda_flag=False):
    # 基本参数
    num_epoch = num_epoch
    lr_decay = 500
    lr = learning_rate
    batch_size=batch_size

    model = resnet18()
    if cuda_flag:
        print('using cuda...')
        model = model.cuda()
    else:
        print('using cpu...')
    # 模型参数初始化
    def weight_init(m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_()  # 全连接层参数初始化
        elif isinstance(m, nn.BatchNorm2d):
            # # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    model.apply(weight_init)

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义优化函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1000, gamma=lr_decay)

    train_data_loader, valid_data_loader, test_data_loader, _ = my_data_loader(
        batch_size)
    train_loss_epoch = []
    valid_loss_epoch = []
    best_model = None
    import sys
    min_valid_loss = sys.maxsize
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        train_pred_label = []
        train_true_label = []
        valid_pred_label = []
        valid_true_label = []
        # 评估accuracy
        for x, y in train_data_loader:
            train_true_label.extend(y)
            if cuda_flag:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()  # 这行代码很关键，不加会导致训练误差和验证误差都很大！！！
            pred_y = model(x)
            loss = loss_func(pred_y, y)
            if cuda_flag:
                pred_label = torch.argmax(pred_y.detach().cpu(), dim=1)
                train_pred_label.extend(pred_label)
                train_loss += loss.detach().cpu().numpy()
            else:
                train_loss += loss.detach().numpy()
            loss.backward()
            optimizer.step()
        train_accuracy = accuracy_score(train_true_label, train_pred_label)

        # valid performance
        model.eval()
        valid_loss = 0

        for valid_x, valid_y in valid_data_loader:
            valid_true_label.extend(valid_y)
            if cuda_flag:
                valid_x = valid_x.cuda()
                valid_y = valid_y.cuda()
            with torch.no_grad():  # 这行代码很关键，不加就会导致 cuda out of memory 的问题
                pred_valid_y = model(valid_x)
                cur_valid_loss = loss_func(pred_valid_y, valid_y)
            if cuda_flag:
                pred_valid_label = torch.argmax(
                    pred_valid_y.detach().cpu(), dim=1)
                valid_pred_label.extend(pred_valid_label)
                valid_loss += cur_valid_loss.detach().cpu().numpy()

            else:
                valid_loss += cur_valid_loss.detach().numpy()
        valid_accuracy = accuracy_score(valid_true_label, valid_pred_label)
        print('===========epoch:{}, train_loss:{},  valid_loss:{}, train acc:{},valid acc:{}==========='.format(
            epoch, train_loss, valid_loss, train_accuracy, valid_accuracy))
        train_loss_epoch.append(train_loss)
        valid_loss_epoch.append(valid_loss)
    
        ## 只需要保存当前效果最好的轮次的模型就可以了
        if valid_loss<=min_valid_loss:
            best_model = copy.deepcopy(model)
        if early_stopping(valid_loss_epoch, early_stopping_epoch):
            break
        # checkpoint
        # 每隔10个epoch就checkpoint一下
        # 修改成只保存early_stopping的时候效果最好的模型
        # if epoch % 10 == 0 and epoch != 0:
        #     checkpoint_tmp(model, save_name)
    min_loss_epoch = valid_loss_epoch.index(min(valid_loss_epoch))
    save_name += '-min_loss_epoch_{}'.format(min_loss_epoch)
    checkpoint_tmp(best_model, save_name)

        # 画出误差随着迭代次数的变化曲线
        # ep = ErrorPlot([train_loss_epoch, valid_loss_epoch], **hypara.__dict__)
        # ep.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # num_epoches
    parser.add_argument('-ne', '--num_epoches', type=int, required=True)
    # learning rate
    parser.add_argument('-lr', '--learning_rate', type=float, required=True)
    # learning_rate decay
    parser.add_argument('-ld', '--learning_rate_decay',
                        type=float, default=0.5)
    parser.add_argument('-bs', '--batch_size', type=int, required=True)
    

    hypara = parser.parse_args()
    num_epoch = hypara.num_epoches
    lr = hypara.learning_rate
    lr_decay = hypara.learning_rate_decay
    batch_size = hypara.batch_size

    save_name = ''
    for i, key in enumerate(hypara.__dict__):
        key_value_name = ('_' if i > 0 else '') + str(key) + \
            '-' + str(hypara.__dict__[key])
        save_name += key_value_name

    train(num_epoch, lr, save_name,batch_size=batch_size cuda_flag=True)
