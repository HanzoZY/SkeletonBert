import os
import sys
from ipdb import set_trace
print('Python %s on %s' % (sys.version, sys.platform))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.extend(['../'])
print(sys.path)
from random import randint
import torch
import pickle
from train_val_test import train_val_model, parser_args
from utility.log import TimerBlock, IteratorTimer
from method_choose.data_choose import data_choose, init_seed
from method_choose.model_choose import model_choose
from method_choose.loss_choose import loss_choose
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import numpy as np
import argparse
import yaml
import shutil
from easydict import EasyDict as ed
import colorama
edge_dict = {
    'shrec_skeleton' : ((0, 1),
        (1, 2), (2, 3), (3, 4), (4, 5),
        (1, 6), (6, 7), (7, 8), (8, 9),
        (1, 10), (10, 11), (11, 12), (12, 13),
        (1, 14), (14, 15), (15, 16), (16, 17),
        (1, 18), (18, 19), (19, 20), (20, 21)),
    'ntu_skeleton' : ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
        (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
        (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)),
    'kinetics_skeleton' : ((4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14))
}

# def parser_args_eval(config_path=None):
#     # params
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-config', default='./config/val/shrec_dstanet_14.yaml')
#     parser.add_argument('-model', default='resnet3d_50')
#     parser.add_argument('-model_param', default={}, help=None)
#     # classify_multi_crop classify classify_pose
#     parser.add_argument('-train', default='classify')
#     parser.add_argument('-val_first', default=False)
#     parser.add_argument('-data', default='jmdbgulp')
#     parser.add_argument('-data_param', default={}, help='')
#     # train_val test train_test
#     parser.add_argument('-mode', default='train_val')
#     # cross_entropy mse_ce
#     parser.add_argument('-loss', default='cross_entropy')
#     parser.add_argument('-ls_param', default={
#     })
#     # reduce_by_acc reduce_by_loss reduce_by_epoch cosine_annealing_lr
#     parser.add_argument('-lr_scheduler', default='reduce_by_acc')
#     parser.add_argument('-lr_param', default={})
#     parser.add_argument('-warm_up_epoch', default=0)
#     parser.add_argument('-step', default=[80, ])
#     parser.add_argument('-lr', default=0.01)  # 0.001
#     parser.add_argument('-wd', default=1e-4)  # 5e-4
#     parser.add_argument('-lr_decay_ratio', default=0.1)
#     parser.add_argument('-lr_multi_keys', default=[
#         ['fc', 1, 1, 0], ['bn', 1, 1, 0],
#     ], help='key, lr ratio, wd ratio, epoch')
#     parser.add_argument('-optimizer', default='sgd_nev')
#     parser.add_argument('-freeze_keys', default=[
#         ['PA', 5],
#     ], help='key, epoch')
#
#     parser.add_argument('-class_num', default=12)
#     parser.add_argument('-batch_size', default=32)
#     parser.add_argument('-worker', default=16)
#     parser.add_argument('-pin_memory', default=False)
#     parser.add_argument('-max_epoch', default=50)
#
#     parser.add_argument('-num_epoch_per_save', default=2)
#     parser.add_argument('-model_saved_name', default='')
#     parser.add_argument('-last_model', default=None, help='')
#     parser.add_argument('-ignore_weights', default=['fc'])
#     parser.add_argument('-pre_trained_model', default='')
#     parser.add_argument('--label_smoothing_num', default=0, help='0-1: 0 denotes no smoothing')
#     parser.add_argument('--mix_up_num', default=0, help='0-1: 1 denotes uniform distribution, smaller, more concave')
#     parser.add_argument('-device_id', default=[0, 1, 2, 3])
#     parser.add_argument('-debug', default=False)
#     parser.add_argument('-cuda_visible_device', default='0, 1, 2, 3, 4, 5, 6, 7')
#     parser.add_argument('-grad_clip', default=0)
#     p = parser.parse_args()
#     if config_path is not None:
#         p.config = config_path
#     if p.config is not None:
#         with open(p.config, 'r') as f:
#             default_arg = yaml.load(f)
#         key = vars(p).keys()
#         for k in default_arg.keys():
#             if k not in key:
#                 print('WRONG ARG: {}'.format(k))
#                 assert (k in key)
#         parser.set_defaults(**default_arg)
#
#     args = parser.parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_device
#
#     if args.debug:
#         args.device_id = [0]
#         args.batch_size = 1
#         args.worker = 0
#     args = ed(vars(args))
#     return args

def get_label(data_set='ntu_60'):
    if data_set == 'ntu_60':
        labels = open('../prepare/ntu_60/label.txt', 'r').readlines()
    elif data_set == 'ntu_120':
        labels = open('../prepare/ntu_120/label.txt', 'r').readlines()
    elif data_set == 'knitics':
        raise ValueError("not support yet")
    elif data_set == 'shrec_14':
        labels = open('../prepare/shrec/label.txt', 'r').readlines()
    else:
        labels = open('../prepare/shrec/label_28.txt', 'r').readlines()

    for i in range(len(labels)):
        labels[i] = labels[i].strip('\n')

    return labels

def vis_all(data, edge=None, is_3d=False, tag='', pause=0.01):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('macosx')
    N, C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    # add label
    fig.suptitle(tag)

    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(edge) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    pose = []
    for m in range(M):
        a = []
        for i in range(len(edge)):
            # if i==3:
            #     # color = 'y-'
            #     color = colors[i]
            # else:
            #     color = 'k-'
            color = colors[i]
            if is_3d:
                a.append(ax.plot(np.zeros(3), np.zeros(3), color)[0])
            else:
                a.append(ax.plot(np.zeros(2), np.zeros(2), color)[0])
        # for i in range(len(edge)):
        #     if is_3d:
        #         a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
        #     else:
        #         a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
        pose.append(a)
    ax.axis([-1, 1, -1, 1])
    if is_3d:
        ax.set_zlim3d(-1, 1)
    plt.axis('off')
    # while True:
    for t in range(T):
        for m in range(M):
            for i, (v1, v2) in enumerate(edge):
                x1 = data[0, :2, t, v1, m]
                x2 = data[0, :2, t, v2, m]
                pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                if is_3d: pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                # if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                #     pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                #     pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                #     if is_3d:
                #         pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])

        fig.canvas.draw()
        # if t % 2 == 0 and t <= 58:
        #     # plt.savefig('./skeleton_sequence/' + str(t) + '.eps', dpi=300,format='eps')
        #     plt.savefig('./skeleton_sequence/' + str(t) + '.png', dpi=300, format='png')
        plt.pause(pause)
    plt.close()
    plt.ioff()




def vis_all_color(data, edge=None, is_3d=False, tag='', pause=0.01):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('macosx')
    N, C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    # add label
    fig.suptitle(tag)

    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(edge) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    pose = []
    for m in range(M):
        a = []
        for i in range(len(edge)):
            b = []
            if is_3d:
                b.append(ax.plot(np.zeros(3), np.zeros(3), colors[i])[0])
                b.append(ax.scatter(0.0, 0.0, 0.0, c=colors[i], marker='o'))
                b.append(ax.scatter(0.0, 0.0, 0.0, c=colors[i], marker='o'))
            else:
                b.append(ax.plot(np.zeros(2), np.zeros(2), colors[i])[0])
                b.append(ax.scatter(0.0, 0.0, c=colors[i], marker='o'))
                b.append(ax.scatter(0.0, 0.0, c=colors[i], marker='o'))
            a.append(b)
        pose.append(a)
    ax.axis([-1, 1, -1, 1])
    if is_3d:
        ax.set_zlim3d(-1, 1)
    plt.axis('on')
    # while True:
    for t in range(T):
        for m in range(M):
            for i, (v1, v2) in enumerate(edge):
                pose[m][i][0].set_xdata(data[0, 0, t, [v1, v2], m])
                pose[m][i][0].set_ydata(data[0, 1, t, [v1, v2], m])
                pose[m][i][1].set_offsets(np.array([data[0, 0, t, v1, m], data[0, 1, t, v1, m]]))
                pose[m][i][2].set_offsets(np.array([data[0, 0, t, v2, m], data[0, 1, t, v2, m]]))
                if is_3d:
                    pose[m][i][0].set_3d_properties(data[0, 2, t, [v1, v2], m])
                    pose[m][i][1].set_3d_properties(data[0, 2, t, v1, m], zdir='z')
                    pose[m][i][2].set_3d_properties(data[0, 2, t, v2, m], zdir='z')

        fig.canvas.draw()
        # if t % 2 == 0 and t <= 58:
        # #     pass
        #     # plt.savefig('./skeleton_sequence/' + str(t) + '.eps', dpi=300,format='eps')
        #     plt.savefig('./skeleton_sequence/' + str(t) + '.png', dpi=300, format='png')
        plt.pause(pause)
    plt.close()
    plt.ioff()




def eval_vis_model(vid, idx_shotcut=None, config_path=None, pause=0.01, view=0.25):
    import matplotlib.pyplot as plt
    import matplotlib
    with TimerBlock("Good Luck") as block:

        args = parser_args.parser_args(block, config_path=config_path)
        init_seed(1)
        edge_chose = args.data
        if args.data == 'knitics':
            is_3d = False
        else:
            is_3d = True
        edge = edge_dict[edge_chose]
        data_loader_train, data_loader_val = data_choose(args, block)
        # set_trace()
        global_step, start_epoch, model, optimizer_dict = model_choose(args, block)
        lables = get_label()
        if idx_shotcut is not None:
            index = idx_shotcut
        else:
            sample_name = data_loader_val.dataset.sample_name
            sample_id = [name.split('.')[0] for name in sample_name]
            index = sample_id.index(vid)
        data, label, index = data_loader_val.dataset[index]
        tgt = lables[label]
        if type(args.device_id) is list and len(args.device_id) > 0:
            model.cuda()
        else:
            pass
        model.eval()
        data = torch.tensor(data).unsqueeze(0)
        outputs, pretext_loss = model(data)
        if len(outputs.data.shape) == 3:  # T N cls
            _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
        else:
            _, predict_label = torch.max(outputs.data, 1)
        predict_label = predict_label.squeeze()
        print('predict_label:',predict_label)
        print('tgt:',label)
        predict_label_tag = lables[predict_label]
        tag = 'Pred: ' + predict_label_tag + '\n' + 'TGT:' + tgt
        # vis_all(data=data.numpy(), edge=edge, is_3d=is_3d, tag=tag, pause=pause)
        vis_all_color(data=data.numpy(), edge=edge, is_3d=is_3d, tag=tag, pause=pause)












if __name__ == '__main__':
    # config_path = './work_dir/ntu60/dstanet_temporalattTrueThenTCN_PEAtInput_SpatialEmbdedding_TempEncoding_2/ntu60_dstanet.yaml'
    # negative_samples = open('./work_dir/ntu60/dstanet_temporalattTrueThenTCN_PEAtInput_SpatialEmbdedding_TempEncoding_2/wrong_path_pre_true.txt', 'r').readlines()
    config_path = './work_dir/ntu60/dstanet_temporalattTrueThenTCN_PEAtInput_SpatialEmbdedding_TempEncoding_2/ntu60_dstanet.yaml'
    negative_samples = open('./work_dir/ntu60/dstanet_temporalattTrueThenTCN_PEAtInput_SpatialEmbdedding_TempEncoding_2/wrong_path_pre_true.txt', 'r').readlines()
    num_neg = len(negative_samples)
    negative_samples_dict = {}
    for i in range(len(negative_samples)):
        n_id = negative_samples[i].split('.')[0]

    # idx = randint(0, num_neg)
    idx = 3
    vid = negative_samples[idx].split('.')[0]
    eval_vis_model(vid=vid, config_path=config_path)

