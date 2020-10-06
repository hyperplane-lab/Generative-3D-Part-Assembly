'''
    for test of the trained model
    calculate the accuracy on test set
    v17 the same as v12
'''
import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')
from PIL import Image
from subprocess import call
from data_dynamic import PartNetPartDataset
import utils
import render_using_blender as render_utils
from quaternion import qrot
import ipdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))


def test(conf):
    data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'contact_points', 'sym', 'pairs', 'match_ids']
    # data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'contact_points']
    # data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'pairs']

    val_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.val_data_fn, data_features, \
                                     max_num_part=20, level=conf.level)
    #utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False,
                                                 pin_memory=True, \
                                                 num_workers=0, drop_last=True,
                                                 collate_fn=utils.collate_feats_with_none,
                                                 worker_init_fn=utils.worker_init_fn)
    
    model_def = utils.get_model_module(conf.model_version)
    network = model_def.Network(conf)
    network.load_state_dict(torch.load(conf.model_dir))
    #network = torch.load(conf.model_dir)
    #ipdb.set_trace()

    # utils.printout(conf.flog, '\n' + str(network) + '\n')

    models = [network]
    model_names = ['network']

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    optimizers = [network_opt]
    optimizer_names = ['network_opt']

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every,
                                                           gamma=conf.lr_decay_by)



    # send parameters to device
    for m in models:
        m.to(conf.device)
    for o in optimizers:
        utils.optimizer_to_device(o, conf.device)

    # start training
    start_time = time.time()

    #last_checkpoint_step = None
    last_val_console_log_step = None
    val_num_batch = len(val_dataloader)

    # train for every epoch
    #for in range(conf.epochs):
        # if not conf.no_console_log:
        #     utils.printout(conf.flog, f'training run {conf.exp_name}')
        #     utils.printout(conf.flog, header)

    val_batches = enumerate(val_dataloader, 0)
    val_fraction_done = 0.0
    val_batch_ind = -1
    
    sum_part_cd_loss = 0
    sum_shape_cd_loss = 0
    sum_contact_point_loss = 0
    total_acc_num = 0
    sum_resample_shape_cd_loss = 0
    total_valid_num = 0
    total_max_count = 0
    total_total_num = 0

    # validate one batch
    while val_batch_ind + 1 < val_num_batch:
        val_batch_ind, val_batch = next(val_batches)

        val_fraction_done = (val_batch_ind + 1) / val_num_batch

        if len(val_batch)==0:
            continue
        #val_step = (epoch + val_fraction_done) * train_num_batch - 1

        # log_console = not conf.no_console_log and (last_val_console_log_step is None or \
        #                                            val_step - last_val_console_log_step >= conf.console_log_interval)
        # if log_console:
        #     last_val_console_log_step = val_step

        # set models to evaluation mode
        for m in models:
            m.eval()
            
        #ipdb.set_trace()
        with torch.no_grad():
            # forward pass (including logging)

            part_cd_loss, shape_cd_loss, contact_point_loss, acc_num, valid_num, max_count, total_num = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                         batch_ind=val_batch_ind, num_batch=val_num_batch,
                         start_time=start_time, \
                         log_console=1, log_tb=not conf.no_tb_log, tb_writer=None,
                         lr=network_opt.param_groups[0]['lr'])
            
            sum_part_cd_loss += part_cd_loss
            sum_shape_cd_loss += shape_cd_loss
            sum_contact_point_loss += contact_point_loss
            total_acc_num += acc_num
            total_valid_num += valid_num 
            total_max_count += max_count
            total_total_num += total_num  
            
    total_max_count = total_max_count.float()
    total_total_num = float(total_total_num)
    total_shape_loss = sum_shape_cd_loss / val_num_batch
    total_part_loss = sum_part_cd_loss / val_num_batch
    total_contact_loss = sum_contact_point_loss / val_num_batch
    total_acc = total_acc_num / total_valid_num
    total_contact = total_max_count / total_total_num
    print('total_shape_loss:',total_shape_loss.item())
    print('total_part_loss:',total_part_loss.item())
    print('total_contact_loss:', total_contact_loss.item())
    print('total_acc:',100 * total_acc.item())
    print('total_contact', total_contact)
    print(total_max_count, total_total_num)
    
def forward(batch, data_features, network, conf, \
        is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
        log_console=False, log_tb=False, tb_writer=None, lr=None):
    # prepare input
    input_part_pcs = torch.cat(batch[data_features.index('part_pcs')], dim=0).to(conf.device)           # B x P x N x 3
    input_part_valids = torch.cat(batch[data_features.index('part_valids')], dim=0).to(conf.device)     # B x P
    input_part_pairs = torch.cat(batch[data_features.index('pairs')], dim=0).to(conf.device)
    batch_size = input_part_pcs.shape[0]
    num_part = input_part_pcs.shape[1]
    num_point = input_part_pcs.shape[2]
    part_ids = torch.cat(batch[data_features.index('part_ids')], dim=0).to(conf.device)      # B x P 
    match_ids=batch[data_features.index('match_ids')]  
    gt_part_poses = torch.cat(batch[data_features.index('part_poses')], dim=0).to(conf.device)      # B x P x (3 + 4)
    
    
    contact_points = torch.cat(batch[data_features.index("contact_points")], dim=0).to(conf.device)
    # input_part_pairs = torch.squeeze(contact_points[:, :, :, :1], dim=3)
    # cope with the sym_info
    sym_info = torch.cat(batch[data_features.index("sym")], dim=0)  # B x P x 3
    
    
    
    # get instance label
    instance_label = torch.zeros(batch_size, num_part, num_part).to(conf.device)
    same_class_list = []
    for i in range(batch_size):
        num_class = [ 0 for i in range(160) ]
        cur_same_class_list = [[] for i in range(160)]
        for j in range(num_part):
            cur_class = int(part_ids[i][j])
            if j < input_part_valids[i].sum():
                cur_same_class_list[cur_class].append(j)
            if cur_class == 0: continue
            cur_instance = int(num_class[cur_class])
            instance_label[i][j][cur_instance] = 1
            num_class[int(part_ids[i][j])] += 1
        for i in range(cur_same_class_list.count([])):
            cur_same_class_list.remove([])
        same_class_list.append(cur_same_class_list)

    repeat_times = 10
    array_trans_l2_loss_per_data = []
    array_rot_l2_loss_per_data = []
    array_rot_cd_loss_per_data = []
    array_total_cd_loss_per_data = []
    array_shape_cd_loss_per_data = []
    array_contact_point_loss_per_data = []
    array_acc = []
    array_pred_part_poses = []

    for repeat_ind in range(repeat_times):
        # forward through the network
        total_pred_part_poses = network(conf, input_part_pairs.float(), input_part_valids.float(),
                                        input_part_pcs.float(), instance_label, same_class_list)  # B x P x P, B x P, B x P x N x 3

        # for iter_ind in range(conf.iter):
        pred_part_poses = total_pred_part_poses[conf.iter - 1]
        # pred_part_poses = gt_part_poses
        array_pred_part_poses.append(pred_part_poses)

        # matching loss
        for ind in range(len(batch[0])):
            cur_match_ids = match_ids[ind]
            for i in range(1, 10):
                need_to_match_part = []
                for j in range(conf.max_num_part):
                    if cur_match_ids[j] == i:
                        need_to_match_part.append(j)
                if len(need_to_match_part) == 0: break
                cur_input_pts = input_part_pcs[ind, need_to_match_part]
                cur_pred_poses = pred_part_poses[ind, need_to_match_part]
                cur_pred_centers = cur_pred_poses[:, :3]
                cur_pred_quats = cur_pred_poses[:, 3:]
                cur_gt_part_poses = gt_part_poses[ind, need_to_match_part]
                cur_gt_centers = cur_gt_part_poses[:, :3]
                cur_gt_quats = cur_gt_part_poses[:, 3:]
                matched_pred_ids, matched_gt_ids = network.linear_assignment(cur_input_pts, cur_pred_centers,
                                                                             cur_pred_quats, cur_gt_centers,
                                                                             cur_gt_quats)
                match_pred_part_poses = pred_part_poses[ind, need_to_match_part][matched_pred_ids]
                pred_part_poses[ind, need_to_match_part] = match_pred_part_poses
                match_gt_part_poses = gt_part_poses[ind, need_to_match_part][matched_gt_ids]
                gt_part_poses[ind, need_to_match_part] = match_gt_part_poses

        # prepare gt
        input_part_pcs = input_part_pcs[:, :, :1000, :]
        # for each type of loss, compute losses per data
        trans_l2_loss_per_data = network.get_trans_l2_loss(pred_part_poses[:, :, :3], gt_part_poses[:, :, :3],
                                                           input_part_valids)  # B
        rot_l2_loss_per_data = network.get_rot_l2_loss(input_part_pcs, pred_part_poses[:, :, 3:],
                                                       gt_part_poses[:, :, 3:], input_part_valids)  # B
        rot_cd_loss_per_data = network.get_rot_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:],
                                                       gt_part_poses[:, :, 3:], input_part_valids, conf.device)  # B
        # # for each type of loss, compute avg loss per batch
        # trans_l2_loss = trans_l2_loss_per_data.mean()
        # rot_l2_loss = rot_l2_loss_per_data.mean()
        # rot_cd_loss = rot_cd_loss_per_data.mean()
        # # compute total loss
        # if iter_ind == 0:
        #     total_loss =    trans_l2_loss * conf.loss_weight_trans_l2 + \
        #                     rot_l2_loss * conf.loss_weight_rot_l2 + \
        #                     rot_cd_loss * conf.loss_weight_rot_cd
        #     total_trans_l2_loss = trans_l2_loss
        #     total_rot_l2_loss = rot_l2_loss
        #     total_rot_cd_loss = rot_cd_loss
        # else:
        #     total_loss +=    trans_l2_loss * conf.loss_weight_trans_l2 + \
        #                 rot_l2_loss * conf.loss_weight_rot_l2 + \
        #                 rot_cd_loss * conf.loss_weight_rot_cd
        #     total_trans_l2_loss += trans_l2_loss
        #     total_rot_l2_loss += rot_l2_loss
        #     total_rot_cd_loss += rot_cd_loss

        # prepare gt
        input_part_pcs = input_part_pcs[:, :, :1000, :]
        # if iter_ind == 2:
        total_cd_loss_per_data, acc = network.get_total_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:],
                                                                gt_part_poses[:, :, 3:],
                                                                input_part_valids, pred_part_poses[:, :, :3],
                                                                gt_part_poses[:, :, :3], conf.device)  # B)
        # total_cd_loss = total_cd_loss_per_data.mean()
        shape_cd_loss_per_data = network.get_shape_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:],
                                                           gt_part_poses[:, :, 3:],
                                                           input_part_valids, pred_part_poses[:, :, :3],
                                                           gt_part_poses[:, :, :3], conf.device)
        # shape_cd_loss = shape_cd_loss_per_data.mean()
        contact_point_loss_per_data, count, total_num = network.get_contact_point_loss(pred_part_poses[:, :, :3],
                                                                     pred_part_poses[:, :, 3:], contact_points, sym_info)

        array_trans_l2_loss_per_data.append(trans_l2_loss_per_data)
        array_rot_l2_loss_per_data.append(rot_l2_loss_per_data)
        array_rot_cd_loss_per_data.append(rot_cd_loss_per_data)
        array_total_cd_loss_per_data.append(total_cd_loss_per_data)
        array_shape_cd_loss_per_data.append(shape_cd_loss_per_data)
        array_contact_point_loss_per_data.append(contact_point_loss_per_data)
        # B x P -> B
        acc = torch.tensor(acc)
        acc = acc.sum(-1).float()  # B
        valid_number = input_part_valids.sum(-1).float().cpu()  # B
        acc_rate = acc / valid_number
        array_acc.append(acc_rate)
        count = torch.tensor(count)

        if repeat_ind == 0:
            res_total_cd = total_cd_loss_per_data
            res_shape_cd = shape_cd_loss_per_data
            res_contact_point = contact_point_loss_per_data
            res_acc = acc
            res_count = count
        else:
            res_total_cd = res_total_cd.min(total_cd_loss_per_data)
            res_shape_cd = res_shape_cd.min(shape_cd_loss_per_data)
            res_contact_point = res_contact_point.min(contact_point_loss_per_data)
            res_acc = res_acc.max(acc)  # B
            res_count = res_count.max(count)

    shape_cd_loss = res_shape_cd.mean()
    total_cd_loss = res_total_cd.mean()
    contact_point_loss = res_contact_point.mean()
    acc_num = res_acc.sum()  # how many parts are right in total in a certain batch
    valid_num = input_part_valids.sum()  # how many parts in total in a certain batch

    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'
    with torch.no_grad():
        # gen visu
        is_val = False
        if is_val and (not conf.no_visu):
            visu_dir = os.path.join(conf.exp_dir, 'val_visu')
            out_dir = os.path.join(visu_dir, 'test_196')
            input_part_pcs_dir = os.path.join(out_dir, 'input_part_pcs')
            gt_assembly_dir = os.path.join(out_dir, 'gt_assembly')
            pred_assembly_dir = os.path.join(out_dir, 'pred_assembly')
            info_dir = os.path.join(out_dir, 'info')

            if batch_ind == 0:
                # create folders
                os.mkdir(out_dir)
                os.mkdir(input_part_pcs_dir)
                os.mkdir(gt_assembly_dir)
                os.mkdir(pred_assembly_dir)
                os.mkdir(info_dir)

            if batch_ind < conf.num_batch_every_visu:
                #utils.printout(conf.flog, 'Visualizing ...')

                for repeat_ind in range(repeat_times):
                    pred_center = array_pred_part_poses[repeat_ind][:, :, :3]
                    gt_center = gt_part_poses[:, :, :3]

                    # compute pred_pts and gt_pts
                    # import ipdb; ipdb.set_trace()

                    pred_pts = qrot(array_pred_part_poses[repeat_ind][:, :, 3:].unsqueeze(2).repeat(1, 1, num_point, 1),
                                    input_part_pcs) + pred_center.unsqueeze(2).repeat(1, 1, num_point, 1)
                    gt_pts = qrot(gt_part_poses[:, :, 3:].unsqueeze(2).repeat(1, 1, num_point, 1),
                                  input_part_pcs) + gt_center.unsqueeze(2).repeat(1, 1, num_point, 1)

                    for i in range(batch_size):
                        fn = 'data-%03d-%03d.png' % (batch_ind * batch_size + i, repeat_ind)

                        cur_input_part_cnt = input_part_valids[i].sum().item()
                        # print(cur_input_part_cnt)
                        cur_input_part_cnt = int(cur_input_part_cnt)
                        cur_input_part_pcs = input_part_pcs[i, :cur_input_part_cnt]
                        cur_gt_part_poses = gt_part_poses[i, :cur_input_part_cnt]
                        cur_pred_part_poses = array_pred_part_poses[repeat_ind][i, :cur_input_part_cnt]

                        pred_part_pcs = qrot(cur_pred_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1),
                                             cur_input_part_pcs) + \
                                        cur_pred_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)
                        gt_part_pcs = qrot(cur_gt_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1),
                                           cur_input_part_pcs) + \
                                      cur_gt_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

                        part_pcs_to_visu = cur_input_part_pcs.cpu().detach().numpy()
                        render_utils.render_part_pts(os.path.join(BASE_DIR, input_part_pcs_dir, fn), part_pcs_to_visu,
                                                     blender_fn='object_centered.blend')
                        part_pcs_to_visu = pred_part_pcs.cpu().detach().numpy()
                        render_utils.render_part_pts(os.path.join(BASE_DIR, pred_assembly_dir, fn), part_pcs_to_visu,
                                                     blender_fn='object_centered.blend')
                        part_pcs_to_visu = gt_part_pcs.cpu().detach().numpy()
                        render_utils.render_part_pts(os.path.join(BASE_DIR, gt_assembly_dir, fn), part_pcs_to_visu,
                                                     blender_fn='object_centered.blend')

                        with open(os.path.join(info_dir, fn.replace('.png', '.txt')), 'w') as fout:
                            fout.write('shape_id: %s\n' % batch[data_features.index('shape_id')][i])
                            fout.write('num_part: %d\n' % cur_input_part_cnt)
                            fout.write('trans_l2_loss: %f\n' % array_trans_l2_loss_per_data[repeat_ind][i].item())
                            fout.write('rot_l2_loss: %f\n' % array_rot_l2_loss_per_data[repeat_ind][i].item())
                            fout.write('rot_cd_loss: %f\n' % array_rot_cd_loss_per_data[repeat_ind][i].item())
                            fout.write('total_cd_loss: %f\n' % array_total_cd_loss_per_data[repeat_ind][i].item())
                            fout.write('shape_cd_loss: %f\n' % array_shape_cd_loss_per_data[repeat_ind][i].item())
                            fout.write('contact_point_loss: %f\n' % array_contact_point_loss_per_data[repeat_ind][i].item())
                            fout.write('part_accuracy: %f\n' % array_acc[repeat_ind][i].item())

            # if batch_ind == conf.num_batch_every_visu - 1:
            #     # visu html
            #     utils.printout(conf.flog, 'Generating html visualization ...')
            #     sublist = 'input_part_pcs,gt_assembly,pred_assembly,info'
            #     cmd = 'cd %s && python %s . 10 htmls %s %s > /dev/null' % (out_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierarchy_local.py'), sublist, sublist)
            #     call(cmd, shell=True)
            #     utils.printout(conf.flog, 'DONE')

    return total_cd_loss, shape_cd_loss, contact_point_loss, acc_num, valid_num, res_count, total_num
   
   




  
      




if __name__ == '__main__':

    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--train_data_fn', type=str, help='training data file that indexs all data tuples')
    parser.add_argument('--val_data_fn', type=str, help='validation data file that indexs all data tuples')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3124256514,
                        help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    # parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--data_dir', type=str, default='../../prepare_data', help='data directory')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--feat_len', type=int, default=256)
    parser.add_argument('--max_num_part', type=int, default=20)

    # loss weights
    parser.add_argument('--loss_weight_trans_l2', type=float, default=1.0, help='loss weight')
    parser.add_argument('--loss_weight_rot_l2', type=float, default=1.0, help='loss weight')
    parser.add_argument('--loss_weight_rot_cd', type=float, default=10.0, help='loss weight')

    # training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--iter', default = 5, help = 'times to iteration')
    parser.add_argument('--iter_to_test', default = 4, help = 'times to iteration')


    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10,
                        help='number of optimization steps beween console log prints')
    parser.add_argument('--checkpoint_interval', type=int, default=10000,
                        help='number of optimization steps beween checkpoints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=1, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # data
    parser.add_argument('--level', default='3', help='level of dataset')

    #model path
    parser.add_argument('--model_dir', type=str, help='the path of the model')

    # parse args
    conf = parser.parse_args()
    

    conf.exp_name = f'exp-{conf.category}-{conf.model_version}-level{conf.level}{conf.exp_suffix}'
    # conf.exp_name = f'exp-{conf.category}-{conf.model_version}-{conf.train_data_fn.split(".")[0]}-{conf.exp_suffix}'

    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    
    #flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    #conf.flog = flog
    
    print("conf", conf)

    ### start training
    test(conf)
