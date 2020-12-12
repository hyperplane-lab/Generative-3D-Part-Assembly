"""
    Training models
"""

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
from subprocess import call
from data_dynamic import PartNetPartDataset
import utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import render_using_blender as render_utils
from quaternion import qrot


def train(conf):
    # create training and validation datasets and data loaders
    data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'pairs']
    
    train_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.train_data_fn, data_features, \
            max_num_part=conf.max_num_part, level=conf.level)
    utils.printout(conf.flog, str(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
            num_workers=conf.num_workers, drop_last=True, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)
    
    val_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.val_data_fn, data_features, \
            max_num_part=conf.max_num_part,level=conf.level)
    utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, \
            num_workers=0, drop_last=True, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)

    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.Network(conf)

    utils.printout(conf.flog, '\n' + str(network) + '\n')

    models = [network]
    model_names = ['network']

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    optimizers = [network_opt]
    optimizer_names = ['network_opt']

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TransL2Loss    RotL2Loss   RotCDLoss  ShapeCDLoss   TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'val'))

    # send parameters to device
    for m in models:
        m.to(conf.device)
    for o in optimizers:
        utils.optimizer_to_device(o, conf.device)

    # start training
    start_time = time.time()

    last_checkpoint_step = None
    last_train_console_log_step, last_val_console_log_step = None, None
    train_num_batch = len(train_dataloader)
    val_num_batch = len(val_dataloader)

    # train for every epoch
    for epoch in range(conf.epochs):
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)

        val_batches = enumerate(val_dataloader, 0)
        train_fraction_done = 0.0
        val_fraction_done = 0.0
        val_batch_ind = -1
        val_flag = 0 # to record whether it is the first time


        # train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # set models to training mode
            for m in models:
                m.train()

            # forward pass (including logging)
            if len(batch)==0:continue
            total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss = forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                    step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                    log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])

            #to sum the training loss of all categories
            if train_batch_ind == 0:
                sum_total_trans_l2_loss = total_trans_l2_loss
                sum_total_rot_l2_loss = total_rot_l2_loss
                sum_total_rot_cd_loss = total_rot_cd_loss
                sum_total_shape_cd_loss = total_shape_cd_loss
            else:
                sum_total_trans_l2_loss += total_trans_l2_loss
                sum_total_rot_l2_loss += total_rot_l2_loss
                sum_total_rot_cd_loss += total_rot_cd_loss
                sum_total_shape_cd_loss += total_shape_cd_loss
            #optimize one step
            network_lr_scheduler.step()
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()

            # save checkpoint
            with torch.no_grad():
                if last_checkpoint_step is None or train_step - last_checkpoint_step >= conf.checkpoint_interval:
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.exp_dir, 'ckpts'), \
                            epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=model_names)
                    utils.printout(conf.flog, 'DONE')
                    last_checkpoint_step = train_step

            # validate one batch
            while val_fraction_done <= train_fraction_done and val_batch_ind+1 < val_num_batch:
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_val_console_log_step is None or \
                        val_step - last_val_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_val_console_log_step = val_step

                # set models to evaluation mode
                for m in models:
                    m.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    if len(val_batch)==0:continue
                    total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                            step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                            log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer, lr=network_opt.param_groups[0]['lr'])
                    # to sum the validating loss of all categories
                    if val_flag == 0:
                        val_total_trans_l2_loss = total_trans_l2_loss
                        val_total_rot_l2_loss = total_rot_l2_loss
                        val_total_rot_cd_loss = total_rot_cd_loss
                        val_total_shape_cd_loss = total_shape_cd_loss
                        val_flag = 1
                    else:
                        val_total_trans_l2_loss += total_trans_l2_loss
                        val_total_rot_l2_loss += total_rot_l2_loss
                        val_total_shape_cd_loss += total_shape_cd_loss
                        val_total_rot_cd_loss += total_rot_cd_loss

        #using tensorboard to record the losses for each epoch
        with torch.no_grad():
            if not conf.no_tb_log and train_writer is not None:
                train_writer.add_scalar('sum_total_trans_l2_loss', sum_total_trans_l2_loss.item(), epoch)
                train_writer.add_scalar('sum_total_rot_l2_loss', sum_total_rot_l2_loss.item(), epoch)
                train_writer.add_scalar('sum_total_rot_cd_loss', sum_total_rot_cd_loss.item(), epoch)
                train_writer.add_scalar('sum_total_shape_cd_loss', sum_total_shape_cd_loss.item(), epoch)
                train_writer.add_scalar('lr', network_opt.param_groups[0]['lr'], epoch)
            if not conf.no_tb_log and val_writer is not None:
                val_writer.add_scalar('val_total_trans_l2_loss', val_total_trans_l2_loss.item(), epoch)
                val_writer.add_scalar('val_total_rot_l2_loss', val_total_rot_l2_loss.item(), epoch)
                val_writer.add_scalar('val_total_rot_cd_loss', val_total_rot_cd_loss.item(), epoch)
                val_writer.add_scalar('val_total_shape_cd_loss', val_total_shape_cd_loss.item(), epoch)
                val_writer.add_scalar('lr', network_opt.param_groups[0]['lr'], epoch)
    # save the final models
    utils.printout(conf.flog, 'Saving final checkpoint ...... ')
    utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.exp_dir, 'ckpts'), \
            epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)
    utils.printout(conf.flog, 'DONE')


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
    # forward through the network

    repeat_times = 5
    for repeat_ind in range(repeat_times):
        total_pred_part_poses = network(conf, input_part_pairs.float(), input_part_valids.float(), input_part_pcs.float(), instance_label, same_class_list)     # B x P x P, B x P, B x P x N x 3

        for iter_ind in range(conf.iter):
            pred_part_poses = total_pred_part_poses[iter_ind]

            # matching loss
            for ind in range(len(batch[0])):
                cur_match_ids = match_ids[ind]
                for i in range(1,10):
                    need_to_match_part = []
                    for j in range(conf.max_num_part):
                        if cur_match_ids[j] == i:
                            need_to_match_part.append(j)
                    if len(need_to_match_part) == 0:break
                    cur_input_pts = input_part_pcs[ind,need_to_match_part]
                    cur_pred_poses = pred_part_poses[ind,need_to_match_part]
                    cur_pred_centers = cur_pred_poses[:,:3]
                    cur_pred_quats = cur_pred_poses[:,3:]
                    cur_gt_part_poses = gt_part_poses[ind,need_to_match_part]
                    cur_gt_centers = cur_gt_part_poses[:,:3]
                    cur_gt_quats = cur_gt_part_poses[:,3:]
                    matched_pred_ids , matched_gt_ids = network.linear_assignment(cur_input_pts, cur_pred_centers, cur_pred_quats, cur_gt_centers, cur_gt_quats)
                    match_pred_part_poses = pred_part_poses[ind,need_to_match_part][matched_pred_ids]
                    pred_part_poses[ind,need_to_match_part] = match_pred_part_poses
                    match_gt_part_poses = gt_part_poses[ind,need_to_match_part][matched_gt_ids]
                    gt_part_poses[ind,need_to_match_part] = match_gt_part_poses

            # prepare gt
            input_part_pcs = input_part_pcs[:, :, :1000, :]
            # for each type of loss, compute losses per data
            trans_l2_loss_per_data = network.get_trans_l2_loss(pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], input_part_valids)  # B
            rot_l2_loss_per_data = network.get_rot_l2_loss(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids)      # B
            rot_cd_loss_per_data = network.get_rot_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)      # B
            shape_cd_loss_per_data = network.get_shape_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:],
                                        input_part_valids, pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], conf.device)

            # for each type of loss, compute avg loss per batch
            shape_cd_loss = shape_cd_loss_per_data.mean()
            trans_l2_loss = trans_l2_loss_per_data.mean()
            rot_l2_loss = rot_l2_loss_per_data.mean()
            rot_cd_loss = rot_cd_loss_per_data.mean()
            # compute total loss
            if iter_ind == 0:
                total_loss =    trans_l2_loss * conf.loss_weight_trans_l2 + \
                                rot_l2_loss * conf.loss_weight_rot_l2 + \
                                rot_cd_loss * conf.loss_weight_rot_cd + \
                                shape_cd_loss * conf.loss_weight_shape_cd
                total_shape_cd_loss = shape_cd_loss
                total_trans_l2_loss = trans_l2_loss
                total_rot_l2_loss = rot_l2_loss
                total_rot_cd_loss = rot_cd_loss
            elif iter_ind == conf.iter - 1:
                total_loss += trans_l2_loss * conf.loss_weight_trans_l2 + \
                              rot_l2_loss * conf.loss_weight_rot_l2 + \
                              rot_cd_loss * conf.loss_weight_rot_cd + \
                              shape_cd_loss * conf.loss_weight_shape_cd
                total_shape_cd_loss += shape_cd_loss
                total_trans_l2_loss += trans_l2_loss
                total_rot_l2_loss += rot_l2_loss
                total_rot_cd_loss += rot_cd_loss
                if repeat_ind == 0:
                    res_loss = total_loss
                    res_shape_cd_loss = total_shape_cd_loss
                    res_trans_l2_loss = total_trans_l2_loss
                    res_rot_l2_loss = total_rot_l2_loss
                    res_rot_cd_loss = total_rot_cd_loss
                else:
                    res_loss = res_loss.min(total_loss)
                    res_shape_cd_loss = res_shape_cd_loss.min(total_shape_cd_loss)
                    res_trans_l2_loss = res_trans_l2_loss.min(total_trans_l2_loss)
                    res_rot_l2_loss = res_rot_l2_loss.min(total_rot_l2_loss)
                    res_rot_cd_loss = res_rot_cd_loss.min(total_rot_cd_loss)
            else:
                total_loss += trans_l2_loss * conf.loss_weight_trans_l2 + \
                              rot_l2_loss * conf.loss_weight_rot_l2 + \
                              rot_cd_loss * conf.loss_weight_rot_cd + \
                              shape_cd_loss * conf.loss_weight_shape_cd
                total_shape_cd_loss += shape_cd_loss
                total_trans_l2_loss += trans_l2_loss
                total_rot_l2_loss += rot_l2_loss
                total_rot_cd_loss += rot_cd_loss

    total_loss = res_loss
    total_trans_l2_loss = res_trans_l2_loss
    total_rot_l2_loss = res_rot_l2_loss
    total_rot_cd_loss = res_rot_cd_loss
    total_shape_cd_loss = res_shape_cd_loss

    data_split = 'train'
    if is_val:
        data_split = 'val'

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog, \
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{data_split:^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{trans_l2_loss.item():>10.5f}   '''
                f'''{rot_l2_loss.item():>10.5f}   '''
                f'''{rot_cd_loss.item():>10.5f}  '''
                f'''{shape_cd_loss.item():>10.5f}  '''
                f'''{total_loss.item():>10.5f}  '''
                )
            conf.flog.flush()

        # gen visu
        if is_val and (not conf.no_visu) and epoch % conf.num_epoch_every_visu == conf.num_epoch_every_visu - 1:
            visu_dir = os.path.join(conf.exp_dir, 'val_visu')
            out_dir = os.path.join(visu_dir, 'epoch-%04d' % epoch)
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
                utils.printout(conf.flog, 'Visualizing ...')
                pred_center = pred_part_poses[:, :, :3]
                gt_center = gt_part_poses[:, :, :3]

                # compute pred_pts and gt_pts
                
                pred_pts = qrot(pred_part_poses[:, :, 3:].unsqueeze(2).repeat(1, 1, num_point, 1), input_part_pcs) + pred_center.unsqueeze(2).repeat(1, 1, num_point, 1)
                gt_pts = qrot(gt_part_poses[:, :, 3:].unsqueeze(2).repeat(1, 1, num_point, 1), input_part_pcs) + gt_center.unsqueeze(2).repeat(1, 1, num_point, 1)

                for i in range(batch_size):
                    fn = 'data-%03d.png' % (batch_ind * batch_size + i)
                    
                    cur_input_part_cnt = input_part_valids[i].sum().item()
                    cur_input_part_cnt = int(cur_input_part_cnt)
                    cur_input_part_pcs = input_part_pcs[i, :cur_input_part_cnt]
                    cur_gt_part_poses = gt_part_poses[i, :cur_input_part_cnt]
                    cur_pred_part_poses = pred_part_poses[i, :cur_input_part_cnt]

                    pred_part_pcs = qrot(cur_pred_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_part_pcs) + \
                            cur_pred_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)
                    gt_part_pcs = qrot(cur_gt_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_part_pcs) + \
                            cur_gt_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

                    part_pcs_to_visu = cur_input_part_pcs.cpu().detach().numpy()
                    render_utils.render_part_pts(os.path.join(BASE_DIR, input_part_pcs_dir, fn), part_pcs_to_visu, blender_fn='object_centered.blend')
                    part_pcs_to_visu = pred_part_pcs.cpu().detach().numpy()
                    render_utils.render_part_pts(os.path.join(BASE_DIR, pred_assembly_dir, fn), part_pcs_to_visu, blender_fn='object_centered.blend')
                    part_pcs_to_visu = gt_part_pcs.cpu().detach().numpy()
                    render_utils.render_part_pts(os.path.join(BASE_DIR, gt_assembly_dir, fn), part_pcs_to_visu, blender_fn='object_centered.blend')

                    with open(os.path.join(info_dir, fn.replace('.png', '.txt')), 'w') as fout:
                        fout.write('shape_id: %s\n' % batch[data_features.index('shape_id')][i])
                        fout.write('num_part: %d\n' % cur_input_part_cnt)
                        fout.write('trans_l2_loss: %f\n' % trans_l2_loss_per_data[i].item())
                        fout.write('rot_l2_loss: %f\n' % rot_l2_loss_per_data[i].item())
                        fout.write('rot_cd_loss: %f\n' % rot_cd_loss_per_data[i].item())
                
            if batch_ind == conf.num_batch_every_visu - 1:
                # visu html
                utils.printout(conf.flog, 'Generating html visualization ...')
                sublist = 'input_part_pcs,gt_assembly,pred_assembly,info'
                cmd = 'cd %s && python %s . 10 htmls %s %s > /dev/null' % (out_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierarchy_local.py'), sublist, sublist)
                call(cmd, shell=True)
                utils.printout(conf.flog, 'DONE')

    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss



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
    parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    #parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--data_dir', type=str, default='../../prep_data', help='data directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--level',type=str,default='3',help='level of dataset')

    # network settings
    parser.add_argument('--feat_len', type=int, default=256)
    parser.add_argument('--max_num_part', type=int, default=20)

    # training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--iter', default = 5, type=int, help = 'times to iteration')

    # loss weights
    parser.add_argument('--loss_weight_trans_l2', type=float, default=1.0, help='loss weight')
    parser.add_argument('--loss_weight_rot_l2', type=float, default=1.0, help='loss weight')
    parser.add_argument('--loss_weight_rot_cd', type=float, default=10.0, help='loss weight')
    parser.add_argument('--loss_weight_shape_cd', type=float, default=1.0, help='loss weight')


    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')
    parser.add_argument('--checkpoint_interval', type=int, default=300, help='number of optimization steps beween checkpoints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=1, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # parse args
    conf = parser.parse_args()
    print("conf", conf)


    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.category}-{conf.model_version}-level{conf.level}{conf.exp_suffix}'
    
    # mkdir exp_dir; ask for overwrite if necessary
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    if not os.path.exists(conf.log_dir):
       os.mkdir(conf.log_dir) 
    if os.path.exists(conf.exp_dir):
        if not conf.overwrite:
            response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
            if response != 'y':
                exit(1)
        shutil.rmtree(conf.exp_dir)
    os.mkdir(conf.exp_dir)
    os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
    if not conf.no_visu:
        os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # backup python files used for this training
    os.system('cp data_dynamic.py models/%s.py %s %s' % (conf.model_version, __file__, conf.exp_dir))
     
    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device

    ### start training
    train(conf)


    ### before quit
    # close file log
    flog.close()

