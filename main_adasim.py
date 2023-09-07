# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import argparse
import sys
import datetime
import time
import math
import json
from pathlib import Path
from adasim_utils.loss import AdaSimLoss
from adasim_utils.dataset import DatasetFolderAdaSim
import subprocess
import tarfile
from PIL import Image
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models
import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from adasim_utils.parser import get_args_parser

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def train_adasim(args):
    if 'CXI_FORK_SAFE_HP' in os.environ:
        del os.environ['CXI_FORK_SAFE_HP']
    if 'CXI_FORK_SAFE' in os.environ:
        del os.environ['CXI_FORK_SAFE']
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args
    )

    ###############################################################################################
    start_copy_time = time.time()
    print('Start to copy')
    if len(args.untar_path) > 0 and args.untar_path[0] == '$':
        args.untar_path = os.environ[args.untar_path[1:]]

    start_copy_time = time.time()
    if args.data_path.split('/')[-1] == 'ilsvrc2012.tar':
        if int(args.gpu) == 0:
            with tarfile.open(args.data_path, 'r') as f:
                f.extractall(args.untar_path)

            print('Time taken for untar:', time.time() - start_copy_time)
            print(os.listdir(args.untar_path))

        args.data_path = os.path.join(args.untar_path, 'ilsvrc2012', 'ILSVRC2012_img_train')
        args.data_path_val = os.path.join(args.untar_path, 'ilsvrc2012', 'ILSVRC2012_img_val')

    torch.distributed.barrier()

    dataset = DatasetFolderAdaSim(args.data_path, args, transform=transform, return_index_instead_of_target=True)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim

    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    features_cpu = torch.zeros(len(data_loader.dataset), embed_dim, dtype=torch.float)

    nn_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.long)
    sim_tensor_cpu = torch.zeros(len(data_loader.dataset), args.topk, dtype=torch.float)

    nn_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.long)
    sim_matrix_cpu = torch.zeros(len(data_loader.dataset), args.vote_nn_nb, args.topk, dtype=torch.float)

    # ============ preparing loss ... ============
    adasim_loss = AdaSimLoss(
        args.out_dim,
        2,  # total number of crops = 2 global crops
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs, args=args
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============n
    to_restore = {"epoch": 0,
                  'nn_tensor_cpu': nn_tensor_cpu,
                  'sim_tensor_cpu': sim_tensor_cpu,
                  'features_cpu': features_cpu,
                  "nn_matrix_cpu": nn_matrix_cpu,
                  "sim_matrix_cpu": sim_matrix_cpu}
    default_checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    if not os.path.isfile(default_checkpoint_path) and os.path.isfile(args.start_checkpoint_path):
        utils.master_copy_from_to(args.start_checkpoint_path, default_checkpoint_path)

    torch.distributed.barrier()

    try:
        utils.restart_from_checkpoint(
            default_checkpoint_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            adasim_loss=adasim_loss,
        )
    except:
        # If checkpoint is corrupted, used backedup checkpoint
        utils.restart_from_checkpoint(
            default_checkpoint_path + '.backup',
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            adasim_loss=adasim_loss,
        )
    start_epoch = to_restore["epoch"]
    features_cpu = to_restore["features_cpu"]
    nn_tensor_cpu = to_restore["nn_tensor_cpu"]
    sim_tensor_cpu = to_restore["sim_tensor_cpu"]
    nn_matrix_cpu = to_restore["nn_matrix_cpu"]
    sim_matrix_cpu = to_restore["sim_matrix_cpu"]
    nn_matrix_cpu = nn_matrix_cpu[:, -args.vote_nn_nb:]
    sim_matrix_cpu = sim_matrix_cpu[:, -args.vote_nn_nb:]

    features = features_cpu.cuda()
    nn_tensor = nn_tensor_cpu.cuda()
    sim_tensor = sim_tensor_cpu.cuda()

    start_time = time.time()
    print("Starting AdaSim training !")
    bootstrap_myself_tensor = torch.zeros(len(dataset), dtype=torch.long)
    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        data_loader.sampler.set_epoch(epoch)

        if epoch >= args.vote_nn_nb:
            data_loader.dataset.nn_matrix_cpu = nn_matrix_cpu
            data_loader.dataset.sim_matrix_cpu = sim_matrix_cpu
        try:
            # ============ training one epoch of DINO ... ============

            train_stats = train_one_epoch(student, teacher, teacher_without_ddp, adasim_loss,
                                          data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                          epoch, fp16_scaler, features, nn_tensor, sim_tensor, bootstrap_myself_tensor,
                                          args)
            nn_tensor_cpu = nn_tensor.cpu()
            sim_tensor_cpu = sim_tensor.cpu()
            features_cpu = features.cpu()

            update_nn(nn_tensor_cpu, sim_tensor_cpu, nn_matrix_cpu, sim_matrix_cpu)

        except Exception as e:
            print(e, flush=True)
            with open(os.path.join(args.output_dir, 'error_{}'.format(args.rank)), 'w') as f:
                f.write(str(e))
            time.sleep(3)  # Should not be needed

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'adasim_loss': adasim_loss.state_dict(),
            'nn_tensor_cpu': nn_tensor_cpu,
            'sim_tensor_cpu': sim_tensor_cpu,
            'features_cpu': features_cpu,
            'nn_matrix_cpu': nn_matrix_cpu,
            'sim_matrix_cpu': sim_matrix_cpu
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        if epoch == 0:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
                         'epoch_time': time.time() - start_epoch_time}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
                         'epoch_time': time.time() - start_epoch_time,
                         'nn_accuracy_top1': get_nn_acuracy(dataset, nn_tensor_cpu[:, 0]),
                         'nn_accuracy_top2': get_nn_acuracy(dataset, nn_tensor_cpu[:, 1]),
                         'nn_self_accuracy_top1': get_nn_self_accuracy(nn_tensor_cpu[:, 0]),
                         'nn_self_accuracy_top2': get_nn_self_accuracy(nn_tensor_cpu[:, 1]),
                         'nn_ratio_self': bootstrap_myself_tensor.sum().item() / len(bootstrap_myself_tensor)}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_nn_acuracy(dataset, nn_tensor):
    t = torch.Tensor(dataset.targets)
    return sum(t[nn_tensor] == t).item() / len(nn_tensor)


def get_nn_self_accuracy(nn_tensor):
    return sum(nn_tensor == torch.Tensor(range(len(nn_tensor)))).item() / len(nn_tensor)


def update_nn(nn_tensor_cpu, sim_tensor_cpu, nn_matrix_cpu, sim_matrix_cpu):
    with torch.no_grad():
        # Shift tensor by 1 index
        nn_matrix_cpu[:, :-1] = nn_matrix_cpu[:, 1:]
        nn_matrix_cpu[:, -1] = nn_tensor_cpu

        sim_matrix_cpu[:, :-1] = sim_matrix_cpu[:, 1:]
        sim_matrix_cpu[:, -1] = sim_tensor_cpu


def gather_all_gpus(local_tensor):
    feats_all = torch.empty(dist.get_world_size(), *local_tensor.shape, dtype=local_tensor.dtype,
                            device=local_tensor.device)
    output_l = list(feats_all.unbind(0))
    output_all_reduce = torch.distributed.all_gather(output_l, local_tensor, async_op=True)
    output_all_reduce.wait()
    return torch.cat(output_l)


def update_state(teacher_output, indices_local, features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor,
                 args):
    with torch.no_grad():
        feats_local = F.normalize(teacher_output[1].reshape(2, -1, teacher_output[1].shape[-1]), p=2, dim=-1)
        if args.nn_rep_type == "mean":
            feats_local = feats_local.mean(dim=0)
            feats_local = F.normalize(feats_local, p=2, dim=-1)
        elif args.nn_rep_type == "first":
            feats_local = feats_local[0]
        elif args.nn_rep_type == "second":
            feats_local = feats_local[1]
        else:
            raise NotImplemented

        # Compute knn
        similarity = feats_local @ features.T
        sims_knn_local, indices_knn_local = similarity.topk(dim=-1, k=args.topk)

        indices_batch_all = gather_all_gpus(indices_local)
        features_batch_all = gather_all_gpus(feats_local)
        indices_knn_batch_all = gather_all_gpus(indices_knn_local)
        sims_knn_batch_all = gather_all_gpus(sims_knn_local)

        features.index_copy_(0, indices_batch_all, features_batch_all)
        nn_tensor.index_copy_(0, indices_batch_all, indices_knn_batch_all)
        sim_tensor.index_copy_(0, indices_batch_all, sims_knn_batch_all)

        # Same as above but for cpu tensor
        bootstrap_myself_tensor[indices_batch_all.cpu()] = gather_all_gpus(same_im_bool).cpu()


def train_one_epoch(student, teacher, teacher_without_ddp, adasim_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, features, nn_tensor, sim_tensor, bootstrap_myself_tensor, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, (images, indices, same_im_bool) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        indices = indices.cuda(non_blocking=True)
        same_im_bool = same_im_bool.cuda(non_blocking=True)
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = adasim_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            subprocess.run("scancel $SLURM_JOB_ID", shell=True, check=True, env=dict(os.environ))
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        update_state(teacher_output, indices, features, nn_tensor, sim_tensor, same_im_bool, bootstrap_myself_tensor,
                     args)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, args):
        self.args = args
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image, image2):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image2))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    # For benchmarking the code
    if args.epochs <= 2:
        args.warmup_teacher_temp_epochs = 0
        args.warmup_epochs = 0

    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_adasim(args)
