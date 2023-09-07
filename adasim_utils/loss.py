import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class AdaSimLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, args=None):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.args = args

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_output, student_output_feat = student_output
        teacher_output, teacher_output_feat = teacher_output
        teacher_output = teacher_output.detach()

        # Student sharpening
        student_out = student_output / self.student_temp
        student_out = F.log_softmax(student_out, dim=-1)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)

        # Reshape projected tensor of tokens to B x (nb_global_crops + nb_local_crops) x projected_dim
        # The projection function is the head network
        proj_student = student_out.reshape(self.args.local_crops_number + 2, self.args.batch_size_per_gpu,
                                           -1).transpose(0, 1)
        reshaped_teacher = teacher_out.reshape(2, self.args.batch_size_per_gpu, -1).transpose(0, 1)

        total_loss = 0

        # Global<->Global matching
        total_loss += (-proj_student[:, 0] * reshaped_teacher[:, 1]).mean(dim=0).sum()
        total_loss += (-proj_student[:, 1] * reshaped_teacher[:, 0]).mean(dim=0).sum()

        total_loss /= 2
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
