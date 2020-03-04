import os

import numpy as np
import torch
import torch.distributed as dist
from fp16_utils import maybe_half


MAX_GROUP_SIZE = 32
current_process_group = None


def init_distributed_training(local_rank, port_idx=0):
    ports = ['8787', '8686', '8585', '8484']
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = ports[port_idx]
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         rank=local_rank)
    create_groups()


def create_groups():
    global current_process_group
    # collect some useful information
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    # assign gpus to groups
    group, groups = [], []
    for i in range(world_size):
        group.append(i)
        if (len(group) == MAX_GROUP_SIZE) or (i == (world_size - 1)):
            groups.append(group)
            group = []
    # tell pytorch about each group
    for i, group in enumerate(groups):
        process_group = dist.new_group(ranks=group)
        if my_rank in group:
            # record which process group includes current process
            current_process_group = process_group
        if my_rank == 0:
            print('Adding distributed group {}, including GPUs [{}...{}]'
                  .format(i, group[0], group[-1]))


def get_group():
    return current_process_group


def get_group_idx():
    return dist.get_rank() // MAX_GROUP_SIZE


def get_group_size():
    return get_group().size()


def get_group_rank():
    return dist.get_rank(get_group())


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def reduce_scalar(scalar):
    if hasattr(scalar, 'device'):
        scalar = scalar.item()
    reduced_scalar = reduce_tensor(torch.tensor(scalar).cuda()).item()
    return reduced_scalar


def all_gather_no_grad(tensor, *output_list):
    dist.all_gather(list(output_list), tensor, get_group())
    return tuple(output_list)


class AllGatherWithGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, *output_list):
        dist.all_gather(list(output_list), tensor, get_group())
        return tuple(output_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # collect some info about current process and process groups
        group = get_group()
        group_rank = get_group_rank()
        group_size = get_group_size()
        global_rank = dist.get_rank()
        # figure out which processes are in group with current process
        start_idx = get_group_idx() * MAX_GROUP_SIZE
        group_idx = np.arange(start_idx, start_idx + group_size)
        # gather gradient info from all processes in current process group
        handles = []
        t_grad = grad_outputs[group_rank].clone()
        for i in range(group_size):
            if i == group_rank:
                # gradient info from self
                hdl = dist.reduce(t_grad.contiguous(), global_rank,
                                  group=group, async_op=True)
            else:
                # gradient info from other group members
                hdl = dist.reduce(grad_outputs[i].contiguous(), group_idx[i],
                                  group=group, async_op=True)
            handles.append(hdl)
        # wait for async ops to finish
        for h in handles:
            h.wait()
        return (t_grad,) + grad_outputs


def all_gather_local_group(*tensors):
    all_gather_fn = AllGatherWithGrads.apply
    # all_gather_fn = all_gather_no_grad
    t_out = []
    for t in tensors:
        out = [torch.empty_like(t) for i in range(get_group_size())]
        out = all_gather_fn(t, *out)
        out = torch.cat(out)
        out = maybe_half(out)
        t_out.append(out)
    return t_out
