import torch
import apex


FP16_ENABLED = False
# Apex setting. O2 level means we keep batchnorms layer and a copy of the weight in fp32
OPTIMIZATION_LEVEL = 'O2'
LOSS_SCALING_ENABLED = True


def disable_loss_scaling():
    global LOSS_SCALING_ENABLED
    LOSS_SCALING_ENABLED = False
    return


def enable_fp16():
    global FP16_ENABLED
    FP16_ENABLED = True
    return


def is_fp16():
    return FP16_ENABLED


def is_loss_scaling_enabled():
    return LOSS_SCALING_ENABLED


def get_optim_level():
    return OPTIMIZATION_LEVEL


def set_optim_level(opt_level):
    global OPTIMIZATION_LEVEL
    OPTIMIZATION_LEVEL = opt_level
    return


def clip_grad(optimizer, parameters, norm=5.):
    if is_fp16():
        torch.nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer), norm)
    else:
        torch.nn.utils.clip_grad_norm_(parameters, norm)


# Convert a tensor to half precision if FP16_ENABLED
def maybe_half(tensor):
    return tensor.half() if is_fp16() else tensor


def initialize(model, lr=0.0005):
    if is_fp16():
        # from apex.optimizers import FP16_Optimizer
        # from apex.optimizers import FusedAdam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=get_optim_level())
        # model = apex.parallel.DistributedDataParallel(model)
        # model = model.half()
        # optimizer = FusedAdam(model.parameters(), lr=lr, bias_correction=False)
        # optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        return model, optimizer
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model, optimizer


def get_optimizer(obj):
    from apex.fp16_utils import FP16_Optimizer

    # Apex introduce the FP16_optimizer object
    # However this isn't really an optimizer, but only a wrapper around one.
    # This function returns the actual optimizer.
    if type(obj) == FP16_Optimizer:
        return obj.optimizer
    # If obj is not an FP16_Optimizer then we are not running in mixed precision
    # and the passed object is already an actual optimizer
    return obj


def backward(loss, optimizer):
    if FP16_ENABLED and is_loss_scaling_enabled():
        # optimizer.backward(loss)
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    return


#
# EYE BUFFER
#
