import torch
import numpy as np
import matplotlib.pyplot as plt
import traceback


def plot_tensor(tensor, img_size, mode=None):
    assert np.prod(tensor.size()) == np.prod(img_size)
    if mode:
        if mode == "RGB":
            data = tensor.numpy()
            data = data.reshape(img_size).astype(np.uint8)

            plt.imshow(data, interpolation="nearest")
    else:
        data = tensor.numpy()
        data = data.reshape(img_size)
        imgplot = plt.imshow(data, interpolation="nearest")
        imgplot.set_cmap('gray')

    plt.show()


def get_modules(model: torch.nn.Module) -> dict:
    modules = {}
    keys_to_use = model.get_used_keys()
    for name, param in model.named_parameters():
        module = name[0: name.index('.')]
        if module in keys_to_use:
            if module not in modules.keys():
                modules[module] = []

            modules[module].append((name, param))

    return modules


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FreezeWeightsHook:
    """
    Resets the gradient according to the passed masks. True is frozen.
    """

    def __init__(self, mask):
        self.mask = mask
        self.__name__ = ""  # Bug in pytorch see https://github.com/pytorch/pytorch/issues/37672

    def __call__(self, grad):
        try:  # Errors don't get propagated up, this is necessary. See https://github.com/pytorch/pytorch/issues/37672
            grad_clone = grad.clone().detach()

            grad_clone[self.mask] = 0

            return grad_clone

        except Exception:
            print(self.mask)
            traceback.print_exc()
