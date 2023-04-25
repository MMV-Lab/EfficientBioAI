from collections import namedtuple
import functools
import time
import torch


# adapted from https://stackoverflow.com/questions/6866600/how-to-parse-read-a-yaml-file-into-a-python-object # noqa: E501
class Dict2ObjParser:
    """Parse a nested dictionary into a nested named tuple."""

    def __init__(self, nested_dict):
        self.nested_dict = nested_dict

    def parse(self):
        nested_dict = self.nested_dict
        if (obj_type := type(nested_dict)) is not dict:
            raise TypeError(f"Expected 'dict' but found '{obj_type}'")
        return self._transform_to_named_tuples("root", nested_dict)

    def _transform_to_named_tuples(self, tuple_name, possibly_nested_obj):
        if type(possibly_nested_obj) is dict:
            named_tuple_def = namedtuple(tuple_name, possibly_nested_obj.keys())
            transformed_value = named_tuple_def(
                *[
                    self._transform_to_named_tuples(key, value)
                    for key, value in possibly_nested_obj.items()
                ]
            )
        elif type(possibly_nested_obj) is list:
            transformed_value = [
                self._transform_to_named_tuples(
                    f"{tuple_name}_{i}", possibly_nested_obj[i]
                )
                for i in range(len(possibly_nested_obj))
            ]
        else:
            transformed_value = possibly_nested_obj

        return transformed_value


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def pad(img: torch.Tensor, N) -> torch.Tensor:
    """for calibration. Unet requires size of input to be muptiples of 2^N, where N is the depth of the network. # noqa: E501

    Args:
        img (torch.Tensor):
        N: depth of network

    Returns:
        output (torch.Tensor): return padded img
    """
    z, y, x = img.shape[-3:]
    f = (
        lambda length, N: (2 ^ N) * (length // 2 ^ N)
        if (length % (2 ^ N)) <= (2 ^ (N - 1))
        else (2 ^ N) * (length // 2 ^ N + 1)
    )  # to nearest multiples of 2^N

    z, y, x = f(z), f(y), f(x)

    pass
