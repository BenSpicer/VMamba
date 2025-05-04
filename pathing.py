import torch
from torch import LongTensor, Tensor
# from space_filling_pytorch import encode_hilbert
import spacefill.curvetools as ct

hcurve = torch.tensor([0, 1, 14, 15, 
                         3, 2, 13, 12, 
                         4, 7, 8, 11,
                         5, 6, 9, 10])

def space_fill(values: Tensor, method='hilbert', reverse=False) -> Tensor:
    print(values.shape)
    
    indices = 
    
    if not reverse:
        indices = torch.argsort(indices)
    
    num_dims = indices.dim()

    new_shape = tuple(indices.shape) + tuple(
        1
        for _ in range(values.dim() - num_dims)
    )
    repeats = tuple(
        1
        for _ in range(num_dims)
    ) + tuple(values.shape[num_dims:])

    repeated_indices = indices.reshape(*new_shape).repeat(*repeats)
    return torch.gather(values, num_dims - 1, repeated_indices)