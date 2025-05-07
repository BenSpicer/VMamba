import torch
# from space_filling_pytorch import encode_hilbert
import spacefill.curvetools as ct

# hcurve = torch.tensor([0, 1, 14, 15, 
#                          3, 2, 13, 12, 
#                          4, 7, 8, 11,
#                          5, 6, 9, 10])

def space_fill(values: torch.Tensor, reverse=False) -> torch.Tensor:
    # print('val:', values.shape)
    x_size, y_size = values.shape[-1], values.shape[-2]

    curve_map = ct.generate_map(x_size, -y_size)
    
    indices = torch.tensor([(-y-1)*x_size + x for x, y in curve_map[0]], dtype=torch.int64).cuda()
    
    if not reverse:
        indices = torch.argsort(indices)
    
    # print('val_flat: ', values.flatten(start_dim=-2).shape)
    indices = indices.expand(*values.flatten(start_dim=-2).shape)
    # print('ind: ', indices.shape)
    values = torch.gather(values.flatten(start_dim=-2), dim=-1, index=indices).reshape(values.shape)
    return values