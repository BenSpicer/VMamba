import torch
import spacefill.curvetools as ct

def space_fill(values: torch.Tensor, reverse=False) -> torch.Tensor:
    x_size, y_size = values.shape[-1], values.shape[-2]

    curve_map = ct.generate_map(x_size, -y_size)
    
    indices = torch.tensor([(-y-1)*x_size + x for x, y in curve_map[0]], dtype=torch.int64).cuda()
    
    if not reverse:
        indices = torch.argsort(indices)
    
    indices = indices.expand(*values.flatten(start_dim=-2).shape)
    values = torch.gather(values.flatten(start_dim=-2), dim=-1, index=indices).reshape(values.shape)
    return values