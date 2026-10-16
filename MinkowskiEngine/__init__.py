import torch
import torch.nn as nn
import itertools

STRIDE_TO_COORDS = {}

class SparseTensor:
    def __init__(self, features, coordinates, device=None, tensor_key=None, coords_key=None, coords_manager=None, stride=1):
        self.F = features  # [N, C]
        self.C = coordinates  # [N, 4] (b, x, y, z)
        self.device = device if device is not None else features.device
        self.stride = stride
        
        batch_hash = coordinates.shape[0]
        STRIDE_TO_COORDS[(batch_hash, stride)] = coordinates

    def __add__(self, other):
        if isinstance(other, SparseTensor):
            return SparseTensor(self.F + other.F, self.C, device=self.device, stride=self.stride)
        else:
            return SparseTensor(self.F + other, self.C, device=self.device, stride=self.stride)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, SparseTensor):
            return SparseTensor(self.F - other.F, self.C, device=self.device, stride=self.stride)
        else:
            return SparseTensor(self.F - other, self.C, device=self.device, stride=self.stride)

    def __rsub__(self, other):
        if isinstance(other, SparseTensor):
            return SparseTensor(other.F - self.F, self.C, device=self.device, stride=self.stride)
        else:
            return SparseTensor(other - self.F, self.C, device=self.device, stride=self.stride)

    def __mul__(self, other):
        if isinstance(other, SparseTensor):
            return SparseTensor(self.F * other.F, self.C, device=self.device, stride=self.stride)
        else:
            return SparseTensor(self.F * other, self.C, device=self.device, stride=self.stride)

    def __rmul__(self, other):
        return self.__mul__(other)

class MinkowskiNetwork(nn.Module):
    def __init__(self, dimension=3):
        super().__init__()
        self.dimension = dimension

def hash_coords(coords):
    b = coords[:, 0].to(torch.int64)
    x = coords[:, 1].to(torch.int64) + 500000
    y = coords[:, 2].to(torch.int64) + 500000
    z = coords[:, 3].to(torch.int64) + 500000
    return b * 10**18 + x * 10**12 + y * 10**6 + z

def get_kernel_map(C_in, C_out, offsets):
    N_out = C_out.shape[0]
    K = len(offsets)
    device = C_in.device
    
    keys_in = hash_coords(C_in)
    sorted_keys, perm = torch.sort(keys_in)
    
    offsets_tensor = torch.zeros((K, 4), dtype=C_out.dtype, device=device)
    offsets_tensor[:, 1:4] = torch.tensor(offsets, dtype=C_out.dtype, device=device)
    
    C_query = C_out.unsqueeze(1) + offsets_tensor.unsqueeze(0)
    keys_query_flat = hash_coords(C_query.view(-1, 4))
    
    idx = torch.bucketize(keys_query_flat, sorted_keys)
    idx_clamped = torch.clamp(idx, 0, len(sorted_keys) - 1)
    
    matched = (sorted_keys[idx_clamped] == keys_query_flat)
    flat_M = torch.where(matched, perm[idx_clamped], -1)
    
    return flat_M.view(N_out, K)

class MinkowskiConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False, dimension=3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        
        if kernel_size % 2 == 1:
            half_k = kernel_size // 2
            offsets_list = list(itertools.product(range(-half_k, half_k + 1), repeat=3))
        else:
            offsets_list = list(itertools.product(range(kernel_size), repeat=3))
            
        self.offsets = offsets_list
        self.K = len(offsets_list)
        
        if self.K == 1:
            self.kernel = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.Tensor(self.K, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        C_in = x.C
        F_in = x.F
        device = F_in.device
        
        if self.stride == 1:
            C_out = C_in
            stride_out = x.stride
        elif self.stride == 2:
            C_out_raw = C_in.clone()
            C_out_raw[:, 1:4] = torch.div(C_in[:, 1:4], 2, rounding_mode='floor')
            C_out = torch.unique(C_out_raw, dim=0)
            stride_out = x.stride * 2
        else:
            raise NotImplementedError(f"Stride {self.stride} not supported")
            
        N_out = C_out.shape[0]
        offsets_scaled = [(ox * self.dilation, oy * self.dilation, oz * self.dilation) for ox, oy, oz in self.offsets]
        M = get_kernel_map(C_in, C_out, offsets_scaled)
        
        F_in_padded = torch.cat([F_in, torch.zeros((1, self.in_channels), dtype=F_in.dtype, device=device)], dim=0)
        F_neighbors = F_in_padded[M]
        
        if self.kernel.ndim == 2:
            F_out = F_neighbors.squeeze(1) @ self.kernel
        else:
            F_out = torch.einsum('nki,kio->no', F_neighbors, self.kernel)
            
        if self.bias is not None:
            F_out = F_out + self.bias
            
        return SparseTensor(F_out, C_out, device=device, stride=stride_out)


class MinkowskiConvolutionTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, dilation=1, bias=False, dimension=3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        
        if kernel_size % 2 == 1:
            half_k = kernel_size // 2
            offsets_list = list(itertools.product(range(-half_k, half_k + 1), repeat=3))
        else:
            offsets_list = list(itertools.product(range(kernel_size), repeat=3))
            
        self.offsets = offsets_list
        self.K = len(offsets_list)
        
        if self.K == 1:
            self.kernel = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.Tensor(self.K, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        C_in = x.C
        F_in = x.F
        device = F_in.device
        
        target_stride = x.stride // self.stride
        C_out = None
        for (batch_hash, stride_val), C_cand in STRIDE_TO_COORDS.items():
            if stride_val == target_stride:
                if self.stride == 1:
                    if C_cand.shape == C_in.shape and torch.equal(C_cand, C_in):
                        C_out = C_cand
                        break
                else:
                    C_cand_down_raw = C_cand.clone()
                    C_cand_down_raw[:, 1:4] = torch.div(C_cand_down_raw[:, 1:4], self.stride, rounding_mode='floor')
                    C_cand_down = torch.unique(C_cand_down_raw, dim=0)
                    if C_cand_down.shape == C_in.shape and torch.equal(C_cand_down, C_in):
                        C_out = C_cand
                        break
                        
        if C_out is None:
            for (batch_hash, stride_val), C_cand in STRIDE_TO_COORDS.items():
                if stride_val == target_stride:
                    C_out = C_cand
                    break
                    
        if C_out is None:
            raise RuntimeError(f"Could not find coordinates for target stride {target_stride} in STRIDE_TO_COORDS")
            
        N_out = C_out.shape[0]
        
        offsets_neg = [(-ox * self.dilation, -oy * self.dilation, -oz * self.dilation) for ox, oy, oz in self.offsets]
        M = get_kernel_map(C_in, C_out, offsets_neg)
        
        F_in_padded = torch.cat([F_in, torch.zeros((1, self.in_channels), dtype=F_in.dtype, device=device)], dim=0)
        F_neighbors = F_in_padded[M]
        
        if self.kernel.ndim == 2:
            F_out = F_neighbors.squeeze(1) @ self.kernel
        else:
            F_out = torch.einsum('nki,kio->no', F_neighbors, self.kernel)
            
        if self.bias is not None:
            F_out = F_out + self.bias
            
        return SparseTensor(F_out, C_out, device=device, stride=target_stride)

class MinkowskiBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        
    def forward(self, x):
        F_out = self.bn(x.F)
        return SparseTensor(F_out, x.C, device=x.device, stride=x.stride)

class MinkowskiReLU(nn.Module):
    def __init__(self, inplace=False, **kwargs):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        
    def forward(self, x):
        F_out = self.relu(x.F)
        return SparseTensor(F_out, x.C, device=x.device, stride=x.stride)

class MinkowskiLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x):
        F_out = self.linear(x.F)
        return SparseTensor(F_out, x.C, device=x.device, stride=x.stride)

class MinkowskiGlobalPooling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x):
        device = x.F.device
        F_out = torch.mean(x.F, dim=0, keepdim=True)
        C_out = torch.zeros((1, 4), dtype=x.C.dtype, device=device)
        return SparseTensor(F_out, C_out, device=device, stride=1)

class MinkowskiBroadcastMultiplication(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x, y):
        F_out = x.F * y.F
        return SparseTensor(F_out, x.C, device=x.device, stride=x.stride)

class MinkowskiSigmoid(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x):
        F_out = torch.sigmoid(x.F)
        return SparseTensor(F_out, x.C, device=x.device, stride=x.stride)

def cat(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        tensors = args[0]
    else:
        tensors = args
    features = torch.cat([t.F for t in tensors], dim=-1)
    return SparseTensor(features, tensors[0].C, device=tensors[0].device, stride=tensors[0].stride)

class utils:
    @staticmethod
    def kaiming_normal_(tensor, **kwargs):
        pass

class RegionType:
    HYPER_CUBE = 0
    HYPER_CROSS = 1
    CUSTOM = 2

class KernelGenerator:
    def __init__(self, kernel_size, stride=1, dilation=1, region_type=None, axis_types=None, dimension=3, **kwargs):
        pass

class MinkowskiInstanceNorm(nn.Module):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features)
    def forward(self, x):
        # x.F shape: [N, C] -> unsqueeze(0) -> [1, N, C] -> transpose(1, 2) -> [1, C, N]
        feat = x.F.unsqueeze(0).transpose(1, 2)
        out = self.norm(feat)
        # [1, C, N] -> transpose(1, 2) -> [1, N, C] -> squeeze(0) -> [N, C]
        F_out = out.transpose(1, 2).squeeze(0)
        return SparseTensor(F_out, x.C, device=x.device, stride=x.stride)

class MinkowskiAvgPooling(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class MinkowskiAvgUnpooling(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class MinkowskiSumPooling(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

