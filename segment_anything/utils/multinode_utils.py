import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
from segment_anything.utils.transforms import ResizeLongestSide
import webdataset as wds
import numpy as np
import random
import torch
import os
import builtins
import tqdm
from torch.distributed.nn.functional import all_gather


def setup_comm(rank, world_size, backend):
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def setup_multinode_training (seed, backend):

    # This function initialized multinode training by inizializing everything needed
    use_cuda = torch.cuda.is_available()
    world_size = int(os.environ["WORLD_SIZE"])

    if "SLURM_PROCID" in os.environ:
        print("Using Slurm")
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    else:
        print("Using torchrun")
        rank = int(os.environ["LOCAL_RANK"])
        gpus_per_node = torch.cuda.device_count()  

    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    torch.manual_seed(seed + rank)
    setup_comm(rank, world_size, backend)

    return world_size, rank, local_rank

def all_gather_with_gradient (input):
    # Just a wrapper around all_gather then cat all results, as all_gather returns a list and not a tensor
    return torch.cat(all_gather(input))


def unequal_all_gather (input, buffer_size, async_all_gather = False):
    # Handles sharing tensors between devices if the input tensors are unequal size, a padding is added to get them to a uniform size, the padding is removed before return the tensor
    # TODO: currently a fixed buffer size is used calculated from the max possible size from each device, actuall figure out how big this buffer must be in each iteration, could be much more memory efficient.
    dim_0 = buffer_size - input.shape[0]
    pad = torch.zeros([dim_0] +  list(input.shape[1:])).to(input.device)
    input = torch.cat([input, pad])
    input = all_gather_with_gradient(input)
    # Remove Pad (Removes every row that is not zero summed over everything that is not the first dimension)
    return input[input.sum(dim = [x for x in range(1, input.dim())]) != 0]


def supress_output ():
    def print_pass(*args):
        pass
    ## Supress printing from non-master nodes
    builtins.print = print_pass
    ## We have to remove TQDM outputs from non-master nodes seperately
    class _TQDM(tqdm.tqdm):
        def __init__(self, *argv, **kwargs):
            kwargs['disable'] = True
            if kwargs.get('disable_override', 'def') != 'def':
                kwargs['disable'] = kwargs['disable_override']
            super().__init__(*argv, **kwargs)
    tqdm.tqdm = _TQDM