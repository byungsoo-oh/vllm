# SPDX-License-Identifier: Apache-2.0
from vllm.v1.request import Request, RequestStatus
import torch
import torch.distributed as dist

def check_stop(request: Request, max_model_len: int) -> bool:
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    sampling_params = request.sampling_params
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False

def log_gpu_memory():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # in GB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # in GB

    print(f"[BS] (gpu memory | rank {rank} | device {device}) "
          f"allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")

