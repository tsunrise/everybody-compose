import torch
def status_check():
    cuda_available = torch.cuda.is_available()
    print("CUDA available: {}".format(cuda_available))
    if not cuda_available:
        return
    cuda_device_count = torch.cuda.device_count()
    print("CUDA device count: {}".format(cuda_device_count))
    cuda_device_name = torch.cuda.get_device_name(0)
    print("CUDA device name: {}".format(cuda_device_name))