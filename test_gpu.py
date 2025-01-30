import torch
import time

N = 10000

a = torch.randn((N, N), device='cpu')
b = torch.randn((N, N), device='cpu')

p_time = time.time()
c = a @ b
print(time.time() - p_time)

if torch.cuda.is_available():    
    print("CUDA is available")
    a = torch.randn((N, N), device='cuda')
    b = torch.randn((N, N), device='cuda')

    p_time = time.time()
    c = a @ b
    print(time.time() - p_time)

    p_time = time.time()
    d = a @ b
    print(time.time() - p_time)

    p_time = time.time()
    e = a @ b
    print(time.time() - p_time)
