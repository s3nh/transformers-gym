import torch
import triton

import triton.language as tl
#https://en.cppreference.com/w/cpp/language/constexpr
@triton.jit
def add_kernel(x_ptr, 
               y_ptr, 
               output_ptr, 
               n_elements,  # Size of the vector
               BLOCK_SIZE: tl.constexpr ):
    # Identify program pid
    pid =  tl.program_id(axis = 0) # 1D launch grid
    # Identify when its started
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Offsets is a list of pointers
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y
    # Write backto DRAM
    tl.store(output_ptr, output, mask = mask)


def add(x: torch.Tensor, 
        y: torch.Tensor):
    output = torch.empty_like(x).to(device = 'cpu')
    # Check if they are on the same device 
    print(x.is_cuda)
    print(y.is_cuda)
    print(output.is_cuda)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE = 1024)
    return output

def main():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device = 'cpu')
    y = torch.rand(size, device = 'cpu')
    output_torch = x + y
    output_triton = add(x, y)
    print(f"1e-4 difference precision {torch.max(torch.abs(output_torch - output_triton)) > 1e-3}" )

if __name__ == '__main__':
    main()
