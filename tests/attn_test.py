from modules import ScaleDotProductAttention
import torch



def main():
    q = torch.rand( 2, 24)
    k = torch.rand( 2,  24)
    v = torch.rand( 2, 24)

    sdpa =     ScaleDotProductAttention()
    output, attn  = sdpa(q, k, v) 
    print(output)
    print(output.shape)


    print(attn)
    print(attn.shape)

if __name__ == '__main__':
    main()

