import time

import torch


@torch.no_grad()
def get_mask_similarity_matrix_by_threshold(inner_sim: torch.Tensor,
                                            atten: torch.Tensor, threshold: float, fill_value: float,
                                            device='cpu', ) -> torch.tensor:
    b, k = atten.shape[0], atten.shape[-1]
    eye_mask = torch.eye(b, dtype=bool, device=device).unsqueeze(-1).unsqueeze(-1)
    eye_mask = eye_mask.expand(-1, -1, k, k)
    inner_sim.masked_fill_(eye_mask, 0.)
    bool_mask = (inner_sim >= threshold).to(device=device)  # [b,b,k,k]
    position_tensor = bool_mask.nonzero()  # [nums, 4]
    atten_mask = torch.zeros(atten.shape, dtype=bool, device=device)
    # TODO 复杂度较高
    for pos in position_tensor:
        a, b, c, d = pos.tolist()
        atten_mask[a, b, :, d] = True
        atten_mask[b, a, :, c] = True
    atten.masked_fill_(atten_mask, fill_value)
    # TODO 存在一些问题 原来padding的地方是不是-inf还没确定，需要打个断点看一下loss的变化情况
    return atten

@torch.no_grad()
def get_mask_similarity_matrix_by_threshold_(inner_sim: torch.Tensor,
                                             atten: torch.Tensor, threshold: float, fill_value: float,
                                             device='cpu') -> torch.tensor:
    b, k = atten.shape[0], atten.shape[-1]
    eye_mask = torch.eye(b, dtype=bool, device=device).unsqueeze(-1).unsqueeze(-1)
    eye_mask = eye_mask.expand(-1, -1, k, k)
    inner_sim = inner_sim.masked_fill(eye_mask, 0.)
    bool_mask = (inner_sim >= threshold)  # [b, b, k, k]

    pos = bool_mask.nonzero(as_tuple=True)
    atten_mask = torch.zeros(atten.shape, dtype=bool, device=device)
    a, b, c, d = pos
    atten_mask[a, b, :, d] = True
    atten_mask[b, a, :, c] = True
    atten.masked_fill_(atten_mask, fill_value)
    return atten

    # return atten


if __name__ == "__main__":
    device = 'cuda:0'
    inner_sim = torch.randn(256, 256, 30, 30).to(device)
    atten = torch.randn(256, 256, 40, 30).to(device)
    print(atten.device)
    threshold = 0.7
    fill_value = -float('inf')
    import time
    t1 = time.time()
    print("a")
    at1 = get_mask_similarity_matrix_by_threshold(inner_sim, atten, threshold, fill_value, device)
    t2 = time.time()
    print("b")
    at2 = get_mask_similarity_matrix_by_threshold_(inner_sim, atten, threshold, fill_value, device)
    t3 = time.time()
    # print(torch.allclose(at1, at2))
    print(t3-t2,t2-t1)
    # print(at1)
    # print(at2)
