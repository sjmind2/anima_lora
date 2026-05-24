import torch


def rebuild_tucker(t, wa, wb):
    return torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)


def factorization(dimension: int, factor: int = -1):
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1d, w1u, w2d, w2u, scale=torch.tensor(1)):
        ctx.save_for_backward(w1d, w1u, w2d, w2u, scale)
        diff_weight = ((w1u @ w1d) * (w2u @ w2d)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1d, w1u, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2u @ w2d)
        grad_w1u = temp @ w1d.T
        grad_w1d = w1u.T @ temp
        temp = grad_out * (w1u @ w1d)
        grad_w2u = temp @ w2d.T
        grad_w2d = w2u.T @ temp
        del temp
        return grad_w1d, grad_w1u, grad_w2d, grad_w2u, None


class HadaWeightTucker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1, w1d, w1u, t2, w2d, w2u, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1d, w1u, t2, w2d, w2u, scale)
        rebuild1 = torch.einsum("i j ..., j r, i p -> p r ...", t1, w1d, w1u)
        rebuild2 = torch.einsum("i j ..., j r, i p -> p r ...", t2, w2d, w2u)
        return rebuild1 * rebuild2 * scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1d, w1u, t2, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale

        temp = torch.einsum("i j ..., j r -> i r ...", t2, w2d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w2u)
        grad_w = rebuild * grad_out
        del rebuild

        grad_w1u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w1u.T)
        del grad_w, temp

        grad_w1d = torch.einsum("i r ..., i j ... -> r j", t1, grad_temp)
        grad_t1 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w1d.T)
        del grad_temp

        temp = torch.einsum("i j ..., j r -> i r ...", t1, w1d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w1u)
        grad_w = rebuild * grad_out
        del rebuild

        grad_w2u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w2u.T)
        del grad_w, temp

        grad_w2d = torch.einsum("i r ..., i j ... -> r j", t2, grad_temp)
        grad_t2 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w2d.T)
        del grad_temp
        return grad_t1, grad_w1d, grad_w1u, grad_t2, grad_w2d, grad_w2u, None


def make_kron(w1, w2, scale):
    for _ in range(w2.dim() - w1.dim()):
        w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    if scale != 1:
        rebuild = rebuild * scale
    return rebuild


def factored_kron_forward(x, w1, w2):
    out_l, in_m = w1.shape
    out_k, in_n = w2.shape
    leading = x.shape[:-1]
    x_flat = x.reshape(-1, in_m * in_n)
    X = x_flat.reshape(x_flat.shape[0], in_m, in_n)
    temp = X @ w2.T
    result = torch.einsum("pr,brk->bpk", w1, temp)
    return result.reshape(*leading, out_l * out_k)
