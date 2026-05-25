import math
from collections import defaultdict

import torch


def _came_rms(tensor):
    return tensor.norm(2) / (tensor.numel() ** 0.5)


def _came_approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = (
        (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
        .rsqrt_()
        .unsqueeze(-1)
    )
    c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
    return torch.mul(r_factor, c_factor)


def _came_approx_sq_grad_batch(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = (
        (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
        .rsqrt_()
        .unsqueeze(-1)
    )
    c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
    return torch.mul(r_factor, c_factor)


def _came_step_factored_kernel(
    param_data, grad, exp_avg, exp_avg_sq_row, exp_avg_sq_col,
    exp_avg_res_row, exp_avg_res_col,
    lr, beta0, beta1, beta2, eps0, eps1, clip_threshold, weight_decay,
):
    update = (grad ** 2) + eps0
    exp_avg_sq_row = exp_avg_sq_row.mul(beta1).add(update.mean(dim=-1), alpha=1.0 - beta1)
    exp_avg_sq_col = exp_avg_sq_col.mul(beta1).add(update.mean(dim=-2), alpha=1.0 - beta1)
    update = _came_approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col).mul(grad)
    rms = _came_rms(update)
    update = update.div((rms / clip_threshold).clamp(min=1.0))
    exp_avg = exp_avg.mul(beta0).add(update, alpha=1.0 - beta0)
    res = (update - exp_avg) ** 2 + eps1
    exp_avg_res_row = exp_avg_res_row.mul(beta2).add(res.mean(dim=-1), alpha=1.0 - beta2)
    exp_avg_res_col = exp_avg_res_col.mul(beta2).add(res.mean(dim=-2), alpha=1.0 - beta2)
    res_approx = _came_approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
    update = res_approx.mul(exp_avg)
    if weight_decay != 0:
        param_data = param_data.add(param_data, alpha=-weight_decay * lr)
    update = update.mul(lr)
    param_data = param_data.add(-update)
    return param_data, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_res_row, exp_avg_res_col


def _came_step_unfactored_kernel(
    param_data, grad, exp_avg, exp_avg_sq,
    lr, beta0, beta1, eps0, clip_threshold, weight_decay,
):
    update = (grad ** 2) + eps0
    exp_avg_sq = exp_avg_sq.mul(beta1).add(update, alpha=1.0 - beta1)
    update = exp_avg_sq.rsqrt().mul(grad)
    rms = _came_rms(update)
    update = update.div((rms / clip_threshold).clamp(min=1.0))
    exp_avg = exp_avg.mul(beta0).add(update, alpha=1.0 - beta0)
    update = exp_avg.clone()
    if weight_decay != 0:
        param_data = param_data.add(param_data, alpha=-weight_decay * lr)
    update = update.mul(lr)
    param_data = param_data.add(-update)
    return param_data, exp_avg, exp_avg_sq


_came_step_factored_compiled = None
_came_step_unfactored_compiled = None


def _ensure_compiled():
    global _came_step_factored_compiled, _came_step_unfactored_compiled
    if _came_step_factored_compiled is not None:
        return
    _came_step_factored_compiled = torch.compile(_came_step_factored_kernel, fullgraph=False)
    _came_step_unfactored_compiled = torch.compile(_came_step_unfactored_kernel, fullgraph=False)


def _step_group_factored_stacked(group_items, lr, beta0, beta1, beta2, eps0, eps1, clip_threshold, weight_decay):
    n = len(group_items)
    params_stack = torch.stack([p.data for p, g, s in group_items])
    grads_stack = torch.stack([g for p, g, s in group_items])
    exp_avg_stack = torch.stack([s["exp_avg"] for p, g, s in group_items])
    sq_row_stack = torch.stack([s["exp_avg_sq_row"] for p, g, s in group_items])
    sq_col_stack = torch.stack([s["exp_avg_sq_col"] for p, g, s in group_items])
    res_row_stack = torch.stack([s["exp_avg_res_row"] for p, g, s in group_items])
    res_col_stack = torch.stack([s["exp_avg_res_col"] for p, g, s in group_items])

    update = (grads_stack ** 2) + eps0

    sq_row_stack = sq_row_stack.mul(beta1).add(update.mean(dim=-1), alpha=1.0 - beta1)
    sq_col_stack = sq_col_stack.mul(beta1).add(update.mean(dim=-2), alpha=1.0 - beta1)

    update = _came_approx_sq_grad_batch(sq_row_stack, sq_col_stack).mul(grads_stack)

    rms_per_param = update.flatten(start_dim=1).norm(2, dim=1) / math.sqrt(update[0].numel())
    clamp_factors = (rms_per_param / clip_threshold).clamp(min=1.0)
    update = update / clamp_factors.view(n, 1, 1)

    exp_avg_stack = exp_avg_stack.mul(beta0).add(update, alpha=1.0 - beta0)

    res = (update - exp_avg_stack) ** 2 + eps1

    res_row_stack = res_row_stack.mul(beta2).add(res.mean(dim=-1), alpha=1.0 - beta2)
    res_col_stack = res_col_stack.mul(beta2).add(res.mean(dim=-2), alpha=1.0 - beta2)

    res_approx = _came_approx_sq_grad_batch(res_row_stack, res_col_stack)
    update = res_approx.mul(exp_avg_stack)

    if weight_decay != 0:
        params_stack = params_stack.add(params_stack, alpha=-weight_decay * lr)

    update = update.mul(lr)
    params_stack = params_stack.add(-update)

    for i, (p, g, s) in enumerate(group_items):
        p.data = params_stack[i]
        s["exp_avg"] = exp_avg_stack[i]
        s["exp_avg_sq_row"] = sq_row_stack[i]
        s["exp_avg_sq_col"] = sq_col_stack[i]
        s["exp_avg_res_row"] = res_row_stack[i]
        s["exp_avg_res_col"] = res_col_stack[i]


def _step_group_unfactored_foreach(group_items, lr, beta0, beta1, eps0, clip_threshold, weight_decay):
    params_list = [p.data for p, g, s in group_items]
    grads_list = [g for p, g, s in group_items]
    exp_avg_list = [s["exp_avg"] for p, g, s in group_items]
    exp_avg_sq_list = [s["exp_avg_sq"] for p, g, s in group_items]

    updates = torch._foreach_pow(grads_list, [2.0])
    torch._foreach_add_(updates, [eps0])

    torch._foreach_mul_(exp_avg_sq_list, [beta1])
    torch._foreach_add_(exp_avg_sq_list, updates, alpha=1.0 - beta1)

    torch._foreach_rsqrt_(exp_avg_sq_list)
    torch._foreach_mul_(exp_avg_sq_list, grads_list)
    updates = [v.clone() for v in exp_avg_sq_list]

    rms_vals = torch._foreach_norm(updates, 2)
    nelem_sqrt = math.sqrt(group_items[0][0].numel())
    clamp_factors = [max(1.0, (r.item() / nelem_sqrt) / clip_threshold) for r in rms_vals]
    torch._foreach_div_(updates, clamp_factors)

    torch._foreach_mul_(exp_avg_list, [beta0])
    torch._foreach_add_(exp_avg_list, updates, alpha=1.0 - beta0)

    final_updates = [ea.clone() for ea in exp_avg_list]

    if weight_decay != 0:
        torch._foreach_add_(params_list, params_list, alpha=-weight_decay * lr)

    torch._foreach_mul_(final_updates, [lr])
    torch._foreach_sub_(params_list, final_updates)

    for i, (p, g, s) in enumerate(group_items):
        p.data = params_list[i]
        s["exp_avg"] = exp_avg_list[i]
        s["exp_avg_sq"] = exp_avg_sq_list[i]


class CAME(torch.optim.Optimizer):

    supports_memory_efficient_fp16 = True
    supports_flat_params = False

    def __init__(self, params, lr, eps=(1e-30, 1e-16), clip_threshold=1.0, betas=(0.9, 0.999, 0.9999), weight_decay=0.0):
        assert lr > 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_state(self, p, grad, state):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad)
        if len(grad.shape) >= 2:
            state["exp_avg_sq_row"] = torch.zeros(grad.shape[:-1], dtype=grad.dtype, device=grad.device)
            state["exp_avg_sq_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], dtype=grad.dtype, device=grad.device)
            state["exp_avg_res_row"] = torch.zeros(grad.shape[:-1], dtype=grad.dtype, device=grad.device)
            state["exp_avg_res_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], dtype=grad.dtype, device=grad.device)
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta0, beta1, beta2 = group["betas"]
            eps0, eps1 = group["eps"]
            clip_threshold = group["clip_threshold"]
            weight_decay = group["weight_decay"]

            factored_groups = defaultdict(list)
            unfactored_groups = defaultdict(list)
            single_factored = []
            single_unfactored = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, grad, state)
                state["step"] += 1

                item = (p, grad, state)
                if len(grad.shape) >= 2:
                    shape_key = tuple(grad.shape)
                    factored_groups[shape_key].append(item)
                else:
                    shape_key = tuple(grad.shape)
                    unfactored_groups[shape_key].append(item)

            for shape_key, items in factored_groups.items():
                if len(items) > 1:
                    _step_group_factored_stacked(items, lr, beta0, beta1, beta2, eps0, eps1, clip_threshold, weight_decay)
                else:
                    p, grad, state = items[0]
                    if grad.is_cuda:
                        _ensure_compiled()
                        fn = _came_step_factored_compiled
                    else:
                        fn = _came_step_factored_kernel
                    p.data, state["exp_avg"], state["exp_avg_sq_row"], state["exp_avg_sq_col"], state["exp_avg_res_row"], state["exp_avg_res_col"] = fn(
                        p.data, grad, state["exp_avg"], state["exp_avg_sq_row"], state["exp_avg_sq_col"], state["exp_avg_res_row"], state["exp_avg_res_col"],
                        lr, beta0, beta1, beta2, eps0, eps1, clip_threshold, weight_decay,
                    )

            for shape_key, items in unfactored_groups.items():
                if len(items) > 1:
                    _step_group_unfactored_foreach(items, lr, beta0, beta1, eps0, clip_threshold, weight_decay)
                else:
                    p, grad, state = items[0]
                    if grad.is_cuda:
                        _ensure_compiled()
                        fn = _came_step_unfactored_compiled
                    else:
                        fn = _came_step_unfactored_kernel
                    p.data, state["exp_avg"], state["exp_avg_sq"] = fn(
                        p.data, grad, state["exp_avg"], state["exp_avg_sq"],
                        lr, beta0, beta1, eps0, clip_threshold, weight_decay,
                    )
        return loss
