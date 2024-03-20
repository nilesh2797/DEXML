import torch
import torch.distributed as dist
from torch.autograd import Function

# topk code adapted from: https://gist.github.com/thomasahle/4c1e85e5842d01b007a8d10f5fed3a18

sigmoid = torch.sigmoid
def sigmoid_grad(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)

class DistLogTopK(Function):
    @staticmethod
    def forward(ctx, xs, k, alpha, n_iter=32):
        logits, log_sig = _dist_find_ts(xs, k, alpha=alpha, n_iter=n_iter, return_log_sig=True)
        ctx.save_for_backward(torch.tensor(alpha), logits)
        return log_sig

    @staticmethod
    def backward(ctx, grad_output):
        # Compute vjp, that is grad_output.T @ J.
        alpha, logits = ctx.saved_tensors
        # Let v = sigmoid'(x + t)
        sig_x = sigmoid(logits)
        v = alpha*sig_x*(1-sig_x)
        s = v.sum(dim=1, keepdims=True)
        if dist.is_initialized(): dist.all_reduce(s, op=dist.ReduceOp.SUM)
        # Jacobian is -(1-v)v.T/s + diag(v)
        uv = grad_output * alpha * (1-sig_x)
        uv_sum = uv.sum(dim=1, keepdims=True)
        if dist.is_initialized(): dist.all_reduce(uv_sum, op=dist.ReduceOp.SUM)
        t1 = - uv_sum * v / s 
        return t1.add_(uv), None, None, None
    
class DistTopK(Function):
    @staticmethod
    def forward(ctx, xs, k, alpha, n_iter=32):
        logits, sig = _dist_find_ts(xs, k, alpha=alpha, n_iter=n_iter, return_log_sig=False)
        ctx.save_for_backward(torch.tensor(alpha), logits)
        return sig

    @staticmethod
    def backward(ctx, grad_output):
        # Compute vjp, that is grad_output.T @ J.
        alpha, logits = ctx.saved_tensors
        # Let v = sigmoid'(x + t)
        v = alpha*sigmoid_grad(logits)
        s = v.sum(dim=1, keepdims=True)
        if dist.is_initialized(): dist.all_reduce(s, op=dist.ReduceOp.SUM)
        uv = grad_output * v
        uv_sum = uv.sum(dim=1, keepdims=True)
        if dist.is_initialized(): dist.all_reduce(uv_sum, op=dist.ReduceOp.SUM)
        t1 = - uv_sum * v / s
        return t1 + uv, None, None, None

@torch.no_grad()
def _dist_find_ts(xs, k, alpha=1, n_iter=64, return_log_sig=False):
    assert alpha > 0
    b, n = xs.shape
    if dist.is_initialized(): n *= dist.get_world_size()
    if isinstance(k, int):
        assert 0 < k < n
    elif isinstance(k, torch.LongTensor):
        assert (0 < k).all() and (k < n).all()
    # Lo should be small enough that all sigmoids are in the 0 area.
    # Similarly Hi is large enough that all are in their 1 area.
    xs_min = xs.min(dim=1, keepdims=True).values
    xs_max = xs.max(dim=1, keepdims=True).values
    if dist.is_initialized(): dist.all_reduce(xs_min, op=dist.ReduceOp.MIN)
    if dist.is_initialized(): dist.all_reduce(xs_max, op=dist.ReduceOp.MAX)
    lo = -xs_max - 10/alpha
    hi = -xs_min + 10/alpha
    for _ in range(n_iter):
        mid = (hi + lo)/2
        sigmoid_sum = sigmoid(alpha*(xs + mid)).sum(dim=1)
        if dist.is_initialized(): dist.all_reduce(sigmoid_sum, op=dist.ReduceOp.SUM)
        mask = sigmoid_sum < k
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]

    ts = (lo + hi)/2
    logits = alpha*(xs + ts)
    if return_log_sig:
        log_sig = logits - torch.logaddexp(logits, torch.zeros_like(logits[:, :1]))
        return logits, log_sig
    else:
        return logits, sigmoid(logits)

dist_log_topk = DistLogTopK.apply
dist_topk = DistTopK.apply

class TopK(Function):
    @staticmethod
    def forward(ctx, xs, k, alpha, n_iter=32):
        '''
        Compute the top-k loss for each row of xs.
        xs: (b, n) input tensor
        k: int (number of top elements to consider)
        alpha: float (multiplicative factor to xs)
        n_iter: int (number of iterations for binary search)
        '''
        ts, ps = _find_ts(xs, k, alpha=alpha, n_iter=n_iter)
        ctx.save_for_backward(torch.tensor(alpha), xs, ts)
        return ps

    @staticmethod
    def backward(ctx, grad_output):
        # Compute vjp, that is grad_output.T @ J.
        alpha, xs, ts = ctx.saved_tensors
        # Let v = sigmoid'(x + t)
        v = alpha*sigmoid_grad(alpha*(xs + ts))
        s = v.sum(dim=1, keepdims=True)
        # Jacobian is -vv.T/s + diag(v)
        uv = grad_output * v
        t1 = - uv.sum(dim=1, keepdims=True) * v / s
        return t1 + uv, None, None, None

@torch.no_grad()
def _find_ts(xs, k, alpha=1, n_iter=64):
    assert alpha > 0
    b, n = xs.shape
    if isinstance(k, int):
        assert 0 < k < n
    elif isinstance(k, torch.LongTensor):
        assert (0 < k).all() and (k < n).all()
    # Lo should be small enough that all sigmoids are in the 0 area.
    # Similarly Hi is large enough that all are in their 1 area.
    lo = -xs.max(dim=1, keepdims=True).values - 10/alpha
    hi = -xs.min(dim=1, keepdims=True).values + 10/alpha
    for _ in range(n_iter):
        mid = (hi + lo)/2
        mask = sigmoid(alpha*(xs + mid)).sum(dim=1) < k
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]
    ts = (lo + hi)/2
    return ts, sigmoid(alpha*(xs + ts))

topk = TopK.apply