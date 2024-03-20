import torch.nn as nn
import torch.nn.functional as F
import torch 
from utils.dl_utils import unwrap
from utils.dl_utils import GatherLayer

import torch.distributed as dist
from utils.topk_utils import dist_log_topk

class SimLoss(object):
    def __init__(self, args):
        super().__init__()
        self.loss_criterion = args.loss_criterion
        self.loss_sample = args.loss_sample
        self.loss_reduction = args.loss_reduction
        self.loss_weighted = args.loss_weighted
        self.numy = args.numy
        if self.loss_criterion == 'topk':
            self.topk_K = args.topk_K
            self.topk_alpha = args.topk_alpha
            self.topk_n_iter = args.topk_n_iter

    def compute_loss(self, sim, all_targets, filter_non_local_yinds):
        if isinstance(all_targets, tuple):
            EPSILON=1e-8
            with torch.no_grad():
                target_inds, target_vals = all_targets
                nnz = (target_vals > EPSILON).nonzero(as_tuple=True)
                pos_cols = target_inds[nnz[0], nnz[1]]
                pos_vals = target_vals[nnz[0], nnz[1]]
                pos_rows = torch.repeat_interleave(torch.arange(target_inds.shape[0], device=target_inds.device), (target_vals > EPSILON).sum(dim=1))
                num_pos = (target_vals > EPSILON).sum(dim=1)

            if self.loss_sample or (self.loss_criterion == 'topk' and isinstance(self.topk_K, int) and self.topk_K < target_inds.shape[1]):
                with torch.no_grad():
                    num_pos_sample = min(self.topk_K, target_inds.shape[1]) if self.loss_criterion == 'topk' else 1

                    if not dist.is_initialized() or dist.get_rank() == 0:
                        b_y_inds = torch.multinomial(target_vals.double(), num_pos_sample, replacement=False)
                    else:
                        b_y_inds = torch.zeros_like(target_inds[:, :num_pos_sample])
                    # broadcast to all processes
                    if dist.is_initialized(): dist.broadcast(b_y_inds, 0)

                    y_inds = target_inds.gather(1, b_y_inds)
                    y_vals = target_vals.gather(1, b_y_inds)
                    pos_rows, pos_cols = y_vals.nonzero(as_tuple=True)
                    pos_vals = y_vals[pos_rows, pos_cols]
                    pos_cols = y_inds[pos_rows, pos_cols]

                    non_sampled_pos_vals = target_vals > EPSILON
                    non_sampled_pos_vals.scatter_(1, b_y_inds, False)
                    nnz = non_sampled_pos_vals.nonzero(as_tuple=True)
                    non_sampled_pos_cols = target_inds[nnz[0], nnz[1]]
                    non_sampled_pos_rows = torch.repeat_interleave(torch.arange(target_inds.shape[0], device=target_inds.device), non_sampled_pos_vals.sum(dim=1))
                    num_pos = (y_vals > EPSILON).sum(dim=1)

                if self.loss_criterion in ['topk', 'decoupled-softmax']:
                    non_sampled_pos_rows, non_sampled_pos_cols =  filter_non_local_yinds(non_sampled_pos_rows, non_sampled_pos_cols)
                    sim[non_sampled_pos_rows, non_sampled_pos_cols] = -100.0 # put large negative value at labels which are positive but not sampled                    

            pos_rows, pos_cols, pos_vals = filter_non_local_yinds(pos_rows, pos_cols, pos_vals)

            if self.loss_criterion == 'softmax':
                log_denom = torch.logsumexp(sim, dim=1, keepdim=False)
                log_denom = GatherLayer.apply(log_denom)
                log_denom = torch.logsumexp(log_denom, dim=0, keepdim=False)

                log_prob = sim[pos_rows, pos_cols] - log_denom[pos_rows]
                if self.loss_weighted: log_prob *= pos_vals
                loss = -log_prob.sum() / sim.shape[0]
                return loss
            elif self.loss_criterion == 'bce':
                if self.loss_weighted: raise NotImplementedError
                sim_pos = sim[pos_rows, pos_cols]
                loss = torch.logaddexp(sim, torch.zeros_like(sim[:, 0:1])).sum() - sim_pos.sum()
                loss /= (sim.shape[0] * self.numy)
                return loss
            elif self.loss_criterion == 'decoupled-softmax':
                sim_pos = sim[pos_rows, pos_cols]
                sim_neg = sim
                sim_neg[pos_rows, pos_cols] = -100.0

                log_denom = sim_neg.logsumexp(dim=1, keepdim=False)
                log_denom = GatherLayer.apply(log_denom)
                log_denom = torch.logsumexp(log_denom, dim=0)
                log_denom = torch.logaddexp(log_denom[pos_rows], sim_pos)

                log_prob = sim_pos - log_denom
                if self.loss_weighted: log_prob *= pos_vals

                loss = -(log_prob / num_pos[pos_rows]).sum() / sim.shape[0]
                return loss
            elif self.loss_criterion == 'topk':
                if isinstance(self.topk_K, str) and self.topk_K.startswith('num_pos'):
                    topk_K = (target_vals.sum(dim=1)).ceil().long()
                else:
                    topk_K = self.topk_K

                log_topk = dist_log_topk(sim, topk_K, self.topk_alpha, self.topk_n_iter)
                pos_log_topk = log_topk[pos_rows, pos_cols] / num_pos[pos_rows]
                if self.loss_weighted: pos_log_topk *= pos_vals
                loss = -pos_log_topk.sum() / sim.shape[0]
                return loss
            
LOSSES = {
    'sim-loss': SimLoss,
    }