import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import os
import numpy as np
from transformers import AutoModel
import scipy.sparse as sp
import pandas as pd
from tqdm import tqdm

from utils.nns_utils import ExactSearch
from utils.dl_utils import create_tf_pooler, ToD, BatchIterator, DistBatchIterator, apply_and_accumulate, GatherLayer, RandContext
from utils.topk_utils import TopK

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def ToD(self, batch):
        return ToD(batch, self.get_device())

    def get_device(self):
        if hasattr(self, 'device'):
            return self.device
        return list(self.parameters())[0].device

    def get_embs(self, data_source, bsz=256, accelerator=None, encode_func=None):
        self.eval()
        if isinstance(data_source, torch.utils.data.Dataset): 
            data_source = BatchIterator(data_source, bsz) if accelerator is None else DistBatchIterator(data_source, bsz)

        encode_func = self.encode if encode_func is None else encode_func 
        
        out = apply_and_accumulate(
            data_source, 
            lambda b: {'embs': encode_func(self.ToD(b))},
            accelerator,
            display_name='Embedding'
            )
        return out['embs'] if 'embs' in out else None

    def _predict_batch(self, b, K):
        b = ToD(b, self.get_device())
        out = self(b)
        if isinstance(out, torch.Tensor): # BxL shaped out
            top_vals, top_inds = torch.topk(out, K)
        elif isinstance(out, tuple) and len(out) == 2: # (logits, indices) shaped out 
            top_vals, temp_inds = torch.topk(out[0], K)
            top_inds = torch.gather(out[1], 1, temp_inds)
        return {'top_vals': top_vals, 'top_inds': top_inds}

    def predict(self, data_source, K=100, bsz=256, accelerator=None):
        self.eval()
        if isinstance(data_source, torch.utils.data.Dataset):
            data_source = BatchIterator(data_source, bsz)

        out = apply_and_accumulate(
            data_source, 
            self._predict_batch,
            accelerator,
            display_name='Predicting',
            **{'K': K}
            )

        if accelerator is None or accelerator.is_main_process:
            labels = data_source.dataset.labels
            indptr = np.arange(0, labels.shape[0]*K+1, K)
            score_mat = sp.csr_matrix((out['top_vals'].ravel(), out['top_inds'].ravel(), indptr), labels.shape)
            # remove padding if any
            if any(score_mat.indices == labels.shape[1]):
                score_mat.data[score_mat.indices == labels.shape[1]] = 0
                score_mat.eliminate_zeros()
            return score_mat

    def update_non_parameters(self, *args, **kwargs):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        loaded_state = torch.load(path, map_location='cpu')
        return self.load_state_dict(loaded_state, strict=False)

class TFEncoder(BaseNet):
    def __init__(self, args):
        super().__init__()
        tf_args = {'add_pooling_layer': False} if args.tf.startswith('bert-base') else {} 
        self.tf = AutoModel.from_pretrained(args.tf, **tf_args) if args.tf else None
        self.tf_pooler, self.tf_dims = create_tf_pooler(args.tf_pooler)
        self.bottleneck = nn.Linear(self.tf_dims, args.bottleneck_dim) if args.bottleneck_dim else None
        self.embs_dim = args.embs_dim = args.bottleneck_dim if args.bottleneck_dim else self.tf_dims
        self.dropout = nn.Dropout(args.dropout)
        self.norm_embs = args.norm_embs
        self.amp_encode = args.amp_encode
      
    def encode(self, b):
        with torch.cuda.amp.autocast(self.amp_encode):
            embs = b['xfts']
            if self.tf is not None:
                embs = self.tf_pooler(self.tf(**embs, output_hidden_states=True), embs)
            if self.bottleneck is not None:
                embs = self.bottleneck(embs)

            embs = self.dropout(embs)
            if self.norm_embs:
                embs = F.normalize(embs)
            return embs.float()

class DistBaseNetwork(TFEncoder):
    def __init__(self, args, accelerator, data_manager):
        super().__init__(args)
        self.accelerator = accelerator
        if accelerator is not None:
            self.rank = accelerator.state.process_index
            self.world_size = accelerator.state.num_processes
        else:
            self.rank = 0
            self.world_size = 1
        self.numy = args.numy
        self.numy_after_pad = int(self.world_size * np.ceil(self.numy / self.world_size))
        self.local_y_inds = torch.arange(self.rank*self.numy_after_pad//self.world_size, min((self.rank+1)*self.numy_after_pad//self.world_size, self.numy))
        self.tau = args.tau
        print(f"Rank {self.rank} of {self.world_size} initialized")

    def encode(self, b):
        return super().encode(b).contiguous()

    def compute_loss(self, sim, targets, loss_fn):
        @torch.no_grad()
        def filter_non_local_yinds(rows, cols, vals=None):
            local_y_mask = (cols >= self.local_y_inds[0]) & (cols < self.local_y_inds[-1])
            if vals is None: return rows[local_y_mask], cols[local_y_mask] - self.local_y_inds[0]
            else: return rows[local_y_mask], cols[local_y_mask] - self.local_y_inds[0], vals[local_y_mask]

        with self.accelerator.no_sync(self):
            return loss_fn(sim, targets, filter_non_local_yinds)
     
class DistClassifierNetwork(DistBaseNetwork):
    def __init__(self, args, accelerator, data_manager):
        super().__init__(args, accelerator, data_manager)
        self.local_y_inds = self.local_y_inds.to(f'cuda:{self.rank}')
        self.local_w = nn.Linear(self.embs_dim, self.numy_after_pad//self.world_size)

    def forward_backward(self, b, loss_fn, scaler=None):
        # forward
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            local_xembs = self.encode(b)
            all_xembs = GatherLayer.apply(local_xembs).view(-1, local_xembs.shape[1])

        with torch.no_grad():
            max_pad_len = self.accelerator.gather([torch.tensor(b['y']['inds'].shape[1], device=self.get_device())])[0].max().item()
            b['y']['inds'] = F.pad(b['y']['inds'], (0, max_pad_len - b['y']['inds'].shape[1]), value=self.numy)
            b['y']['vals'] = F.pad(b['y']['vals'], (0, max_pad_len - b['y']['vals'].shape[1]), value=0)
            target_inds, target_vals = self.accelerator.gather([b['y']['inds'], b['y']['vals']])

        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        sim = self.local_w(all_xembs)
        sim /= self.tau 
        if self.rank == self.world_size-1 and self.numy_after_pad > self.numy:
            sim = sim[:, :-(self.numy_after_pad-self.numy)]
                
        all_targets = (target_inds, target_vals)
        loss = self.compute_loss(sim, all_targets, loss_fn)

        with self.accelerator.no_sync(self):
            loss.backward() if scaler is None else scaler.scale(loss).backward()
            for name, param in self.named_parameters():
                if 'local_w' not in name:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) if dist.is_initialized() else None
                    # param.grad.data /= self.world_size
        dist.all_reduce(loss, op=dist.ReduceOp.SUM) if dist.is_initialized() else None
        return loss
    
    def _predict_batch(self, b, K):
        xembs = self.encode(b)
        xinds, xembs = self.accelerator.gather([b['ids'], xembs])

        sim = self.local_w(xembs)
        if self.rank == self.world_size-1 and self.numy_after_pad > self.numy:
            sim = sim[:, :-(self.numy_after_pad-self.numy)]

        topk_sim = torch.topk(sim, min(K, sim.shape[1]), dim=1)
        topk_inds = self.local_y_inds[topk_sim.indices]
        topk_vals = topk_sim.values
        del sim, xembs
        topk_inds, topk_vals = self.accelerator.gather([topk_inds.unsqueeze(0), topk_vals.unsqueeze(0)])
        topk_inds = topk_inds.permute(1, 0, 2).reshape(topk_inds.shape[1], -1)
        topk_vals = topk_vals.permute(1, 0, 2).reshape(topk_vals.shape[1], -1)
        topk_vals, topk_temp_inds = topk_vals.topk(K, dim=1)
        topk_inds = topk_inds.gather(1, topk_temp_inds)
        return xinds, topk_inds, topk_vals
    
    @torch.no_grad()
    def predict(self, data_loader, K=100, bsz=256, **kwargs):
        self.eval()
        if self.rank == 0:
            all_topk_inds = torch.zeros((len(data_loader.dataset), K), dtype=torch.long)
            all_topk_vals = torch.zeros((len(data_loader.dataset), K), dtype=torch.float)

        for b in tqdm(data_loader, desc='Predicting', disable=self.rank!=0):
            xinds, topk_inds, topk_vals = self._predict_batch(b, K)
            if self.rank == 0:
                all_topk_inds[xinds] = topk_inds.detach().cpu()
                all_topk_vals[xinds] = topk_vals.detach().cpu()
        
        if self.rank == 0:
            labels = data_loader.dataset.labels
            indptr = np.arange(0, labels.shape[0]*K+1, K)
            score_mat = sp.csr_matrix((all_topk_vals.ravel(), all_topk_inds.ravel(), indptr), labels.shape)
            # remove padding if any
            if any(score_mat.indices == labels.shape[1]):
                score_mat.data[score_mat.indices == labels.shape[1]] = 0
                score_mat.eliminate_zeros()
            return score_mat
        
    @torch.no_grad()
    def evaluate(self, data_loader, eval_Ks=[1,3,5,10,20,50,100], **kwargs):
        if data_loader.dataset.filter_mat is not None:
            score_mat = self.predict(data_loader, K=max(eval_Ks))
            if self.rank == 0:
                from utils.helper_utils import _filter, compute_xmc_metrics
                _filter(score_mat, data_loader.dataset.filter_mat, copy=False)
                metrics = compute_xmc_metrics(score_mat, data_loader.dataset.labels, None, disp=False)
                return metrics
            return None
        
        self.eval()
        K = max(eval_Ks)
        if self.rank == 0:
            metrics = {**{f'P@{k}': 0 for k in eval_Ks}, **{f'nDCG@{k}': 0 for k in eval_Ks}, **{f'R@{k}': 0 for k in eval_Ks}, **{f'wpR@{k}': 0 for k in eval_Ks}}

        total_count = 0
        total_true_val_sum = 0
        for b in tqdm(data_loader, desc='Evaluating', disable=self.rank!=0):
            xinds, topk_inds, topk_vals = self._predict_batch(b, K)
            true_inds, true_vals = b['y']['inds'], b['y']['vals']
            true_vals[true_vals < 1e-8] = 0
            true_inds[true_vals < 1e-8] = -100
            total_count += xinds.shape[0]
            
            true_inds_count = (true_inds >= 0).sum(dim=1).reshape(-1, 1)
            bsz = true_inds.shape[0]
            topk_inds = topk_inds[self.rank*bsz:(self.rank+1)*bsz]
            weighted_intrsxn = ((topk_inds.view(topk_inds.shape[0], -1, 1) == true_inds.view(true_inds.shape[0], 1, -1))*(true_vals.view(true_vals.shape[0], 1, -1))).sum(dim=-1)
            true_inds_count, weighted_intrsxn = self.accelerator.gather([true_inds_count, weighted_intrsxn])
            
            true_val_sum = true_vals.sum()
            dist.all_reduce(true_val_sum, op=dist.ReduceOp.SUM) if dist.is_initialized() else None
            

            if self.rank == 0:
                # Assumption topk_inds is a result of torch.topk operation
                weighted_intrsxn = weighted_intrsxn.cpu()   
                intrsxn = weighted_intrsxn.bool()
                total_true_val_sum += true_val_sum.item()
                true_inds_count = true_inds_count.cpu()

                for k in eval_Ks:
                    intrsxn_at_k = intrsxn[:, :k].sum(axis=-1)
                    dcg_coeff = 1/torch.log2(torch.arange(k)+2)
                    dcg_coeff_cumsum = torch.cumsum(dcg_coeff, 0)
                    dcg_at_k = torch.multiply(intrsxn[:, :k], dcg_coeff.reshape(1, -1)).sum(axis=-1)

                    metrics[f'P@{k}'] += intrsxn_at_k.sum().item()/k
                    metrics[f'R@{k}'] += (intrsxn_at_k/true_inds_count.ravel()).sum().item()

                    dcg_denom = dcg_coeff_cumsum[torch.minimum(true_inds_count, torch.tensor(k))-1].ravel()
                    metrics[f'nDCG@{k}'] += (dcg_at_k/dcg_denom).sum().item()

                    metrics[f'wpR@{k}'] += weighted_intrsxn[:, :k].sum().item()
        
        if self.rank == 0:
            metrics = {
                **{k: [v*100/total_count] for k, v in metrics.items() if not k.startswith('wpR')},
                **{k: [v*100/total_true_val_sum] for k, v in metrics.items() if k.startswith('wpR')}
            }
            return pd.DataFrame(metrics)
        
    def save(self, path):
        all_w_weight = [torch.zeros_like(self.local_w.weight) for _ in range(dist.get_world_size())] if self.rank == 0 else None
        dist.gather(self.local_w.weight, all_w_weight, dst=0)
        all_w_bias = [torch.zeros_like(self.local_w.bias) for _ in range(dist.get_world_size())] if self.rank == 0 else None
        dist.gather(self.local_w.bias, all_w_bias, dst=0)
        if self.rank == 0:
            all_w_weight = torch.cat(all_w_weight, dim=0)
            all_w_bias = torch.cat(all_w_bias, dim=0)
            state_dict = self.state_dict()
            state_dict = {n: p for n, p in state_dict.items() if 'local_w' not in n}
            state_dict['all_w.weight'] = all_w_weight
            state_dict['all_w.bias'] = all_w_bias
            torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.get_device())
        local_state_dict = {n: p for n, p in state_dict.items() if 'all_w' not in n}
        local_state_dict['local_w.weight'] = torch.zeros_like(self.local_w.weight)
        local_state_dict['local_w.bias'] = torch.zeros_like(self.local_w.bias)

        clf_chunk_size = self.numy_after_pad // self.world_size
        local_clf_inds = range(self.rank*clf_chunk_size, min((self.rank+1)*clf_chunk_size, self.numy))
        local_state_dict['local_w.weight'][:len(local_clf_inds)] = state_dict['all_w.weight'][local_clf_inds]
        local_state_dict['local_w.bias'][:len(local_clf_inds)] = state_dict['all_w.bias'][local_clf_inds]

        return self.load_state_dict(local_state_dict)

from collections import UserDict
class DistDualEncoderAll(DistBaseNetwork):
    def __init__(self, args, accelerator, data_manager):
        super().__init__(args, accelerator, data_manager)
        self.gc_bsz = args.gc_bsz
        self.lbl_dataset = data_manager.lbl_dataset
        num_local_y = self.numy_after_pad//self.world_size
        if self.rank*num_local_y + self.local_y_inds.shape[0] > self.numy:
            self.local_y_inds = self.local_y_inds[:-(self.numy_after_pad - self.numy)]

    def get_input_tensors(self, model_input):
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, torch.Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def encode(self, b):
        return super().encode(b).contiguous()
    
    def encode_x(self, b):
        return self.encode({'xfts': b['xfts']})

    def encode_y(self, b):
        return self.encode({'xfts': b['yfts'] if 'yfts' in b else b['xfts']})

    def local_encode_y(self, b):
        # return emb of all y and the random states
        rand_states = []
        inputs = []
        with torch.no_grad():
            concat_yembs = torch.zeros((self.local_y_inds.shape[0], self.embs_dim), device=self.get_device())
            for i in range(0, self.local_y_inds.shape[0], self.gc_bsz):
                input_range = range(i, min(i+self.gc_bsz, self.local_y_inds.shape[0]))
                yembs_input = self.ToD({'yfts': self.lbl_dataset.get_fts(self.local_y_inds[input_range])})
                inputs.append((yembs_input, input_range))
                rand_states.append(RandContext(*self.get_input_tensors(yembs_input)))
                concat_yembs[input_range] = self.encode_y(yembs_input)
        return inputs, concat_yembs, rand_states

    def forward_backward(self, b, loss_fn, scaler=None):
        # forward
        local_xembs = self.encode_x(b)
        all_xembs = GatherLayer.apply(local_xembs).view(-1, local_xembs.shape[1])
        local_yinputs, local_yembs, local_yembs_rand_states = self.local_encode_y(b)
        local_yembs.requires_grad = True

        with torch.no_grad():
            max_pad_len = self.accelerator.gather([torch.tensor(b['y']['inds'].shape[1], device=self.get_device())])[0].max().item()
            b['y']['inds'] = F.pad(b['y']['inds'], (0, max_pad_len - b['y']['inds'].shape[1]), value=self.numy)
            b['y']['vals'] = F.pad(b['y']['vals'], (0, max_pad_len - b['y']['vals'].shape[1]), value=0)
            target_inds, targe_vals = self.accelerator.gather([b['y']['inds'], b['y']['vals']])

        sim = all_xembs @ local_yembs.t()
        sim /= self.tau
        all_targets = (target_inds, targe_vals)
        loss = self.compute_loss(sim, all_targets, loss_fn)

        # backward using gradient caching
        with self.accelerator.no_sync(self):
            local_xembs.grad, local_yembs.grad = torch.autograd.grad(loss, (local_xembs, local_yembs))

            # do normal backward through xembs
            torch.autograd.backward(local_xembs, local_xembs.grad)
            
            # do backward through yembs using cached gradients
            for (yembs_input, yinds_range), rand_state in zip(local_yinputs, local_yembs_rand_states):
                with rand_state:
                    yembs = self.encode_y(yembs_input)
                yembs.grad = local_yembs.grad[yinds_range]
                torch.autograd.backward(yembs, yembs.grad)

            for param in self.parameters():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) if dist.is_initialized() else None
                param.grad.data /= self.world_size
        dist.all_reduce(loss, op=dist.ReduceOp.SUM) if dist.is_initialized() else None
        return loss

    def predict(self, data_loader, K = 100, bsz = 256, **kwargs):
        x_embs = self.get_embs(data_loader.dataset.x_dataset, bsz, encode_func=self.encode_x, accelerator=self.accelerator)
        y_embs = self.get_embs(data_loader.dataset.y_dataset, bsz, encode_func=self.encode_y, accelerator=self.accelerator)
        if self.accelerator and not self.accelerator.is_main_process:
            return None
        es = ExactSearch(y_embs, device=self.get_device(), K=K)
        score_mat = es.search(x_embs)
        return score_mat

class DistDualEncoderHNM(DistDualEncoderAll):
    def __init__(self, args, accelerator, data_manager):
        self.OUT_DIR = args.OUT_DIR
        self.hard_neg_start = args.hard_neg_start
        self.eval_interval = args.eval_interval
        self.update_trn_shorty = False
        self.hard_neg_topk = args.hard_neg_topk
        self.trn_dataset = data_manager.trn_dataset
        self.avg_labels_per_batch = [0, 0]
        super().__init__(args, accelerator, data_manager)

    def accumulate_batch(self, b):
        with torch.no_grad():
            b_y_max_len = self.accelerator.gather([torch.tensor(b['batch_y'].shape[0], device=self.get_device())])[0].max().item()
            b_y = F.pad(b['batch_y'], (0, b_y_max_len - b['batch_y'].shape[0]), value=self.numy)
            b_y = self.accelerator.gather([b_y])[0]
            b_y = torch.unique(b_y)
            b_y = b_y[b_y < self.numy]
            remap_inds = torch.full((self.numy+1,), self.numy, dtype=torch.long, device=self.get_device())
            remap_inds[b_y] = torch.arange(b_y.shape[0], device=self.get_device())
            b['y']['inds'] = remap_inds[b['y']['inds']]
            b['y']['vals'][b['y']['inds'] >= self.numy] = 0
            b['batch_y'] = b_y
            split_size = int(np.ceil(b_y.shape[0] / self.world_size))
            self.local_y_inds = torch.arange(b_y.shape[0]).split(split_size)[self.rank].to(self.get_device())
        return b
    
    def local_encode_y(self, b):
        # return emb of all y and the random states
        rand_states = []
        inputs = []
        with torch.no_grad():
            concat_yembs = torch.zeros((self.local_y_inds.shape[0], self.embs_dim), device=self.get_device())
            for i in range(0, self.local_y_inds.shape[0], self.gc_bsz):
                input_range = range(i, min(i+self.gc_bsz, self.local_y_inds.shape[0]))
                yembs_input = self.ToD({'yfts': self.lbl_dataset.get_fts(b['batch_y'][self.local_y_inds[input_range]].cpu())})
                inputs.append((yembs_input, input_range))
                rand_states.append(RandContext(*self.get_input_tensors(yembs_input)))
                concat_yembs[input_range] = self.encode_y(yembs_input)
        return inputs, concat_yembs, rand_states

    def forward_backward(self, b, loss_fn, scaler=None):
        b = self.accumulate_batch(b)
        self.avg_labels_per_batch[0] += b['batch_y'].shape[0]
        self.avg_labels_per_batch[1] += 1
        return super().forward_backward(b, loss_fn)
    
    def update_non_parameters(self, epoch, step, *args, **kwargs):
        epoch_end = 'epoch_end' in kwargs and kwargs['epoch_end']
        self.update_trn_shorty = (epoch % self.eval_interval == 0) and (epoch >= self.hard_neg_start)
        
        if epoch_end:
            self.accelerator.print(f'Average labels per batch: {self.avg_labels_per_batch[0] / self.avg_labels_per_batch[1]}')
            self.avg_labels_per_batch = [0, 0]

        if self.update_trn_shorty and epoch_end:
            print(f'[Rank {self.rank}] Updating trn_shorty at epoch {epoch}...')
            trn_loader = kwargs['data_loader']
            trn_loader.dataset.shorty = sp.load_npz(os.path.join(self.OUT_DIR, f'trn_shorty.npz'))
            trn_loader.collate_fn.neg_type = 'shorty'
            self.accelerator.wait_for_everyone()
    
    def predict(self, data_loader, K=100, bsz=256):
        tstx_embs = self.get_embs(data_loader.dataset.x_dataset, bsz, encode_func=self.encode_x, accelerator=self.accelerator)
        y_embs = self.get_embs(data_loader.dataset.y_dataset, bsz, encode_func=self.encode_y, accelerator=self.accelerator)
        if self.update_trn_shorty:
            trnx_embs = self.get_embs(self.trn_dataset.x_dataset, bsz, encode_func=self.encode_x, accelerator=self.accelerator)

        if self.accelerator.is_main_process:
            es = ExactSearch(y_embs, device=self.get_device(), K=K)
            score_mat = es.search(tstx_embs)

            # Code for generating negative shortlist
            if self.update_trn_shorty:
                print('Updating negative shortlist...')
                es = ExactSearch(y_embs, device=self.get_device(), K=self.hard_neg_topk)
                trn_score_mat = es.search(trnx_embs)
                from utils.helper_utils import _filter
                _filter(trn_score_mat, self.trn_dataset.labels, copy=False)
                trn_score_mat.data[:] = 1
                self.trn_dataset.shorty = trn_score_mat
                sp.save_npz(os.path.join(self.OUT_DIR, 'trn_shorty.npz'), trn_score_mat)

            return score_mat
        
NETS = {
    'dist-clf-net': DistClassifierNetwork,
    'dist-de-all': DistDualEncoderAll,
    'dist-de-hnm': DistDualEncoderHNM
    }