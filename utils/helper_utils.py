#!/usr/bin/env python
# coding: utf-8

import os, sys, yaml, argparse, re
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import csr_matrix

np.random.seed(22)

if '__sysstdout__' not in locals():
    __sysstdout__ = sys.stdout

def load_yaml(fname):
    yaml_dict = yaml.safe_load(open(fname))
    yaml_dict_list = []
    if '__dependency__' in yaml_dict:
        yaml_dict['__dependency__'] = ', '.join([dep_fname.strip() if os.path.isabs(dep_fname.strip()) else f'{os.getcwd()}/{dep_fname.strip()}' for dep_fname in yaml_dict['__dependency__'].split(',')])
        for dep_fname in yaml_dict['__dependency__'].split(','):
            yaml_dict_list = yaml_dict_list + load_yaml(dep_fname.strip())
    yaml_dict_list.append(yaml_dict)
    return yaml_dict_list

def load_config_and_runtime_args(argv):
    try: config_sep_index = [x.startswith('-') for x in argv[1:]].index(True) 
    except: config_sep_index = len(argv[1:])
    config_args = argv[1:1+config_sep_index]
    runtime_args = argv[1+config_sep_index:]

    parser = argparse.ArgumentParser()
    yaml_dict_lol = [load_yaml(fname) for fname in config_args]
    yaml_dict_list = [yaml_dict for yaml_dict_list in yaml_dict_lol for yaml_dict in yaml_dict_list]
    config = {k: v for d in yaml_dict_list for k, v in d.items()}
    config = pd.json_normalize(config, sep='_').to_dict(orient="records")[0]
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=str_to_bool if isinstance(v, bool) else type(v))
    args = parser.parse_args(runtime_args)
    args.__dict__ = {k: re.sub(r'\[(\w+)\]', lambda x: args.__dict__[x.group(0)[1:-1]], v) if isinstance(v, str) else v for k, v in args.__dict__.items()}
    return args

# From https://github.com/kunaldahiya/pyxclib/blob/8d9af7093c32e258c1340862868ff0856a7fc235/xclib/evaluation/xc_metrics.py#L195C1-L222C25
def get_inv_prop(X_Y, dataset_name):
    if "amazon" in dataset_name.lower(): A = 0.6; B = 2.6
    elif "wiki" in dataset_name.lower() and "wikiseealso" not in dataset_name.lower(): A = 0.5; B = 0.4
    else : A = 0.55; B = 1.5
    num_instances, _ = X_Y.shape
    freqs = np.ravel(np.sum(X_Y, axis=0))
    C = (np.log(num_instances)-1)*np.power(B+1, A)
    wts = 1.0 + C*np.power(freqs+B, -A)
    return np.ravel(wts)

def load_filter_mat(fname, shape):
    filter_mat = None
    if os.path.exists(fname):
        temp = np.fromfile(fname, sep=' ').astype(int)
        temp = temp.reshape(-1, 2).T
        filter_mat = sp.coo_matrix((np.ones(temp.shape[1]), (temp[0], temp[1])), shape).tocsr()
    return filter_mat

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def dump_diff_config(config_fname, config_dict):
    if os.path.exists(config_fname):
        with open(config_fname, 'a+') as f:
            print('# New experiment', file=f)
            prev_config = yaml.safe_load(open(config_fname))
            config_dict = {k: tuple(v) if isinstance(v, list) else v for k, v in config_dict.items()}
            prev_config = {k: tuple(v) if isinstance(v, list) else v for k, v in prev_config.items()}
            diff_config = dict(set(config_dict.items()) - set(prev_config.items()))
            print('', file=f)
            if len(diff_config) > 0:
                yaml.safe_dump(diff_config, f)
                return diff_config
    else:
        yaml.safe_dump(config_dict, open(config_fname, 'w'))

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def get_text(x, text, X_Xf, sep=' ', K=-1, attr='bold underline'):
    if K == -1: K = X_Xf[x].nnz
    sorted_inds = X_Xf[x].indices[np.argsort(-X_Xf[x].data)][:K]
    return '%d : \n'%x + sep.join(['%s(%.2f, %d)'%(_c(text[i], attr=attr), X_Xf[x, i], i) for i in sorted_inds])

import decimal
def myprint(*args, sep = ' ', end = '\n'):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args), sep = sep, end = end)

def drange(x, y, jump):
    x = decimal.Decimal(x)
    y = decimal.Decimal(y)
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)

def _filter(score_mat, filter_mat, copy=True):
    if filter_mat is None:
        return score_mat
    if copy:
        score_mat = score_mat.copy()
    
    temp = filter_mat.tocoo()
    score_mat[temp.row, temp.col] = 0
    del temp
    score_mat = score_mat.tocsr()
    score_mat.eliminate_zeros()
    return score_mat

def get_sorted_spmat(spmat):
    coo = spmat.tocoo()
    temp = np.array([coo.col, -coo.data, coo.row])
    temp = temp[:, np.lexsort(temp)]
    del coo

    inds, cnts = np.unique(temp[2].astype(np.int32), return_counts=True)
    indptr = np.zeros_like(spmat.indptr)
    indptr[inds+1] = cnts
    indptr = np.cumsum(indptr)

    new_spmat = csr_matrix((-temp[1], temp[0].astype(np.int32), indptr), (spmat.shape))
    del inds, cnts, indptr, temp
    return new_spmat

# From https://github.com/amzn/pecos/blob/845ca6a66ae90e6e0e4df5e735224e3772110045/pecos/utils/smat_util.py#L174
def sorted_csr_from_coo(shape, row_idx, col_idx, val, only_topk=None):
    csr = sp.csr_matrix((val, (row_idx, col_idx)), shape=shape)
    csr.sort_indices()
    for i in range(shape[0]):
        rng = slice(csr.indptr[i], csr.indptr[i + 1])
        sorted_idx = np.argsort(-csr.data[rng], kind="mergesort")
        csr.indices[rng] = csr.indices[rng][sorted_idx]
        csr.data[rng] = csr.data[rng][sorted_idx]
    if only_topk is not None:
        assert isinstance(only_topk, int), f"Wrong type: type(only_topk) = {type(only_topk)}"
        only_topk = max(min(1, only_topk), only_topk)
        nnz_of_insts = csr.indptr[1:] - csr.indptr[:-1]
        row_idx = np.repeat(np.arange(shape[0], dtype=csr.indices.dtype), nnz_of_insts)
        selected_idx = (np.arange(len(csr.data)) - csr.indptr[row_idx]) < only_topk
        row_idx = row_idx[selected_idx]
        col_idx = csr.indices[selected_idx]
        val = csr.data[selected_idx]
        indptr = np.cumsum(np.bincount(row_idx + 1, minlength=(shape[0] + 1)))
        csr = sp.csr_matrix((val, col_idx, indptr), shape=shape, dtype=val.dtype)
    return csr

def sorted_csr(csr, only_topk=None):
    if not isinstance(csr, sp.csr_matrix):
        raise ValueError("the input matrix must be a csr_matrix.")

    row_idx = np.repeat(np.arange(csr.shape[0], dtype=np.uint32), csr.indptr[1:] - csr.indptr[:-1])
    return sorted_csr_from_coo(csr.shape, row_idx, csr.indices, csr.data, only_topk)

def sp_rank(csr):
    rank_mat = sorted_csr(csr)
    rank_mat.data = np.concatenate([np.arange(1, x+1) for x in rank_mat.getnnz(1)])
    return rank_mat

def _topk(rank_mat, K, inplace=False):
    topk_mat = rank_mat if inplace else rank_mat.copy()
    topk_mat.data[topk_mat.data > K] = 0
    topk_mat.eliminate_zeros()
    return topk_mat

def _compute_xmc_metrics(rank_intrsxn_mat, true_mat, K=[1,3,5,10,20,50,100], inv_prop=None):
    K = sorted(K, reverse=True)
    topk_intrsxn_mat = rank_intrsxn_mat.copy()
    res = {'P': {}, 'R': {}, 'nDCG': {}, 'MRR': {}}
    if inv_prop is not None:
        res['PSP'] = {}
        psp_true_mat = true_mat.copy()
        psp_true_mat.data[:] = 1
        psp_true_mat.data *= inv_prop[psp_true_mat.indices]

    for k in K:
        topk_intrsxn_mat = _topk(topk_intrsxn_mat, k, inplace=True)
        res['R'][k] = (topk_intrsxn_mat.getnnz(1)/true_mat.getnnz(1)).mean()*100.0
        res['P'][k] = (topk_intrsxn_mat.getnnz(1)/k).mean()*100.0
        
        temp_topk_intrsxn_mat = topk_intrsxn_mat.copy()
        temp_topk_intrsxn_mat.data = 1/np.log2(1+temp_topk_intrsxn_mat.data)
        dcg_coeff = 1/np.log2(np.arange(k)+2)
        dcg_coeff_cumsum = np.cumsum(dcg_coeff, 0)
        dcg_denom = dcg_coeff_cumsum[np.minimum(true_mat.getnnz(1), k)-1]
        res['nDCG'][k] = (temp_topk_intrsxn_mat.sum(1).squeeze()/dcg_denom).mean()*100.0

        temp_topk_intrsxn_mat = topk_intrsxn_mat.copy()
        temp_topk_intrsxn_mat.data = 1/temp_topk_intrsxn_mat.data
        max_rr = temp_topk_intrsxn_mat.max(axis=1).toarray().ravel()
        res['MRR'][k] = max_rr.mean()*100.0

        if inv_prop is not None:
            temp_topk_intrsxn_mat = topk_intrsxn_mat.copy()
            temp_topk_intrsxn_mat.data[:] = 1
            temp_topk_intrsxn_mat.data *= inv_prop[temp_topk_intrsxn_mat.indices]
            psp_topk_true_mat = sorted_csr(psp_true_mat, k)
            psp_denom = (psp_topk_true_mat.sum(1)/k).mean()
            res['PSP'][k] = (temp_topk_intrsxn_mat.sum(1)/k).mean()*100.0 / psp_denom

    return res
    
def compute_xmc_metrics(score_mat, X_Y, inv_prop, K=100, disp = True, fname = None, name = 'Method'): 
    Ks = np.array([1,3,5,10,50,100], dtype=np.int32)
    if K <= 100: Ks = Ks[~(Ks > K)]
    else: Ks = np.concatenate([Ks, np.array([100*i for i in range(2, 1+(K//100))], dtype=np.int32)])
    X_Y = X_Y.copy().tocsr()
    X_Y.data[:] = 1
    rank_mat = sp_rank(score_mat)
    rank_intrsxn_mat = rank_mat.multiply(X_Y)
    xmc_eval_metrics = pd.DataFrame(_compute_xmc_metrics(rank_intrsxn_mat, X_Y, K=Ks.tolist(), inv_prop=inv_prop)).round(2).transpose() 
    df = xmc_eval_metrics.stack().to_frame().transpose()
    df.columns = [f'{col[0]}@{col[1]}' for col in df.columns.values]
    df.index = [name]
    psp_cols = ['PSP@1', 'PSP@3', 'PSP@5'] if 'PSP@1' in df.columns else []
    recall_cols = [f'R@{k}' for k in Ks[Ks >= 10]]
    df = df[[*['P@1', 'P@3', 'P@5', 'nDCG@1', 'nDCG@3', 'nDCG@5', 'MRR@10'], *psp_cols, *recall_cols]].round(2)

    if disp:
        print(df.to_csv(sep='\t', index=False))
        print(df.to_csv(sep=' ', index=False))
    if fname is not None:
        if os.path.splitext(fname)[-1] == '.json': df.to_json(fname)
        elif os.path.splitext(fname)[-1] == '.csv': df.to_csv(fname)  
        elif os.path.splitext(fname)[-1] == '.tsv': df.to_csv(fname, sep='\t')  
        else: print(f'ERROR: File extension {os.path.splitext(fname)[-1]} in {fname} not supported')
    return df

class bcolors:
    purple = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    warn = '\033[93m' # dark yellow
    fail = '\033[91m' # dark red
    white = '\033[37m'
    yellow = '\033[33m'
    red = '\033[31m'
    
    ENDC = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'
    reverse = '\033[7m'
    
    on_grey = '\033[40m'
    on_yellow = '\033[43m'
    on_red = '\033[41m'
    on_blue = '\033[44m'
    on_green = '\033[42m'
    on_magenta = '\033[45m'
    
def _c(*args, attr='bold'):
    string = ''.join([bcolors.__dict__[a] for a in attr.split()])
    string += ' '.join([str(arg) for arg in args])+bcolors.ENDC
    return string

def vis_point(x, spmat, X, Y, nnz, true_mat, sep='', K=-1, expand=False, trnx_nnz=None, trn_Y_X=None, trnX=None):
    if K == -1: K = spmat[x].nnz
        
    sorted_inds = spmat[x].indices[np.argsort(-spmat[x].data)][:K]
    print(f'x[{x}]: {_c(X[x], attr="bold")}\n')
    for i, ind in enumerate(sorted_inds):
        myattr = ""
        if true_mat[x, ind] > 0.1: myattr="yellow"
        print(f'{i+1}) {_c(Y[ind], attr=myattr)} [{ind}] ({"%.4f"%spmat[x, ind]}, {nnz[ind]})')
        if expand:
            for j, trn_ind in enumerate(trn_Y_X[ind].indices[:10]):
                print(f'\t{j+1}) {_c(trnX[trn_ind], attr="green")} [{trn_ind}] ({trnx_nnz[trn_ind]})')
        print(sep)

def get_decile_mask(X_Y, N=10):
    nnz = X_Y.getnnz(0)
    sorted_inds = np.argsort(-nnz)
    cumsum_sorted_nnz = nnz[sorted_inds].cumsum()

    deciles = [sorted_inds[np.where((cumsum_sorted_nnz > i*nnz.sum()/N) & (cumsum_sorted_nnz <= (i+1)*nnz.sum()/N))[0]] for i in range(N)]
    decile_mask = np.zeros((N, X_Y.shape[1]), dtype=np.bool)
    for i in range(N):
        decile_mask[i, deciles[i]] = True
    return decile_mask