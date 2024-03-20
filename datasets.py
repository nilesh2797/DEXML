import os, torch
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing
from transformers import AutoTokenizer

from utils.helper_utils import _c, get_inv_prop, load_filter_mat, _filter, compute_xmc_metrics
from utils.dl_utils import unwrap, csr_to_bow_tensor, csr_to_pad_tensor, bert_fts_batch_to_tensor, expand_multilabel_dataset

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, labels: sp.csr_matrix, sample=None, filter_mat=None):
        super().__init__()
        self.sample = np.arange(labels.shape[0]) if sample is None else sample
        self.labels = labels[self.sample]
        self.filter_mat = filter_mat[self.sample] if filter_mat is not None else None

    def __getitem__(self, index):
        return {'index': index}

    def __len__(self):
        return len(self.sample)

class SimpleDataset(BaseDataset):
    def __init__(self, features, labels, **super_kwargs):
        super().__init__(labels, **super_kwargs)
        self.features = features
    
    def get_fts(self, indices):
        if isinstance(self.features, sp.csr_matrix):
            return csr_to_bow_tensor(self.features[self.sample[indices]])
        else:
            return torch.Tensor(self.features[self.sample[indices]])

class OfflineBertDataset(BaseDataset):
    def __init__(self, fname, labels, max_len, token_type='bert-base-uncased', **super_kwargs):
        super().__init__(labels, **super_kwargs)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(token_type)
        nr, nc, dtype = open(f'{fname}.meta').readline().split()
        self.X_ii = np.memmap(f"{fname}", mode='r', shape=(int(nr), int(nc)), dtype=dtype)
    
    def get_fts(self, indices):
        X_ii = np.array(self.X_ii[self.sample[indices]]).reshape(-1, self.X_ii.shape[1])
        X_am = (X_ii != self.tokenizer.pad_token_id)
        return bert_fts_batch_to_tensor(X_ii, X_am)
    
class OnlineBertDataset(BaseDataset):
    def __init__(self, X, labels, max_len, token_type='bert-base-uncased', **super_kwargs):
        super().__init__(labels, **super_kwargs)
        self.max_len = max_len
        self.X = np.array(X, dtype=object)
        self.tokenizer = AutoTokenizer.from_pretrained(token_type)
    
    def get_fts(self, indices):
        return self.tokenizer.batch_encode_plus(list(self.X[self.sample[indices]]), 
                                                max_length=self.max_len, 
                                                padding=True, 
                                                truncation=True, 
                                                return_tensors='pt', 
                                                return_token_type_ids=False).data

class XMCCollator():
    def __init__(self, dataset):
        self.dataset = dataset
        self.numy = self.dataset.labels.shape[1]
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = torch.LongTensor([b['index'] for b in batch])
        
        b = {'batch_size': torch.LongTensor([batch_size]),
             'numy': torch.LongTensor([self.numy]),
             'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy),
             'ids': ids,
             'xfts': self.dataset.get_fts(ids)}
             
        return b

class TwoTowerDataset(BaseDataset):
    def __init__(self, x_dataset, y_dataset, shorty=None):
        super().__init__(labels=x_dataset.labels, filter_mat=x_dataset.filter_mat)
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.shorty = shorty
            
    def __getitem__(self, index):
        ret = {'index': index}
        return ret

    def get_fts(self, indices, source):
        if source == 'x':
            return self.x_dataset.get_fts(indices)
        elif source == 'y':
            return self.y_dataset.get_fts(indices)

    def __len__(self):
        return self.labels.shape[0]
    
class CrossDataset(BaseDataset):
    def __init__(self, x_dataset, y_dataset, shorty=None, iterate_over='labels'):
        super().__init__(labels=x_dataset.labels, filter_mat=x_dataset.filter_mat)
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.iterate_over = iterate_over
        self.labels = self.x_dataset.labels
        self.shorty = shorty
        # Assumption: always binarizing shortlist
        self.shorty.data[:] = 1.0

        if self.iterate_over == 'labels':
            self.rows, self.cols = self.labels.nonzero()
            self.shorty = _filter(self.shorty, self.labels, copy=False)
        elif self.iterate_over == 'shorty':
            self.rows, self.cols = self.shorty.nonzero()
            
    def __getitem__(self, index):
        ret = {'index': index}
        return ret

    def get_fts(self, indices, source):
        if source == 'x':
            return self.x_dataset.get_fts(indices)
        elif source == 'y':
            return self.y_dataset.get_fts(indices)

    def __len__(self):
        return self.rows.shape[0]

class CrossCollator():
    def __init__(self, dataset: CrossDataset, num_neg_samples=1):
        self.numy = dataset.labels.shape[1]
        self.dataset = dataset
        self.num_neg_samples = num_neg_samples
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = torch.LongTensor([b['index'] for b in batch])
        
        b = {'batch_size': torch.LongTensor([batch_size]),
             'numy': torch.LongTensor([self.numy]),
             'ids': ids
             }
        
        b_query_inds = torch.tensor(self.dataset.rows[ids])
        b_label_inds = torch.tensor(self.dataset.cols[ids])
        b_targets = None
        
        if self.dataset.shorty is not None and self.num_neg_samples > 0 and self.dataset.iterate_over == 'labels':
            batch_shorty = csr_to_pad_tensor(self.dataset.shorty[b_query_inds], self.numy-1)
            batch_shorty_inds = torch.multinomial(torch.maximum(batch_shorty['vals'].double(), torch.tensor(1e-8)), self.num_neg_samples)
            batch_shorty = torch.gather(batch_shorty['inds'], 1, batch_shorty_inds)

            b_query_inds = b_query_inds.unsqueeze(1).repeat(1, self.num_neg_samples+1).view(-1)
            b_targets = torch.hstack([torch.ones(*b_label_inds.shape).unsqueeze(1), torch.zeros(*batch_shorty.shape)]).view(-1)
            b_label_inds = torch.hstack([b_label_inds.unsqueeze(1), batch_shorty]).view(-1)

        b['xfts'] = self.dataset.get_fts(b_query_inds, 'x')
        b['yfts'] = self.dataset.get_fts(b_label_inds, 'y')
        b['query_inds'] = b_query_inds
        b['label_inds'] = b_label_inds
        b['targets'] = b_targets
            
        return b

class TwoTowerTrainCollator():
    def __init__(self, dataset: TwoTowerDataset, neg_type='batch-rand', num_neg_samples=1, num_pos_samples=1):
        self.numy = dataset.labels.shape[1]
        self.dataset = dataset
        self.neg_type = neg_type
        self.num_neg_samples = num_neg_samples
        self.num_pos_samples = num_pos_samples
        self.mask = torch.zeros(self.numy+1).long()
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = np.array([b['index'] for b in batch])
        
        batch_data = {'batch_size': batch_size,
                      'numy': self.numy,
                      'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy),
                      'ids': torch.Tensor([b['index'] for b in batch]).long(),
                      'xfts': self.dataset.get_fts(ids, 'x')
                     }
        
        batch_y = None
        
        if self.neg_type == 'shorty':
            batch_shorty = csr_to_pad_tensor(self.dataset.shorty[ids], self.numy-1)

            if self.num_neg_samples > 0:
                batch_shorty_inds = torch.multinomial(torch.maximum(batch_shorty['vals'].double(), torch.tensor(1e-8)), self.num_neg_samples)
                batch_shorty = torch.gather(batch_shorty['inds'], 1, batch_shorty_inds).squeeze()
            else:
                batch_shorty = np.array([], dtype=np.int64)
            batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), min(self.num_pos_samples, batch_data['y']['vals'].shape[1]))
            batch_pos_y = torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze()
            batch_y = torch.LongTensor(np.union1d(batch_pos_y, batch_shorty))
            batch_y = batch_y[batch_y != self.numy]
            
            self.mask[batch_y] = torch.arange(batch_y.shape[0])
            batch_data['pos-inds'] = self.mask[batch_pos_y].reshape(-1, 1)
            batch_data['shorty-inds'] = self.mask[batch_shorty].reshape(-1, 1) if len(batch_shorty) > 0 else torch.zeros((0, 1), dtype=torch.long)
            self.mask[batch_y] = 0
            
            batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
            for i in range(batch_size):
                self.mask[batch_data['y']['inds'][i]] = True
                batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
                self.mask[batch_data['y']['inds'][i]] = False
        
        elif self.neg_type == 'in-batch':
            batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), min(self.num_pos_samples, batch_data['y']['vals'].shape[1]))
            batch_y = torch.unique(torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze())
            if self.numy == batch_y[-1]:
                batch_y = batch_y[:-1]
            batch_data['pos-inds'] = torch.arange(batch_size).reshape(-1, 1)
            batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
            for i in range(batch_size):
                self.mask[batch_data['y']['inds'][i]] = True
                batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
                self.mask[batch_data['y']['inds'][i]] = False

        elif self.neg_type == 'all':
            batch_y = torch.arange(self.numy)
            batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]+1)).scatter_(1, batch_data['y']['inds'], 1.0)[:,:-1]
          
        if batch_y is not None:
            batch_data['batch_y'] = batch_y
            batch_data['yfts'] = self.dataset.get_fts(batch_y.numpy(), 'y')

        return batch_data

class XMCDataManager():
    def __init__(self, args):
        self.trn_X_Y = sp.load_npz(f'{args.DATA_DIR}/Y.trn.npz')
        self.tst_X_Y = sp.load_npz(f'{args.DATA_DIR}/Y.tst.npz')
        self.tst_filter_mat = load_filter_mat(f'{args.DATA_DIR}/filter_labels_test.txt', self.tst_X_Y.shape)
        self.trn_filter_mat = load_filter_mat(f'{args.DATA_DIR}/filter_labels_train.txt', self.trn_X_Y.shape)
        self.inv_prop = get_inv_prop(self.trn_X_Y, args.dataset)


        self.numy = args.numy = self.trn_X_Y.shape[1] # Number of labels
        self.trn_numx = self.trn_X_Y.shape[0] # Number of train data points 
        self.tst_numx = self.tst_X_Y.shape[0] # Number of test data points

        self.data_tokenization = args.data_tokenization
        self.tf_max_len = args.tf_max_len
        self.tf_token_type = args.tf_token_type = 'roberta-base' if 'roberta' in args.tf else 'bert-base-uncased' if 'bert' in args.tf else args.tf # Token type
        self.DATA_DIR = args.DATA_DIR
        self.num_val_points = args.num_val_points
        self.bsz = args.bsz

        if self.num_val_points > 0:
            if os.path.exists(f'{args.DATA_DIR}/val_inds_{args.num_val_points}.npy'): 
                self.val_inds = np.load(f'{args.DATA_DIR}/val_inds_{args.num_val_points}.npy')
            else: 
                self.val_inds = np.random.choice(np.arange(self.trn_numx), size=args.num_val_points, replace=False)
                np.save(f'{args.DATA_DIR}/val_inds_{args.num_val_points}.npy', self.val_inds)
            self.trn_inds = np.setdiff1d(np.arange(self.trn_numx), self.val_inds)
        else:
            self.trn_inds = self.val_inds = None

    def load_raw_texts(self):
        self.trnX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/trn_X.txt')]
        self.tstX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/tst_X.txt')]
        self.Y = [x.strip() for x in open(f'{self.DATA_DIR}/raw/Y.txt')]
        return self.trnX, self.tstX, self.Y

    def load_bow_fts(self, normalize=True):
        trn_X_Xf = sp.load_npz(f'{self.DATA_DIR}/X.trn.npz')
        tst_X_Xf = sp.load_npz(f'{self.DATA_DIR}/X.tst.npz')

        if normalize:
            sklearn.preprocessing.normalize(trn_X_Xf, copy=False)
            sklearn.preprocessing.normalize(tst_X_Xf, copy=False)

        self.trn_X_Xf = trn_X_Xf[self.trn_inds] if self.trn_inds is not None else trn_X_Xf
        self.val_X_Xf = trn_X_Xf[self.val_inds] if self.val_inds is not None else tst_X_Xf
        self.tst_X_Xf = tst_X_Xf

        return self.trn_X_Xf, self.val_X_Xf, self.tst_X_Xf

    def build_datasets(self):
        if self.data_tokenization == 'offline':
            self.trn_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/trn_X.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.trn_inds, filter_mat=self.trn_filter_mat)
            self.val_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/trn_X.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.val_inds, filter_mat=self.trn_filter_mat)
            self.tst_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/tst_X.{self.tf_token_type}_{self.tf_max_len}.dat', self.tst_X_Y, self.tf_max_len, self.tf_token_type, sample = None, filter_mat=self.tst_filter_mat)
        elif self.data_tokenization == 'online':
            trnX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/trn_X.txt').readlines()]
            tstX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/tst_X.txt').readlines()]
            self.trn_dataset = OnlineBertDataset(trnX, self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.trn_inds, filter_mat=self.trn_filter_mat)
            self.val_dataset = OnlineBertDataset(trnX, self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.val_inds, filter_mat=self.trn_filter_mat)
            self.tst_dataset = OnlineBertDataset(tstX, self.tst_X_Y, self.tf_max_len, self.tf_token_type, sample=None, filter_mat=self.tst_filter_mat)
        else:
            raise Exception(f"Unrecongnized data_tokenization argument: {self.data_tokenization}")
        
        if self.num_val_points <= 0:
            self.val_dataset = self.tst_dataset

        return self.trn_dataset, self.val_dataset, self.tst_dataset

    def build_data_loaders(self):
        if not hasattr(self, "trn_dataset"):
            self.build_datasets()

        data_loader_args = {
            'batch_size': self.bsz,
            'num_workers': 4,
            'collate_fn': XMCCollator(self.trn_dataset),
            'shuffle': True,
            'pin_memory': True
        }

        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, **data_loader_args)

        data_loader_args['shuffle'] = False
        data_loader_args['collate_fn'] = XMCCollator(self.val_dataset)
        data_loader_args['batch_size'] = 2*self.bsz
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **data_loader_args)

        data_loader_args['collate_fn'] = XMCCollator(self.tst_dataset)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, **data_loader_args)

        return self.trn_loader, self.val_loader, self.tst_loader

class XMCEmbedDataManager(XMCDataManager):
    def __init__(self, args):
        super().__init__(args)
        self.embed_id = args.embed_id
    
    def build_datasets(self):
        trn_embs = np.load(f'{self.DATA_DIR}/embeddings/{self.embed_id}/trn_embs.npy')
        tst_embs = np.load(f'{self.DATA_DIR}/embeddings/{self.embed_id}/tst_embs.npy')
        self.trn_dataset = SimpleDataset(trn_embs, self.trn_X_Y, sample=self.trn_inds, filter_mat=self.trn_filter_mat)
        self.val_dataset = SimpleDataset(trn_embs, self.trn_X_Y, sample=self.val_inds, filter_mat=self.trn_filter_mat)
        self.tst_dataset = SimpleDataset(tst_embs, self.tst_X_Y, sample=None, filter_mat=self.tst_filter_mat)
        if self.num_val_points <= 0:
            self.val_dataset = self.tst_dataset
        return self.trn_dataset, self.val_dataset, self.tst_dataset

class TwoTowerDataManager(XMCDataManager):
    def __init__(self, args):
        super().__init__(args)
        self.transpose_trn_dataset = args.transpose_trn_dataset
        self.neg_type = args.neg_type
        self.only_keep_trn_labels = args.only_keep_trn_labels if hasattr(args, 'only_keep_trn_labels') else False
        self.num_neg_samples = args.num_neg_samples if hasattr(args, 'num_neg_samples') else 1
        self.num_pos_samples = args.num_pos_samples if hasattr(args, 'num_pos_samples') else 1
        self.trn_shorty = sp.load_npz(args.trn_shorty) if hasattr(args, 'trn_shorty') and os.path.exists(args.trn_shorty) else None

    def build_datasets(self):
        trnx_dataset, valx_dataset, tstx_dataset = super().build_datasets()
        if self.only_keep_trn_labels:
            lbl_sample = np.union1d(np.where(self.trn_X_Y.getnnz(0).ravel() > 0)[0], np.where(self.tst_X_Y.getnnz(0).ravel() > 0)[0])
            trnx_dataset.labels = trnx_dataset.labels[:, lbl_sample]
            tstx_dataset.labels = tstx_dataset.labels[:, lbl_sample]
            if self.num_val_points > 0:
                valx_dataset.labels = valx_dataset.labels[:, lbl_sample]
        else:
            lbl_sample = None

        if self.data_tokenization == 'offline':
            self.lbl_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/Y.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=lbl_sample, filter_mat=None)
        elif self.data_tokenization == 'online':
            Y = [x.strip() for x in open(f'{self.DATA_DIR}/raw/Y.txt').readlines()]
            self.lbl_dataset = OnlineBertDataset(Y, self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=lbl_sample, filter_mat=None)
        else:
            raise Exception(f"Unrecongnized data_tokenization argument: {self.data_tokenization}")
    
        
        if self.transpose_trn_dataset:
            assert (self.trn_shorty is None) or (self.trn_shorty.shape == self.lbl_dataset.shape)
            # self.lbl_dataset.sample = np.where(self.lbl_dataset.labels.getnnz(1).ravel() > 0)[0]
            self.trn_dataset = TwoTowerDataset(self.lbl_dataset, trnx_dataset, self.trn_shorty)
        else:
            self.trn_dataset = TwoTowerDataset(trnx_dataset, self.lbl_dataset, self.trn_shorty)
        self.val_dataset = TwoTowerDataset(valx_dataset, self.lbl_dataset)
        self.tst_dataset = TwoTowerDataset(tstx_dataset, self.lbl_dataset)

        return self.trn_dataset, self.val_dataset, self.tst_dataset

    def build_data_loaders(self):
        if not hasattr(self, "trn_dataset"):
            self.build_datasets()

        print('neg_type:', self.neg_type)
        data_loader_args = {
            'batch_size': self.bsz,
            'num_workers': 4,
            'collate_fn': TwoTowerTrainCollator(self.trn_dataset, neg_type=self.neg_type, num_neg_samples=self.num_neg_samples, num_pos_samples=self.num_pos_samples),
            'shuffle': True,
            'pin_memory': True
        }

        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, **data_loader_args)

        data_loader_args['shuffle'] = False
        data_loader_args['collate_fn'] = None
        data_loader_args['batch_size'] = 2*self.bsz
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **data_loader_args)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, **data_loader_args)

        return self.trn_loader, self.val_loader, self.tst_loader

class CrossDataManager(XMCDataManager):
    def __init__(self, args):
        super().__init__(args)
        self.transpose_trn_dataset = args.transpose_trn_dataset
        self.num_neg_samples = args.num_neg_samples if hasattr(args, 'num_neg_samples') else 1
        self.trn_shorty = sp.load_npz(args.trn_shorty) if hasattr(args, 'trn_shorty') and os.path.exists(args.trn_shorty) else None
        self.tst_shorty = sp.load_npz(args.tst_shorty) if hasattr(args, 'tst_shorty') and os.path.exists(args.tst_shorty) else None
        if self.num_val_points > 0 and self.trn_shorty is not None:
            self.val_shorty = self.trn_shorty[self.val_inds]
            self.trn_shorty = self.trn_shorty[self.trn_inds]
        else:
            self.val_shorty = self.tst_shorty

    def build_datasets(self):
        trnx_dataset, valx_dataset, tstx_dataset = super().build_datasets()
        if self.data_tokenization == 'offline':
            self.lbl_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/Y.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=None, filter_mat=None)
        elif self.data_tokenization == 'online':
            Y = [x.strip() for x in open(f'{self.DATA_DIR}/raw/Y.txt').readlines()]
            self.lbl_dataset = OnlineBertDataset(Y, self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=None, filter_mat=None)
        else:
            raise Exception(f"Unrecongnized data_tokenization argument: {self.data_tokenization}")
        
        if self.transpose_trn_dataset:
            self.trn_dataset = CrossDataset(self.lbl_dataset, trnx_dataset, self.lbl_dataset.labels, iterate_over='labels')
            # TODO: make trn_shorty work with transpose
        else:
            self.trn_dataset = CrossDataset(trnx_dataset, self.lbl_dataset, self.trn_shorty, iterate_over='labels')
        self.val_dataset = CrossDataset(valx_dataset, self.lbl_dataset, self.val_shorty + valx_dataset.labels, iterate_over='shorty')
        self.tst_dataset = CrossDataset(tstx_dataset, self.lbl_dataset, self.tst_shorty, iterate_over='shorty')

        return self.trn_dataset, self.val_dataset, self.tst_dataset

    def build_data_loaders(self):
        if not hasattr(self, "trn_dataset"):
            self.build_datasets()

        data_loader_args = {
            'batch_size': self.bsz,
            'num_workers': 4,
            'collate_fn': CrossCollator(self.trn_dataset, num_neg_samples=self.num_neg_samples),
            'shuffle': True,
            'pin_memory': True
        }

        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, **data_loader_args)

        data_loader_args['shuffle'] = False
        data_loader_args['batch_size'] = 4*self.bsz
        data_loader_args['collate_fn'] = CrossCollator(self.val_dataset, num_neg_samples=0)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **data_loader_args)
        data_loader_args['collate_fn'] = CrossCollator(self.tst_dataset, num_neg_samples=0)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, **data_loader_args)

        return self.trn_loader, self.val_loader, self.tst_loader

class XMCEvaluator:
    def __init__(self, args, data_source, data_manager: XMCDataManager, prefix='default'):
        self.eval_interval = args.eval_interval
        self.num_epochs = args.num_epochs
        self.track_metric = args.track_metric
        self.OUT_DIR = args.OUT_DIR
        self.save = args.save
        self.bsz = args.bsz
        self.eval_topk = args.eval_topk
        self.wandb_id = args.wandb_id if hasattr(args, "wandb_id") else None
        self.prefix = prefix

        self.data_source = data_source
        self.labels = data_source.labels if isinstance(data_source, torch.utils.data.Dataset) else data_source.dataset.labels
        self.filter_mat = data_source.filter_mat if isinstance(data_source, torch.utils.data.Dataset) else data_source.dataset.filter_mat
        self.inv_prop = data_manager.inv_prop
        self.best_score = -99999999

    def predict(self, net):
        score_mat = unwrap(net).predict(self.data_source, K=self.eval_topk, bsz=self.bsz)
        return score_mat

    def eval(self, score_mat, epoch=-1, loss=float('inf')):
        _filter(score_mat, self.filter_mat, copy=False)
        eval_name = f'{self.prefix}' + [f' {epoch}/{self.num_epochs}', ''][epoch < 0]
        metrics = compute_xmc_metrics(score_mat, self.labels, self.inv_prop, K=self.eval_topk, name=eval_name, disp=False)
        metrics.index.names = [self.wandb_id]
        if loss < float('inf'):  metrics['loss'] = ["%.4E"%loss]
        metrics.to_csv(open(f'{self.OUT_DIR}/{self.prefix}_metrics.tsv', 'a+'), sep='\t', header=(epoch <= 0))
        return metrics

    def predict_and_track_eval(self, net, epoch='-', loss=float('inf')):
        if epoch%self.eval_interval == 0 or epoch == (self.num_epochs-1):
            score_mat = self.predict(net)
            return self.track_eval(net, score_mat, epoch, loss)

    def track_eval(self, net, score_mat, epoch='-', loss=float('inf')):
        if score_mat is None: 
            return None
        
        metrics = self.eval(score_mat, epoch, loss)
        if metrics.iloc[0][self.track_metric] > self.best_score:
            self.best_score = metrics.iloc[0][self.track_metric]
            print(_c(f'Found new best model with {self.track_metric}: {"%.2f"%self.best_score}\n', attr='blue'))
            if self.save:
                sp.save_npz(f'{self.OUT_DIR}/{self.prefix}_score_mat.npz', score_mat)
                net.save(f'{self.OUT_DIR}/model.pt')
        return metrics

DATA_MANAGERS = {
    'xmc': XMCDataManager,
    'two-tower': TwoTowerDataManager,
    'xmc-embed': XMCEmbedDataManager,
    'cross': CrossDataManager,
}
            