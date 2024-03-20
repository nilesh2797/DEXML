from accelerate import Accelerator
accelerator = Accelerator()
IS_MAIN_PROC = accelerator.is_main_process

import sys, yaml, logging
import logging.config
import scipy.sparse as sp

from nets import *
from utils.helper_utils import _c, load_config_and_runtime_args, _filter
from datasets import DATA_MANAGERS, XMCEvaluator
from utils.dl_utils import unwrap

import transformers
transformers.set_seed(42)
if torch.__version__ > "1.11":
    torch.backends.cuda.matmul.allow_tf32 = True

# Config and runtime argument parsing
mode = sys.argv[1]
args = load_config_and_runtime_args(sys.argv[1:])
    
args.device = str(accelerator.device)
args.amp = accelerator.state.use_fp16
args.num_gpu = accelerator.state.num_processes
args.DATA_DIR = DATA_DIR = f'Datasets/{args.dataset}'
args.resume_path = f'{args.OUT_DIR}/model.pt'

if IS_MAIN_PROC:
    with open('configs/logging.yaml') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file_handler']['filename'] = f"{args.OUT_DIR}/{log_config['handlers']['file_handler']['filename']}"
        logging.config.dictConfig(log_config)

logging.info(f'Starting {" ".join(sys.argv)}')
logging.info(f'Experiment name: {_c(args.expname, attr="blue")}, Dataset: {_c(args.dataset, attr="blue")}')
logging.info(f'Wandb ID: {args.wandb_id}')

# Data loading
data_manager = DATA_MANAGERS[args.data_manager](args)
trn_loader, val_loader, tst_loader = data_manager.build_data_loaders()
if mode == 'trn': data_loader = trn_loader
elif mode == 'val': data_loader = val_loader
elif mode == 'tst': data_loader = tst_loader

accelerator.wait_for_everyone()
if 'dist-de' not in args.net:
    net = NETS[args.net](args)
else:
    net = NETS[args.net](args, accelerator, data_manager)
logging.info(f'Loading net state dict from: {args.resume_path}')
logging.info(net.load(args.resume_path))
net, data_loader = accelerator.prepare(net, data_loader)

K = args.eval_topk
net.eval()
score_mat = unwrap(net).predict(data_loader, bsz=args.bsz*2, K=args.eval_topk, accelerator=accelerator)

if IS_MAIN_PROC: 
    evaluator = XMCEvaluator(args, data_loader, data_manager, prefix=mode)
    metrics = evaluator.eval(score_mat)
    logging.info('\n'+metrics.to_csv(sep='\t', index=False))
    sp.save_npz(f'{args.OUT_DIR}/{evaluator.prefix}_score_mat.npz', score_mat)

accelerator.wait_for_everyone()