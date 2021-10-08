import os
import GPUtil
import time
import argparse
#########################################################################
# setup CUDA_VISIBLE_DEVICES before importing pytorch
def setup_GPU(ngpu=1, xsize=10000):
    ## setup GPU
    ## detect and use the first available GPUs
    gpus = GPUtil.getGPUs()
    print("Detect available gpus: ")
    idxs = []
    mems = []
    counter = 0
    while len(idxs) == 0:
        for ii,gpu in enumerate(gpus):
            if gpu.memoryFree > xsize:
                idxs.append(ii)
                mems.append(gpu.memoryFree)
        if len(idxs) == 0:
            time.sleep(60)
            counter += 1
            if counter%(60*12) == 0:
                print("%d hours passed"%(counter/60))
    idxs = [v for _, v in sorted(zip(mems, idxs), reverse=True)]
    idxs = sorted(idxs[:ngpu])
    # idxs = [3,]
    GPU_IDS = ",".join([str(v) for v in idxs])
    print("Use Gpu, ", GPU_IDS)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    return list(range(len(idxs)))

setup_GPU()

from tqdm import tqdm
import pandas as pd
import trimesh
import torch
from im2mesh import config
from im2mesh import data as data_io
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.io import load_pointcloud
from im2mesh.checkpoints import CheckpointIO
from im2mesh.net_optim import NetOptim

def eval():
    parser = argparse.ArgumentParser(
        description='Evaluate mesh algorithms.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

    # Get configuration and basic arguments
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")

    # Shorthands
    out_dir = cfg['training']['out_dir']
    out_file = os.path.join(out_dir, 'eval_full.pkl')
    out_file_class = os.path.join(out_dir, 'eval.csv')

    # Dataset
    dataset = config.get_dataset('test', cfg)
    model = config.get_model(cfg, device=device, dataset=dataset)

    # Model, checkpoint_saver, net_trainer,
    optimizer, lr_scheduler = config.get_optimizer(cfg, model, "all")
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)
    net_optim = NetOptim(cfg, model, trainer, checkpoint_io)
    net_optim.restore_model(cfg['test']['model_file'])
    net_optim.print_net_params()

    # Evaluate
    eval_dicts = []
    print('Evaluating networks...')

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=data_io.collate_remove_none,
        worker_init_fn=data_io.worker_init_fn)


    ## TODO: handle per category metric in xx
    # Handle each dataset separately
    time_eval = 0.0
    t_start = time.time()
    for it, data in enumerate(tqdm(test_loader)):
        if data is None:
            print('Invalid data.')
            continue
        # Get index etc.
        idx = data['idx'].item()

        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}

        modelname = model_dict['model']
        category_id = model_dict['category']

        try:
            category_name = dataset.metadata[category_id].get('name', 'n/a')
        except AttributeError:
            category_name = category_id
        if category_name == 'n/a':
            category_name = category_id

        eval_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname':modelname,
        }
        eval_dicts.append(eval_dict)
        eval_data = trainer.eval_step(data)[-1]
        eval_dict.update(eval_data)
    time_eval += time.time() - t_start
    print("Time summary: %s"%(time_eval/len(test_loader)))


    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)
    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class name']).mean()
    eval_df_class.to_csv(out_file_class)

    # Print results
    eval_df_class.loc['mean'] = eval_df_class.mean()
    print(eval_df_class)

eval()
