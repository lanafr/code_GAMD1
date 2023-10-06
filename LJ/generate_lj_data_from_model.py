import sys, os
sys.path.append(os.path.join('../',os.path.dirname(os.path.abspath(''))))
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
sys.path.append('/home/guests/lana_frkin/GAMDplus/code')
sys.path.append('/home/guests/lana_frkin/GAMDplus/code/LJ')
print(sys.path)

from nn_module import SimpleMDNetNew
from train_utils import LJDataNew
from train_network_lj import ParticleNetLightning, NUM_OF_ATOMS
import torch
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn

dataset = LJDataNew(dataset_path='/home/guests/lana_frkin/GAMDplus/code/LJ/md_dataset/lj_data',
                   sample_num=999,
                   case_prefix='data_',
                   seed_num=10,
                   #m_num=NUM_OF_ATOMS,
                   mode='test')
dataloader = DataLoader(dataset, num_workers=2, batch_size=1, shuffle=False,
                          collate_fn=
                          lambda batches: {
                              'pos': [batch['pos'] for batch in batches],
                              'forces': [batch['forces'] for batch in batches],
                              'pos_next': [batch['pos_next'] for batch in batches],
                          })

PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_try5_batch16_and_drop_edge_false/checkpoint_29.ckpt'
SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_try5_batch16_and_drop_edge_false/scaler_29.npz'
args = SimpleNamespace(use_layer_norm=False,
                       encoding_size=128,
                       hidden_dim=256,
                       edge_embedding_dim=256,
                      drop_edge=False,
                       conv_layer=4,
                      rotate_aug=False,
                       update_edge=False,
                       use_part=False,
                      data_dir='',
                      loss='mae')
model = ParticleNetLightning(args).load_from_checkpoint(PATH, args=args)
model.load_training_stats(SCALER_CKPT)
model.cuda()
model.eval()

"""

cos_vals = []
nextpos = []
nextpos_gt = []
mae = []
se = []
num = []
with torch.no_grad():
    count = 0
    for i_batch, batch in enumerate(dataloader):

        pos = batch['pos']
        gt = torch.from_numpy(batch['pos_next'][0]).float().cuda()
        #gt = torch.from_numpy(batch['pos_next']).float().cuda()

        nextpos_np = model.predict_nextpos(pos[0])
        nextpos_tsr = torch.from_numpy(nextpos_np).cuda()
        cos_val = torch.sum(nextpos_tsr * gt, dim=1) / (nextpos_tsr.norm(dim=1)*gt.norm(dim=1))
        #nextpos_tsr = nextpos_tsr.view(-1, 3)*(0.0010364)  # KJ/mol to ev/A
        nextpos_tsr = nextpos_tsr.view(-1,3)  # KJ/mol to ev/A
        #gt = gt.view(-1, 3)*(0.0010364)
        gt = gt.view(-1,3)
        nextpos += [nextpos_tsr.cpu().numpy()]
        nextpos_gt += [gt.cpu().numpy()]
        mae += [torch.sum(torch.mean(torch.abs(nextpos_tsr - gt), dim=1)).cpu().numpy()]
        se += [torch.sum(torch.mean((nextpos_tsr-gt)**2, dim=1)).cpu().numpy()]
        num += [pos[0].shape[0]]
        cos_vals += [cos_val]
        if count%200 == 0:
            print(f"Finished {i_batch}")
        count += 1
        
print(f'cos value: {torch.mean(torch.cat(cos_vals))}')
print(f'MAE {np.sum(mae) / np.sum(num)}')
print(f'std of MAE per sample: {np.std([mae[i]/num[i] for i in range(len(num))])}')
print(f'RMSE: {np.sqrt(np.sum(se)/ np.sum(num))}')
print(f'std of RMSE per sample: {np.std([np.sqrt(se[i]/num[i]) for i in range(len(num))])}')


"""
#this part is for generating data by the model
"""
pos_init_all = np.load('md_dataset/lj_data_to_test/data_0_0.npz')
pos_init = pos_init_all['pos']



pos = pos_init


for i in range(1000):
    np.savez(f'./md_dataset/lj_data_tested/data_test1_{i}.npz',
                 pos=pos)
    pos_next = model.predict_nextpos(pos)
    pos=pos_next
"""

#this part is to check how well the autoencoder works on a test set
pos_lst=[]
gt_lst=[]

for i in range(999):

    gt_all = np.load(f'md_dataset/lj_data_to_test/data_0_{i}.npz')
    gt = gt_all['pos']
    gt_lst.append(gt)

    pos_hopefully_same = model.predict_nextpos(gt)
    pos_lst.append(pos_hopefully_same)

gt_lst = [torch.from_numpy(arr) for arr in gt_lst]
gt_cat = torch.cat(gt_lst, dim=0)
pos_lst = [torch.from_numpy(arr) for arr in pos_lst]
pos_cat = torch.cat(pos_lst, dim=0)

mae = nn.L1Loss()(pos_cat, gt_cat)

print("Loss is:")
print(mae)