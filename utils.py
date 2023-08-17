import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime

import faiss
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

def save_pkl(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def init_cuda(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]


class TwoViewloader(Dataset):
    """
    Return the dataitem and corresponding index
    The batch of the loader: A list
        - [B, L, 1] (For univariate time series)
        - [B]: The corresponding index in the train_set tensors

    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample_tem = self.data[0][index]
        sample_fre = self.data[1][index]


        return index, sample_tem, sample_fre

    def __len__(self):
        return len(self.data[0])


def run_kmeans(x, args, last_clusters = None):
    results = {'im2cluster': [], 'centroids': [], 'density': [], 'distance': [], 'distance_2_center': []}
    if not type(x)==np.ndarray:
        x = x.reshape(x.shape[0], -1).numpy()
    if len(x.shape) == 3:
        x = x.reshape(x.shape[0], -1)
    x = x.astype(np.float32)

    cluster_id = 0
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        # clus.verbose = True
        clus.niter = 20
        clus.nredo = 1
        # clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        if last_clusters is not None:
            cen = (last_clusters['centroids'][cluster_id].cpu().numpy()).astype(np.float32)
            cen2 = faiss.FloatVector()
            faiss.copy_array_to_vector(cen.reshape(-1), cen2)
            clus.centroids = cen2

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu
        cfg.verbose = True
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        clus.train(x, index)
        D, I = index.search(x, k)
        im2cluster = [int(n[0]) for n in I]
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = args.temperature * density / density.mean()  # scale the mean to temperature


        centroids = torch.Tensor(centroids).cuda()
        xx_norm = torch.nn.functional.normalize(torch.tensor(x).cuda(), p=2, dim=1)
        dist = (xx_norm.unsqueeze(-1).repeat((1,1,k))- centroids.t().unsqueeze(0).repeat((x.shape[0],1,1)))**2
        dist = torch.sum(dist, 1)
        dist = torch.nn.functional.softmax(-dist, 1)

        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)
        results['distance'].append(dist)
        results['distance_2_center'].append(D)

        cluster_id += 1

    return results


def prototype_loss_cotrain(out, index, cluster_result=None, args=None, crop_offset=None, crop_eleft=None, crop_right=None, crop_l=None):
    criterion = nn.CrossEntropyLoss().cuda()
    if len(out.shape) == 2:
        out = out.unsqueeze(-1)
    out = out.permute(0, 2, 1)
    if cluster_result is not None:
        proto_labels = []
        proto_logits = []
        for n, (im2cluster, prototypes, density, pro) in enumerate(
                zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'], cluster_result['ma_centroids'])):

            prototypes = torch.unsqueeze(prototypes, 0)
            prototypes = prototypes.repeat(out.shape[0], 1, 1)
            prototypes = prototypes.permute(0, 2, 1)
            prototypes /= density


            try:
                pos_proto_id = im2cluster[index]
                retain_index = torch.where(pos_proto_id >= 0)
                pos_proto_id = pos_proto_id[retain_index]
                out2 = out[retain_index ]
                prototypes2 = prototypes[retain_index]
            except:
                import pdb

            logits_proto_instance = torch.matmul(out2, prototypes2).squeeze(1)
            proto_loss_instance = criterion(logits_proto_instance, pos_proto_id)

            loss_proto = proto_loss_instance
            for cl in range(pro.shape[0]):
                if (pos_proto_id == cl).sum() > 0:
                    pro[cl, :] = args.ma_gamma * pro[cl, :] + (1-args.ma_gamma) * out2.detach()[(pos_proto_id == cl), ...].mean(0).squeeze(0)
                else:
                    pro[cl, :] = pro[cl, :]
            cluster_result['ma_centroids'][n] = pro


        return loss_proto, cluster_result['ma_centroids']
    else:
        return  None, None


def load_config(dataset, args):
    if dataset == 'Epi':
        from configs.Epi_hyper_param import Param as config
        configs = config()
    elif dataset == 'HAR':
        from configs.HAR_hyper_param import Param as config
        configs = config()
    elif dataset == 'SleepEDF':
        from configs.EDF_hyper_param import Param as config
        configs = config()
    elif dataset == 'Waveform':
        from configs.Waveform_hyper_param import Param as config
        configs = config()
    for arg in configs.__dict__.keys():
        exec(f'args.{arg}=configs.__dict__[arg]')
    return args