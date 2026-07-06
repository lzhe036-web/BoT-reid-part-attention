# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking


def camera_mean_debias(features, camids):
    if not torch.is_tensor(camids):
        camids = torch.as_tensor(camids, device=features.device)
    else:
        camids = camids.to(features.device)

    debiased_features = features.clone()
    for camid in torch.unique(camids):
        mask = camids == camid
        debiased_features[mask] = features[mask] - features[mask].mean(dim=0, keepdim=True)

    return F.normalize(debiased_features, p=2, dim=1)


def _log_camera_mean_debias(logger, qf, gf, camids):
    camids_tensor = torch.as_tensor(camids)
    camera_ids, camera_counts = torch.unique(camids_tensor.cpu(), sorted=True, return_counts=True)
    camera_stats = [
        "{}:{}".format(int(camid), int(count))
        for camid, count in zip(camera_ids, camera_counts)
    ]
    logger.info("Camera mean debias enabled")
    logger.info("Query feature shape: {}".format(tuple(qf.shape)))
    logger.info("Gallery feature shape: {}".format(tuple(gf.shape)))
    logger.info("Camera count: {}".format(len(camera_ids)))
    logger.info("Camera sample counts: {}".format(", ".join(camera_stats)))


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', camera_mean_debias_enabled=False):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.camera_mean_debias_enabled = camera_mean_debias_enabled

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        if self.camera_mean_debias_enabled:
            logger = logging.getLogger("reid_baseline.inference")
            q_camids = torch.as_tensor(self.camids[:self.num_query])
            g_camids = torch.as_tensor(self.camids[self.num_query:])
            all_feats = torch.cat([qf, gf], dim=0)
            all_camids = torch.cat([q_camids, g_camids], dim=0)
            _log_camera_mean_debias(logger, qf, gf, all_camids)
            all_feats = camera_mean_debias(all_feats, all_camids)
            num_query = qf.size(0)
            qf = all_feats[:num_query]
            gf = all_feats[num_query:]
        elif self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = F.normalize(feats, dim=1, p=2)
            qf = feats[:self.num_query]
            gf = feats[self.num_query:]
        # query
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', camera_mean_debias_enabled=False):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.camera_mean_debias_enabled = camera_mean_debias_enabled

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        if self.camera_mean_debias_enabled:
            logger = logging.getLogger("reid_baseline.inference")
            q_camids = torch.as_tensor(self.camids[:self.num_query])
            g_camids = torch.as_tensor(self.camids[self.num_query:])
            all_feats = torch.cat([qf, gf], dim=0)
            all_camids = torch.cat([q_camids, g_camids], dim=0)
            _log_camera_mean_debias(logger, qf, gf, all_camids)
            all_feats = camera_mean_debias(all_feats, all_camids)
            num_query = qf.size(0)
            qf = all_feats[:num_query]
            gf = all_feats[num_query:]
        elif self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = F.normalize(feats, dim=1, p=2)
            qf = feats[:self.num_query]
            gf = feats[self.num_query:]

        # query
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP
