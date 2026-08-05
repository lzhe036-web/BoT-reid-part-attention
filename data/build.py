# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms
from utils.reproducibility import (
    data_loader_generator_metadata,
    make_data_loader_generator,
    seed_worker,
)
from utils.experiment_recording import build_dataset_manifest


def collect_dataset_protocol(cfg):
    """Collect split metadata without opening image contents."""
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    batch_size = int(cfg.SOLVER.IMS_PER_BATCH)
    sampler_name = str(cfg.DATALOADER.SAMPLER)
    if sampler_name == 'softmax':
        sampler_length = len(dataset.train)
    else:
        sampler_length = len(RandomIdentitySampler(
            dataset.train,
            batch_size,
            cfg.DATALOADER.NUM_INSTANCE,
            seed=cfg.SEED,
        ))
    train_loader_batches = (sampler_length + batch_size - 1) // batch_size
    configured_name = cfg.DATASETS.NAMES
    if isinstance(configured_name, (tuple, list)):
        configured_name = configured_name[0] if configured_name else ''
    manifest = build_dataset_manifest(
        splits={
            'train': dataset.train,
            'query': dataset.query,
            'gallery': dataset.gallery,
        },
        data_root=cfg.DATASETS.ROOT_DIR,
        dataset_name=configured_name,
        sampler=sampler_name,
        batch_size=batch_size,
        num_instance=cfg.DATALOADER.NUM_INSTANCE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        train_loader_batches=train_loader_batches,
        sampler_base_seed=cfg.SEED,
        data_loader_generators=data_loader_generator_metadata(cfg.SEED),
    )
    return manifest, int(dataset.num_train_pids)


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    train_generator = make_data_loader_generator(cfg.SEED, 'train')
    validation_generator = make_data_loader_generator(cfg.SEED, 'query')
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn, worker_init_fn=seed_worker,
            generator=train_generator
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(
                dataset.train,
                cfg.SOLVER.IMS_PER_BATCH,
                cfg.DATALOADER.NUM_INSTANCE,
                seed=cfg.SEED,
            ),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn, worker_init_fn=seed_worker,
            generator=train_generator
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn, worker_init_fn=seed_worker,
        generator=validation_generator
    )
    return train_loader, val_loader, len(dataset.query), num_classes
