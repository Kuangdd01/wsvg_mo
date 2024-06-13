import os.path
from functools import partial

from torch.utils.data import DataLoader

from ._glove_tokenizer import build_glove_vocab


def build_mat_dataset(args, tokenizer):
    from ._mat_image_features_reader import MatImageFeaturesH5Reader
    from .mat_entity import MatFlickrGroundingDataset

    def dataset(split):
        mat_root = args.mat_root
        det_name = lambda x: os.path.join(mat_root, f"{x}_detection_dict.json")
        idx_name = lambda x: os.path.join(mat_root, f"{x}_imgid2idx.pkl")
        h5_name = lambda x: os.path.join(mat_root, f"{x}_features_compress.hdf5")

        image_features_reader = MatImageFeaturesH5Reader(args, h5_name(split), idx_name(split), det_name(split))
        return MatFlickrGroundingDataset(dataroot=args.dataroot,
                                         split=split,
                                         image_features_reader=image_features_reader,
                                         tokenizer=tokenizer,
                                         )

    if args.debug:
        train_dset = dataset(split='val')
        val_dset = train_dset
        test_dset = dataset(split='test')
    else:
        train_dset = dataset(split='train')
        val_dset = dataset(split='val')
        test_dset = dataset(split='test')

    return train_dset, val_dset, test_dset


def build_dataloader(args, tokenizer):
    train_dset, val_dset, test_dset = build_mat_dataset(args, tokenizer)
    if args.num_workers:
        dataloader = partial(DataLoader, batch_size=args.batch_size, pin_memory=True,
                             num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    else:
        dataloader = partial(DataLoader, batch_size=args.batch_size, pin_memory=True,
                             num_workers=args.num_workers)

    train_dl = dataloader(train_dset, collate_fn=train_dset.collate_fn, shuffle=True, drop_last=True)
    val_dl = dataloader(val_dset, collate_fn=val_dset.collate_fn)
    test_dl = dataloader(test_dset, collate_fn=test_dset.collate_fn)
    return train_dl, val_dl, test_dl
