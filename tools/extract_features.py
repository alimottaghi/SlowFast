"""Wrapper to extract features from an existing video classification model."""
import os
import numpy as np
import torch
#from sklearn.metrics import classification_report
from collections import OrderedDict

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args


import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
#import timesformer.utils.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
import datetime
#from timesformer.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


class LRUCache(OrderedDict):
    """Limit size, evicting the least recently looked-up key when full.

    Cannot add the same item twice.
    
    """

    def __init__(self, *args, maxsize=128, **kwds):
        self.maxsize = maxsize
        self._evicted = set()
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        # self.move_to_end(key)
        return value

    def evicted(self, key):
        return key in self._evicted

    def setitem_getevicteditem(self, key, value):
        """Add the value, return any evicted items or None if maxsize is not reached.
        
        returns: (key, value) pair that was evicted, or None if maxsize not reached

        """
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldkey = next(iter(self))
            olditem = super().__getitem__(oldkey)
            del self[oldkey]
            self._evicted.add(oldkey)
            return oldkey, olditem
        else:
            return None

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)

        if key in self._evicted:
            logger.error(f'tried to set \'{key}\', but already evicted. Raise maxsize?')
            raise ValueError

        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            self._evicted.add(oldest)
            del self[oldest]


def save_feat(fname, feat, num_crops=1):
    feat['feature'] = np.array([feat['feature'][i] for i in range(feat['start'], feat['end'] + 1)])
    shape = feat['feature'].shape
    feat['feature'] = feat['feature'].reshape(-1, num_crops, shape[1]).mean(1)
    feat['feature'] = feat['feature'][np.newaxis, :, :]
    if du.is_master_proc(num_gpus = du.get_local_size()):
        folder = os.path.dirname(fname)
        if not os.path.exists(folder): os.makedirs(folder, exist_ok=True)
        np.savez(fname, feature=feat['feature'], frame_cnt=feat['frame_cnt'], video_name=feat['video_name'])
    shape = feat['feature'].shape
    frame_cnt = feat['frame_cnt']
    logger.info(f'saved feat_shape {shape} to {fname} with {frame_cnt} frames')

def get_feat_filename(feats_dir, vid, task=None):
    '''
    Get the save location for the features npz. 
    Joins [feat_dir], YYY-MM-DD/VCXXXX-ORXXXX_YYYY-MM-DD_..., Features, NNNN-NNNN-NNNN-NNNN_ORTO.npz
    :param: task must be from ['ORTO','SAR']
    :return: `string` file path
    '''
    path = os.path.join(feats_dir, vid)
    path_split = path.split('/')
    
    if task is None:
        path_split[-1] = path_split[-1].split('.mp4')[0]+'.npz'
    else:
        path_split[-1] = path_split[-1].split('.mp4')[0]+f'_{task}.npz'
    
    # path_split.insert(-1,'Features')
    return '/'.join(path_split)


@torch.no_grad()
def extract_features(extract_loader, model, vid_names, cfg, writer=None):
    """
    Perform feature extraction for every clip in a video along its temporal axis.
    Use the center-cropped image.
    Saves the output features in .npz files.
    """
    #print('At feature: {}'.format(vid_names))
    model.eval()

    # TODO(zhe) Isn't it inefficient that every process has a copy of the feature cache.
    # feats_cache = LRUCache(maxsize=3*cfg.TEST.BATCH_SIZE) # TODO: causing error if smaller than num-videos?
    feats_cache = LRUCache(maxsize=len(vid_names)+100)
    
    # if features output dir is not set, assume a output folder
    if cfg.EXTRACT.PATH_TO_FEAT_OUT_DIR == "":
        logger.info(f'Feature folder EXTRACT.PATH_TO_FEAT_OUT_DIR is not set, defaulting to OUTPUT_DIR.')
        feats_dir = os.path.join(cfg.OUTPUT_DIR, 'features')
    else:
        feats_dir = cfg.EXTRACT.PATH_TO_FEAT_OUT_DIR
    os.makedirs(feats_dir, exist_ok=True)


    start_time = datetime.datetime.now()
    iter_end_time=None
    for cur_iter, (inputs, labels, video_idx, _) in enumerate(extract_loader):
        # inputs size is based on batch size, inputs are the clips
        # video_idx is an int increases by one for each clip loaded. video_idx is a list of these ints that are loaded.
        if iter_end_time is not None:
            diff = iter_end_time - start_time
            logger.info(f'Completed Iteration {(cur_iter+1):03d} of {len(extract_loader):03d} | Elapsed Time {str(diff)} | Remaining {str((len(extract_loader) - cur_iter) * diff / cur_iter)}')
        # logger.info(f'feature cache size {len(feats_cache)}, len of inputs {len(inputs)}, video_idx {len(video_idx)}')
        if cfg.NUM_GPUS == 1:
            if all(os.path.exists(get_feat_filename(feats_dir, vid_names[idx])) for idx in np.array(video_idx.detach().cpu())):
                # This file already saved, continue.
                logger.warn(f'All feature files for this batch already exist. Skipping this batch... Continuing to next batch...')
                continue
        # Transfer the data to the current GPU device.        
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        video_idx = video_idx.cuda()
        # forward pass
        preds, features = model(inputs)

        if len(features.shape) == 1:
            features = features[None, :]

        if cfg.NUM_GPUS > 1:
            preds, features, labels, video_idx = du.all_gather(
                [preds, features, labels, video_idx])

        video_idx_cpu = video_idx.detach().cpu()
        #print(video_idx_cpu)
        features_cpu = np.array(features.detach().cpu())
        # import pdb; pdb.set_trace()
        cur_vid_names = [vid_names[idx] for idx in np.array(video_idx_cpu)]

        # iterate over each clip's feature and check if it should be saved.
        # maintain an OrderedDict mapping vid -> feat, when it gets longer than queue length, pop off the front and save.
        assert len(features_cpu) == len(cur_vid_names), f'{features_cpu.shape} != {len(cur_vid_names)}'
        for f, vidx, vid in zip(features_cpu, np.array(video_idx_cpu), cur_vid_names):
            # logger.info(f"Extracting a clip in: {get_feat_filename(feats_dir, vid, task=cfg.MODEL.NAME)}")
            if feats_cache.evicted(vid):
                # TODO: this is BAD. Figure out a more elegant solution. If this happens with multi-gpu, error can go unnoticed.
                logger.warning(f'already evicted {vid}')

            evicted_item = None
            if vid not in feats_cache:
                feat = {}
                feat['video_name'] = vid
                feat['feature'] = {} # map indices to arrays, until joined into a single numpy array at save time, since video-idx may be out of order
                feat['frame_cnt'] = 0
                evicted_item = feats_cache.setitem_getevicteditem(vid, feat)

            if evicted_item is not None:
                old_vid, old_feat = evicted_item
                fname = get_feat_filename(feats_dir, old_vid)
                logger.info(f'EVICTED {fname}, check if file is saved....')
                #fname = os.path.join('/'.join(vid.split('/')[:-1]), vid.split('/')[-1].split('.mp4')[0]+'_'+cfg.MODEL.NAME+'.npz')
                save_feat(fname, old_feat, num_crops=cfg.TEST.NUM_SPATIAL_CROPS)

            feats_cache[vid]['feature'][vidx] = f
            feats_cache[vid]['frame_cnt'] += cfg.EXTRACT.NUM_FRAMES
            feats_cache[vid]['start'] = min(feats_cache[vid].get('start', vidx), vidx)
            feats_cache[vid]['end'] = max(feats_cache[vid].get('end', vidx), vidx)
        iter_end_time = datetime.datetime.now()


    while len(feats_cache) > 0:
        vid = next(iter(feats_cache))
        feat = feats_cache[vid]
        #fname = os.path.join('/'.join(vid.split('/')[:-1]), vid.split('/')[-1].split('.mp4')[0]+'_'+cfg.MODEL.NAME+'.npz')
        fname = get_feat_filename(feats_dir, vid)
        save_feat(fname, feat)
        del feats_cache[vid]


def extract(cfg):
    """
    Extract features for each clip in the whole dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # set up environment
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # set temporary cfg variables
    cfg.MODEL.EXTRACTING = True

    # Print config.
    logger.info("Extract with config:")
    logger.info(cfg)

    # Build the video model and print model statistics
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    # Load a checkpoint to extract if applicable
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        logger.info("Extracting features with random initialization. Only for debugging.")

    # Create video extraction loader
    extract_loader = loader.construct_loader(cfg, 'extract')
    vid_keys = [t[-1] for t in extract_loader.dataset._dataset]  # the split key for each vid

    # Start extracting features
    logger.info(f'Extracting model for {len(extract_loader)} iterations')
    extract_features(extract_loader, model, vid_keys, cfg, writer=None)

    cfg.MODEL.EXTRACTING = False


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)    
    launch_job(cfg=cfg, init_method=args.init_method, func=extract)
