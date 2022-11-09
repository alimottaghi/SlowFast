#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import json
import random
import copy
import numpy as np
from itertools import chain

import torch
import torch.utils.data
from torchvision import transforms

import slowfast.utils.logging as logging
import slowfast.utils.distributed as du

from . import decoder
from . import video_container as container
from . import utils as utils
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Stjude(torch.utils.data.Dataset):
    """
    Stjude video loader. Construct the Stjude video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Stjude video loader with a given json file. The format of
        the json file is:
        ```
        [
            case_id: {
                "filename":
                "domain": 
                "duration": 
                "actions":
            }
        ]
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `lab`, `unl`, `val`, `test`, or `extract` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train", 
            "lab", 
            "unl",
            "val",
            "test",
            "extract"
        ], "Split '{}' not supported for Stjude".format(mode)
        self.mode = mode
        self.cfg = cfg
        
        self._video_meta = {}
        self._num_retries = num_retries
        self.aug = False
        self.rand_erase = False
        if self.mode in ["train", "lab", "unl"] and cfg.AUG.ENABLE:
            self.aug = True
            if cfg.AUG.RE_PROB > 0:
                self.rand_erase = True
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "lab", "unl", "val"]:
            self._num_clips = 1
        elif self.mode in ["test", "extract"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )
        self.data_dir = cfg.DATA.PATH_TO_DATA_DIR
        self.annot_file = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.STJUDE.ANNOT_FILE)
        self.num_frames = cfg.EXTRACT.NUM_FRAMES if mode == 'extract' else cfg.DATA.NUM_FRAMES
        self.sampling_rate = cfg.EXTRACT.SAMPLING_RATE if mode == 'extract' else cfg.DATA.SAMPLING_RATE
        self.loss_func = cfg.MODEL.LOSS_FUNC
        self.num_classes = cfg.MODEL.NUM_CLASSES
        
        logger.info("Constructing Stjude {}...".format(mode))
        self._construct_loader()

    def _filter_cases(self, annotations, key, values):
        """
        Filters the cases based on given key and values in the annotation dictionary.
        """
        return sorted([case for case in annotations if annotations[case][key] in values])

    def _split_data(self, cases:set, val_ratio:float=0.2, seed:int=0):
        """
        Splits a set of cases to train and val.
        """
        random.seed(seed)
        all_cases = sorted(list(cases))
        num_cases = len(all_cases)
        num_val = int(val_ratio*num_cases)
        val_cases = random.sample(all_cases, k=num_val)
        train_cases = sorted(list(set(all_cases).difference(val_cases)))

        return train_cases, val_cases

    def _make_dataset(self, annotations:dict, cases:set=None):
        """
        Makes the datasets from the annotations and the cases in the split.
        """
        if cases is None:
            cases = sorted(annotations.keys())
        else:
            cases = sorted(list(cases))

        dataset = []
        for case in cases:
            # Get filename from field in json if specified, otherwise use keys of split file
            if "filename" in annotations[case].keys():
                vid_path = os.path.join(self.data_dir, annotations[case]['filename'])
            else:
                vid_path = os.path.join(self.data_dir, case)
            if not os.path.exists(vid_path):
                logger.warning(f"not found: {vid_path}. If this happens frequently, it may be a sign that "
                               f"DATA.PATH_TO_DATA_DIR is incorrect or not specified.")
                continue

            # Load video
            video_container = None
            try:
                video_container = container.get_video_container(
                    vid_path,
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        vid_path, e
                    )
                )
                continue
            fps = float(video_container.streams.video[0].average_rate)
            video_length = video_container.streams.video[0].frames 
            video_container.close()
            
            # Skip if video is shorter than single clip
            if video_length < self.num_frames * self.sampling_rate:
                logger.warning(f'Excluding {case} from dataset: {vid_path} contains just {video_length} frames, but clip length is {self.num_frames * self.sampling_rate}.')
                continue

            # Skip if no activities found in annotation
            if len(annotations[case]['actions']) == 0:
                logger.warning(f'Excluding {case} from dataset: No activities in annotation file.')
                continue

            actions = annotations[case]['actions']
            labels_matrix = np.zeros((video_length, self.num_classes), np.uint8)
            case_st, case_en = 0, video_length
            for action in actions:
                label = action[0]
                st_frame = max(int(action[1] * fps), 0)
                en_frame = min(int(action[2] * fps), video_length - 1)
                labels_matrix[st_frame:en_frame + 1, label] = 1
                case_st, case_en = min(case_st, st_frame), max(case_en, en_frame)

            if self.mode in ['train', 'lab', 'unl', 'val']:
                for action in actions:
                    label = action[0]
                    st_frame = max(int(action[1] * fps), 0)
                    en_frame = min(int(action[2] * fps), video_length - 1)
                    if en_frame - st_frame + 1 <= self.num_frames * self.sampling_rate:
                        continue
                    dataset.append((vid_path, st_frame, en_frame, label, labels_matrix, video_length, case))

            elif self.mode in ['test', 'extract']:
                if not self.cfg.EXTRACT.EXTRACT_RELEVANT_FRAMES_ONLY:
                    case_st, case_en = 0, video_length
                for st_frame in range(case_st, case_en, self.num_frames * self.sampling_rate):
                    en_frame = min(st_frame + self.num_frames * self.sampling_rate, video_length - 1)
                    label = np.argmax(np.sum(labels_matrix[st_frame:en_frame], axis=0))
                    dataset.append((vid_path, st_frame, en_frame, label, labels_matrix, video_length, case))

        return dataset

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = self.annot_file
        assert os.path.exists(path_to_file), "{} json file not found".format(
            path_to_file
        )
        with open(path_to_file) as f:
            annotations = json.load(f)
        
        path_to_cases_dir = self.data_dir
        assert os.path.exists(path_to_cases_dir), "{} dir not found".format(
            path_to_cases_dir
        )

        train_split, val_split = self.cfg.STJUDE.TRAIN_SPLIT, self.cfg.STJUDE.VAL_SPLIT
        key = self.cfg.STJUDE.DATA_SPLIT_TYPE
        if self.cfg.STJUDE.RANDOM_DATA_SPLIT:  # Random split of data with VAL_RATIO
            if key=='video':
                all_cases = sorted(annotations.keys())
                train_cases, val_cases = self._split_data(
                        all_cases, 
                        val_ratio=self.cfg.STJUDE.VAL_RATIO,  
                        seed=self.cfg.RNG_SEED
                    )
            else:
                all_values = sorted(set([annotations[case][key] for case in annotations]))
                train_values, val_values = self._split_data(
                    all_values, 
                    val_ratio=self.cfg.STJUDE.VAL_RATIO, 
                    seed=self.cfg.RNG_SEED
                )
                train_cases = self._filter_cases(annotations, key, train_values)
                val_cases = self._filter_cases(annotations, key, val_values)
        else:
            if key=='video':
                train_cases = sorted(train_split)
                val_cases = sorted(val_split)
            else:
                train_values = sorted(train_split)
                val_values = sorted(val_split)
                train_cases = self._filter_cases(annotations, key, train_values)
                val_cases = self._filter_cases(annotations, key, val_values)

        key = self.cfg.STJUDE.LAB_SPLIT_TYPE
        lab_split, unl_split = self.cfg.STJUDE.LAB_SPLIT, self.cfg.STJUDE.UNL_SPLIT
        if self.cfg.STJUDE.RANDOM_LAB_SPLIT:  # Random split of train data with LAB_RATIO
            if key=='video':
                all_cases = train_cases
                lab_cases, unl_cases = self._split_data(
                        all_cases, 
                        val_ratio=1-self.cfg.STJUDE.LAB_RATIO,  
                        seed=self.cfg.RNG_SEED
                    )
            else:
                all_values = sorted(set([annotations[case][key] for case in train_cases]))
                lab_values, unl_values = self._split_data(
                    all_values, 
                    val_ratio=1-self.cfg.STJUDE.LAB_RATIO, 
                    seed=self.cfg.RNG_SEED
                )
                lab_cases = self._filter_cases(annotations, key, lab_values)
                unl_cases = self._filter_cases(annotations, key, unl_values)
        else:
            if key=='video':
                lab_cases = sorted(lab_split)
                unl_cases = sorted(unl_split)
            else:
                lab_values = sorted(lab_split)
                unl_values = sorted(unl_split)
                lab_cases = self._filter_cases(annotations, key, lab_values)
                unl_cases = self._filter_cases(annotations, key, unl_values)

        if self.mode=='train':
            split_cases = train_cases
            split_name = self.cfg.STJUDE.TRAIN_SPLIT
        elif self.mode=='lab':
            split_cases = lab_cases
            split_name = self.cfg.STJUDE.TRAIN_SPLIT
        elif self.mode=='unl':
            split_cases = unl_cases
            split_name = self.cfg.STJUDE.TRAIN_SPLIT
        elif self.mode in ['val', 'test']:
            split_cases = val_cases
            split_name = self.cfg.STJUDE.VAL_SPLIT
        elif self.mode=='extract':
            split_cases = annotations.keys()
            split_name = 'all'
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        dataset = self._make_dataset(annotations, split_cases)
            
        self._dataset = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in dataset]
            )
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [range(self._num_clips) for _ in range(len(dataset))]
            )
        )
        self._video_meta = list(
            chain.from_iterable(
                [[None] * self._num_clips for _ in range(len(dataset))]
            )
        )
        
        assert (
            len(self._dataset) > 0
        ), "Failed to load Stjude split {} from {}".format(
            split_name, path_to_file
        )
        logger.info(
            "Constructed Stjude dataloader (mode: {}, split: {}, size: {}) from {} on process with rank {}.".format(
                self.mode,
                split_name, 
                len(self._dataset), 
                self.data_dir, 
                du.get_rank()
            )
        )

        if self.cfg.STJUDE.SAVE_SPLIT:
            split2save = {}
            for case in annotations:
                split2save[case] = annotations[case].copy()
                if case in lab_cases:
                    split2save[case]['subset'] = 'training-labeled'
                elif case in unl_cases:
                    split2save[case]['subset'] = 'training-unlabeled'
                elif case in val_cases:
                    split2save[case]['subset'] = 'validation'
            output_file = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.STJUDE.ANNOT_FILE)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split2save, f, ensure_ascii=False, indent=4)
            logger.info(
                "Split saved in {}".format(output_file)
            )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "lab", "unl", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test", "extract"]:
            temporal_sample_index = (
                    self._spatial_temporal_idx[index]
                    // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                        self._spatial_temporal_idx[index]
                        % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                     + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
                )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.sampling_rate,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            vid_path, st_frame, en_frame, label_, labels_matrix, video_length, case = self._dataset[index]
            if self.mode in ['train', 'lab', 'unl', 'val']:
                num_frames = self.num_frames
                margin = self.cfg.STJUDE.MARGIN
                if self.cfg.STJUDE.RETURN_WEAK:
                    num_frames = num_frames * self.cfg.STJUDE.ALPHA
                    margin = margin * self.cfg.STJUDE.ALPHA
                if st_frame + num_frames * sampling_rate > en_frame:
                    st_frame = max(st_frame - margin, 0) # Add MARGIN frame to the beginning of very short clips
                    en_frame = min(en_frame + margin, video_length) # Add MARGIN frame to the end of very short clips
                    if en_frame - st_frame < num_frames * sampling_rate:
                        logger.warning(
                            "Video idx {} class {} with {} frames is too short.".format(
                                index, label_, en_frame-st_frame
                            )
                        )
                        # continue
                start_index = random.randint(
                    st_frame, 
                    max(en_frame + 1- num_frames*sampling_rate, st_frame)
                    )
                frames_seq = [max(min(start_index + i*sampling_rate, en_frame - 1), st_frame) for i in range(num_frames)]

            elif self.mode in ['test', 'extract']:
                num_frames = self.num_frames
                frames_seq = [max(min(st_frame + i*sampling_rate, en_frame - 1), st_frame) for i in range(num_frames)]

            labels_matrix = labels_matrix[frames_seq, :]
            if self.loss_func == 'cross_entropy':
                label = int(label_)
            elif self.loss_func in ['bce', 'bce_logit']:
                label = (np.sum(labels_matrix, axis=0) >= len(frames_seq) // 3) * 1.

            video_container = None
            try:
                video_container = container.get_video_container(
                    vid_path,
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
                video_container.streams.video[0].thread_type = "AUTO"
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        vid_path, e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test", "extract"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            timebase = video_container.streams.video[0].duration / video_container.streams.video[0].frames
            video_frames = None
            if video_container.streams.video:
                video_frames, max_pts = decoder.pyav_decode_stream(
                    video_container,
                    int(frames_seq[0] * timebase),
                    int( frames_seq[-1] * timebase), # Edited by ZH (originally just int(frames_seq[-1]))
                    video_container.streams.video[0],
                    {"video": 0},
                    int( sampling_rate + 1 )
                )
            video_frames = video_frames[:int(num_frames*sampling_rate)]
            # print(f'Num of found frames is {len(video_frames)}. Debug: {frames_seq},, {int(frames_seq[0] * timebase)}, {int( (frames_seq[-1]+1) * timebase)},, {video_container.streams.video[0].duration}, {video_container.streams.video[0].frames},, {timebase}')
            if len(video_frames) < int( num_frames*sampling_rate ):
                # The portion below is causing the training to be too slow
                if i_try == 0: # Catching file streaming error by retrying ...
                    video_container.close()
                    #logger.warn(f'Only found {len(video_frames)} frames, require {num_frames*sampling_rate} frames for clip at index {index}. Could by pyav streaming pipe issue, Retrying...')
                    continue
                else: # if repeated errors, duplicate the last frame until we reach the correct num of frames
                    pass
                    # logger.warn(f'Only found {len(video_frames)} frames, require {num_frames*sampling_rate} frames for clip at index {index}. Duplicating last frame until OK...')

            video_container.close()
            unsampled_frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
            # OLD (removed for speed-up) Duplicate last frame until clip is full.
            while len(unsampled_frames) < int(num_frames*sampling_rate):  unsampled_frames.append( copy.deepcopy(unsampled_frames[-1]) )
            
            # Apply the sampling rate
            frames = [ unsampled_frames[int(sampling_rate * n)] for n in range(num_frames) ]
            # NEW Duplicate last frame until frames is full
            # while len(frames) < num_frames: frames.append( copy.deepcopy(unsampled_frames[-1]) )
            frames = torch.as_tensor(np.stack(frames))

            if self.cfg.STJUDE.RETURN_WEAK and self.mode in ["train", "lab", "unl"]:
                slow_start = max(int(len(frames)/2) - int(self.cfg.DATA.NUM_FRAMES/2), 0)
                strong_frames = self._aug_frame(
                    frames,
                    spatial_sample_index,
                    min_scale,
                    max_scale,
                    crop_size,
                )
                if self.cfg.STJUDE.ALPHA > 1 and random.randint(0 ,1):
                    strong_fast_frames = torch.index_select(
                        strong_frames,
                        1,
                        torch.linspace(
                            0, strong_frames.shape[1] - 1, strong_frames.shape[1] // self.cfg.STJUDE.ALPHA
                        ).long(),
                    )
                    strong_input = strong_fast_frames
                else:
                    strong_slow_frames = torch.index_select(
                        strong_frames,
                        1,
                        torch.linspace(
                            slow_start, slow_start + self.cfg.DATA.NUM_FRAMES - 1, self.cfg.DATA.NUM_FRAMES
                        ).long(),
                    )
                    strong_input = strong_slow_frames

                frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.
                weak_frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )
                if self.cfg.STJUDE.ALPHA > 1 and random.randint(0 ,1):
                    weak_fast_frames = torch.index_select(
                        weak_frames,
                        1,
                        torch.linspace(
                            0, weak_frames.shape[1] - 1, weak_frames.shape[1] // self.cfg.STJUDE.ALPHA
                        ).long(),
                    )
                    weak_input = weak_fast_frames
                else:
                    weak_slow_frames = torch.index_select(
                        weak_frames,
                        1,
                        torch.linspace(
                            slow_start, slow_start + self.cfg.DATA.NUM_FRAMES - 1, self.cfg.DATA.NUM_FRAMES
                        ).long(),
                    )
                    weak_input = weak_slow_frames
                frames = [weak_input, strong_input]
                return frames, label, index, {}
            
            elif self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:
                    frame_list = []
                    label_list = []
                    index_list = []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                        new_frames = utils.pack_pathway_output(
                            self.cfg, new_frames
                        )
                        frame_list.append(new_frames)
                        label_list.append(label)
                        index_list.append(index)
                    return frame_list, label_list, index_list, {}

                else:
                    frames = self._aug_frame(
                        frames,
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    )
                    frames = utils.pack_pathway_output(self.cfg, frames)
                    return frames, label, index, {}

            else:
                frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.
                frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )
                frames = utils.pack_pathway_output(self.cfg, frames)
                return frames, label, index, {}

    def _aug_frame(
            self,
            frames,
            spatial_sample_index,
            min_scale,
            max_scale,
            crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train", "lab", "unl"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train", "lab", "unl"] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train", "lab", "unl"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._dataset)
