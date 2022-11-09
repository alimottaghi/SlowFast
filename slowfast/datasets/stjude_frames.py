#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import json
import random
import numpy as np
from itertools import chain
from iopath.common.file_io import g_pathmgr
import torch
import torch.utils.data
from torchvision import transforms

import slowfast.utils.logging as logging

from . import decoder as decoder
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
        Construct the Stjude video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `lab`, `unl`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            aug (boolian): whether to enable augmentation (if aug=False, it overwrites cfg.AUG.ENABLE)
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

        logger.info("Constructing Stjude {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode in ["train", "lab", "unl"] and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True


    def _filter_cases(self, annotations, key, values):

        return sorted([case for case in annotations if annotations[case][key] in values])

    def _split_data(self, cases:set, val_ratio:float=0.2, seed:int=0):

        random.seed(seed)
        all_cases = sorted(list(cases))
        num_cases = len(all_cases)
        num_val = int(val_ratio*num_cases)
        val_cases = random.sample(all_cases, k=num_val)
        train_cases = sorted(list(set(all_cases).difference(val_cases)))

        return train_cases, val_cases


    def _make_dataset_train(self, annotations:dict, data_dir:str, cases:set=None):

        if cases is None:
            cases = sorted(annotations.keys())
        else:
            cases = sorted(list(cases))

        dataset = []
        for case in cases:
            domain = annotations[case]['domain']
            duration = annotations[case]['duration']
            case_dir = os.path.join(data_dir, case)
            num_frames = len(os.listdir(case_dir))
            fps =  num_frames / duration
            actions = annotations[case]['actions']
            for action in actions:
                label = action[0]
                st_frame = int(action[1]*fps)
                en_frame = int(action[2]*fps)
                data_point = (case_dir, st_frame, en_frame, label, num_frames, domain)
                if en_frame - st_frame > 16: # discard activities with less than 16 frames
                    dataset.append(data_point)

        return dataset

    def _make_dataset_test(self, annotations, data_dir, cases=None):

        if cases is None:
            cases = sorted(annotations.keys())
        else:
            cases = sorted(list(cases))

        dataset = []
        for case in cases:
            domain = annotations[case]['domain']
            duration = annotations[case]['duration']
            case_dir = os.path.join(data_dir, case)
            video_frames = len(os.listdir(case_dir))
            fps =  video_frames / duration
            actions = annotations[case]['actions']
            labels = np.zeros((video_frames, self.cfg.MODEL.NUM_CLASSES), np.float32)
            for action in actions:
                st = max(int(action[1]*fps), 0)
                en = min(int(action[2]*fps), video_frames-1)
                labels[st:en+1, action[0]] = 1
            for st_frame in range(0, video_frames, self.cfg.DATA.NUM_FRAMES):
                en_frame = min(st_frame + self.cfg.DATA.NUM_FRAMES, video_frames)
                label = np.argmax(np.sum(labels[st_frame:en_frame], axis=0))
                dataset.append((case_dir, st_frame, en_frame, label, video_frames, domain))

        return dataset

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = self.cfg.STJUDE.PATH_TO_ANNOT_JSON
        assert g_pathmgr.exists(path_to_file), "{} json file not found".format(
            path_to_file
        )
        with open(path_to_file) as f:
            annotations = json.load(f)
        
        path_to_cases_dir = self.cfg.STJUDE.PATH_TO_FRAME_DIR
        assert g_pathmgr.exists(path_to_cases_dir), "{} dir not found".format(
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
            split = train_cases
            domain = self.cfg.STJUDE.TRAIN_SPLIT
        elif self.mode=='lab':
            split = lab_cases
            domain = self.cfg.STJUDE.TRAIN_SPLIT
        elif self.mode=='unl':
            split = unl_cases
            domain = self.cfg.STJUDE.TRAIN_SPLIT
        elif self.mode in ['val', 'test']:
            split = val_cases
            domain = self.cfg.STJUDE.VAL_SPLIT
        elif self.mode=='extract':
            split = annotations.keys()
            domain = 'all'
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        if self.mode in ['train', 'lab', 'unl', 'val']:
            dataset = self._make_dataset_train(
                annotations, 
                path_to_cases_dir, 
                split
            )
        elif self.mode in ['test', 'extract']:
            dataset = self._make_dataset_test(
                annotations, 
                path_to_cases_dir, 
                split
            )
            
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
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing Stjude dataloader (size: {}) from {}".format(
                len(self._dataset), path_to_cases_dir
            )
        )
        logger.info(
            "Using split {} in domain {} with size {}.".format(
                self.mode, domain, len(split),
            )
        )
        if self.cfg.STJUDE.LOG_INFO:
            logger.info(
                "Cases in this split: \n {}".format(
                    split
                )
            )
        if self.cfg.ADAMATCH.ENABLE:
            logger.info("AdaMatch enabled")


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
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):

            case_dir, st_frame, en_frame, label, total_num_frames, domain = self._dataset[index]

            if self.mode in ["train", "lab", "unl", "val"]:
                num_frames = self.cfg.DATA.NUM_FRAMES
                margin = 8
                if self.cfg.ADAMATCH.ENABLE:
                    num_frames = num_frames * self.cfg.ADAMATCH.ALPHA
                    margin = margin * self.cfg.ADAMATCH.ALPHA
                if st_frame + num_frames * sampling_rate > en_frame:
                    st_frame = max(st_frame - margin, 0) # Add 8 frame to the beginning of very short clips
                    en_frame = min(en_frame + margin, total_num_frames) # Add 8 frame to the end of very short clips
                    # if self.cfg.STJUDE.LOG_INFO and en_frame - st_frame < num_frames * sampling_rate:
                    #     logger.warning(
                    #         "Video idx {} class {} with {} frames is too short.".format(
                    #             index, label, en_frame-st_frame
                    #         )
                    # )
                start_index = random.randint(
                    st_frame, 
                    max(en_frame - num_frames*sampling_rate, st_frame)
                    )
                frames_seq = [max(min(start_index + i*sampling_rate, en_frame - 1), st_frame) for i in range(num_frames)]

            # TODO: temporal_sample_index for test case
            elif self.mode in ["test", "extract"]:
                num_frames = self.cfg.DATA.NUM_FRAMES
                frames_seq = [max(min(st_frame + i, en_frame - 1), st_frame) for i in range(num_frames)]

            frame_paths = [os.path.join(case_dir, str(int(frame)).zfill(6) + '.jpg') for frame in frames_seq]
            frames = None
            try:
                frames = utils.retry_load_images(frame_paths, 1)
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        case_dir, e
                    )
                )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to load video idx {} from {}; trial {}".format(
                        index, case_dir, i_try
                    )
                )
                if self.mode not in ["test", "extract"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._dataset) - 1)
                continue

            if self.cfg.ADAMATCH.ENABLE and self.mode in ["train", "lab", "unl"]:
                slow_start = max(int(len(frames)/2) - int(self.cfg.DATA.NUM_FRAMES/2), 0)
                strong_frames = self._aug_frame(
                    frames,
                    spatial_sample_index,
                    min_scale,
                    max_scale,
                    crop_size,
                )
                if self.cfg.ADAMATCH.ALPHA > 1 and random.randint(0 ,1):
                    strong_fast_frames = torch.index_select(
                        strong_frames,
                        1,
                        torch.linspace(
                            0, strong_frames.shape[1] - 1, strong_frames.shape[1] // self.cfg.ADAMATCH.ALPHA
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
                if self.cfg.ADAMATCH.ALPHA > 1 and random.randint(0 ,1):
                    weak_fast_frames = torch.index_select(
                        weak_frames,
                        1,
                        torch.linspace(
                            0, weak_frames.shape[1] - 1, weak_frames.shape[1] // self.cfg.ADAMATCH.ALPHA
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
            
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

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