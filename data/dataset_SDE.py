import os
import os.path as osp
import random

import cv2
import torch
import torch.utils.data as data

import data.util as util


def _image_files(path):
    return [p for p in util.glob_file_list(path) if util.is_image_file(p)]


class VideoSameSizeDataset(data.Dataset):
    """RGB loader for the SDE indoor/outdoor low-light enhancement splits.

    Expected split layout:
        split_root/scene_name/low/*.png
        split_root/scene_name/normal/*.png

    Low and normal frames are paired by sorted frame index because SDE low/normal
    timestamps differ while counts are aligned inside each scene.
    """

    def __init__(self, opt):
        super(VideoSameSizeDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.LQ_root = opt['dataroot_LQ']
        self.GT_root = opt.get('dataroot_GT', None)
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        self.imgs_LQ, self.imgs_GT = {}, {}

        if self.data_type == 'lmdb':
            raise ValueError('SDE loader expects image folders, not LMDB.')
        if not osp.isdir(self.LQ_root):
            raise ValueError('{:s} is not a valid SDE split root.'.format(self.LQ_root))

        self._scan_scenes()
        if self.opt['phase'] == 'train' and not self.data_info['path_GT']:
            raise ValueError('SDE training requires normal-light GT frames.')

    def _resolve_gt_dir(self, scene_lq_dir, scene_name):
        candidates = []
        if self.GT_root is not None:
            candidates.extend([
                osp.join(self.GT_root, scene_name, 'normal'),
                osp.join(self.GT_root, scene_name),
            ])
        candidates.append(osp.join(scene_lq_dir, 'normal'))
        for path in candidates:
            if osp.isdir(path):
                return path
        return None

    def _scan_scenes(self):
        scene_dirs = [p for p in util.glob_file_list(self.LQ_root) if osp.isdir(p)]
        for scene_lq_dir in scene_dirs:
            scene_name = osp.basename(scene_lq_dir)
            low_dir = osp.join(scene_lq_dir, 'low')
            if not osp.isdir(low_dir):
                low_dir = scene_lq_dir

            img_paths_LQ = _image_files(low_dir)
            if not img_paths_LQ:
                continue

            gt_dir = self._resolve_gt_dir(scene_lq_dir, scene_name)
            img_paths_GT = _image_files(gt_dir) if gt_dir is not None else []
            if img_paths_GT and len(img_paths_LQ) != len(img_paths_GT):
                raise ValueError(
                    'Different number of SDE LQ and GT frames in {:s}: {:d} vs {:d}'.format(
                        scene_name, len(img_paths_LQ), len(img_paths_GT)))

            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(img_paths_LQ)
            self.data_info['path_GT'].extend(img_paths_GT if img_paths_GT else [''] * max_idx)
            self.data_info['folder'].extend([scene_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

            border_l = [0] * max_idx
            for i in range(min(self.half_N_frames, max_idx)):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            if self.cache_data:
                self.imgs_LQ[scene_name] = img_paths_LQ
                self.imgs_GT[scene_name] = img_paths_GT

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        if self.cache_data:
            img_LQ_path = self.imgs_LQ[folder][idx]
            img_GT_path = self.imgs_GT[folder][idx] if self.imgs_GT.get(folder) else ''
        else:
            img_LQ_path = self.data_info['path_LQ'][index]
            img_GT_path = self.data_info['path_GT'][index]

        resize_size = self.opt.get('train_size', None)
        img_LQ = util.read_img_seq([img_LQ_path], resize_size)[0]
        has_gt = bool(img_GT_path)
        if has_gt:
            img_GT = util.read_img_seq([img_GT_path], resize_size)[0]

        if self.opt['phase'] == 'train':
            GT_size = self.opt['GT_size']
            _, H, W = img_LQ.shape
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))

            img_LQ = img_LQ[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]

            img_l = [img_LQ, img_GT]
            img_l = util.augment_torch(img_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ, img_GT = img_l[0], img_l[1]

        img_nf = img_LQ.clone().permute(1, 2, 0).numpy() * 255.0
        img_nf = cv2.blur(img_nf, (5, 5))
        img_nf = img_nf * 1.0 / 255.0
        img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)

        data_dict = {
            'LQs': img_LQ,
            'nf': img_nf,
            'folder': folder,
            'idx': '{}/{}'.format(idx, max_idx),
            'border': border,
            'LQ_path': img_LQ_path
        }
        if has_gt:
            data_dict['GT'] = img_GT
            data_dict['GT_path'] = img_GT_path
        return data_dict

    def __len__(self):
        return len(self.data_info['path_LQ'])
