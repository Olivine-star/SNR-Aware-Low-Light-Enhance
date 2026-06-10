import argparse
import logging
import os
import os.path as osp

import cv2
import torch
import torch.nn.functional as F

import options.options as option
import utils.util as util
from data import create_dataloader, create_dataset
from models import create_model


def pad_to_multiple(tensor, multiple=16):
    h, w = tensor.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect'), h, w


def cleanup_model_tensors(model):
    for attr in ('var_L', 'nf', 'real_H', 'fake_H'):
        if hasattr(model, attr):
            delattr(model, attr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to SDE test option YAML file.')
    parser.add_argument('--pretrain_model', type=str, default=None,
                        help='Optional checkpoint path to override path.pretrain_model_G.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Optional root directory for saved SDE inference images.')
    parser.add_argument('--no_save_input', action='store_true',
                        help='Do not save input low-light frames.')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)
    if args.pretrain_model:
        opt['path']['pretrain_model_G'] = args.pretrain_model
    opt = option.dict_to_nonedict(opt)

    save_root = args.save_dir if args.save_dir else osp.join(opt['path']['results_root'], 'images')
    output_folder = osp.join(save_root, 'output')
    input_folder = osp.join(save_root, 'input')
    gt_folder = osp.join(save_root, 'GT')
    util.mkdirs([save_root, output_folder, input_folder, gt_folder])
    util.mkdir(opt['path']['log'])

    util.setup_logger('base', opt['path']['log'], 'test_sde', level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    logger.info('Saving SDE enhanced images to {:s}'.format(output_folder))

    model = create_model(opt)

    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        pbar = util.ProgressBar(len(val_loader))

        psnr_sum = 0.0
        psnr_count = 0

        for val_data in val_loader:
            lq_path = val_data['LQ_path'][0]
            folder = val_data['folder'][0]
            img_name = osp.splitext(osp.basename(lq_path))[0]

            padded_data = dict(val_data)
            padded_lq, orig_h, orig_w = pad_to_multiple(val_data['LQs'], multiple=16)
            padded_nf, _, _ = pad_to_multiple(val_data['nf'], multiple=16)
            padded_data['LQs'] = padded_lq
            padded_data['nf'] = padded_nf

            model.feed_data(padded_data, need_GT=False)
            model.test()
            rlt_tensor = model.fake_H.detach()[0, :, :orig_h, :orig_w].float().cpu()
            rlt_img = util.tensor2img(rlt_tensor)

            scene_output_folder = osp.join(output_folder, folder)
            util.mkdir(scene_output_folder)
            cv2.imwrite(osp.join(scene_output_folder, img_name + '.png'), rlt_img)

            if not args.no_save_input:
                scene_input_folder = osp.join(input_folder, folder)
                util.mkdir(scene_input_folder)
                input_img = util.tensor2img(val_data['LQs'][0])
                cv2.imwrite(osp.join(scene_input_folder, img_name + '.png'), input_img)

            if 'GT' in val_data:
                scene_gt_folder = osp.join(gt_folder, folder)
                util.mkdir(scene_gt_folder)
                gt_img = util.tensor2img(val_data['GT'][0])
                cv2.imwrite(osp.join(scene_gt_folder, img_name + '.png'), gt_img)
                psnr_sum += util.calculate_psnr(rlt_img, gt_img)
                psnr_count += 1

            cleanup_model_tensors(model)
            pbar.update('Test {} - {}'.format(folder, img_name))

        if psnr_count > 0:
            logger.info('# Validation # {} PSNR: {:.4e} over {:d} frames'.format(
                phase, psnr_sum / psnr_count, psnr_count))
        else:
            logger.info('# Inference # {} finished over {:d} frames'.format(phase, len(val_loader)))


if __name__ == '__main__':
    main()
