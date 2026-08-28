"""Extract distance-dependent perception error distribution from nuScenes with BEVFusion.

This is a lean standalone script: it runs BEVFusion inference on the nuScenes val split,
matches predictions to ground truth, extracts the four perception errors (dx, dy, dvx, dvy)
together with the ego-to-target distance d, then fits the distance-dependent Gaussian model

    mu(d)    = a0 + a1*d + a2*d^2
    sigma(d) = b0 + b1*d + b2*d^2

and saves the coefficients (Table III) plus the raw error records.

Usage:
    python tools/extract_perception_error.py \
        --cfg_file tools/cfgs/nuscenes_models/bevfusion.yaml \
        --ckpt /path/to/bevfusion_checkpoint.pth
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm

import _init_path
from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/nuscenes_models/bevfusion.yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='BEVFusion checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=None, help='defaults to cfg batch size')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--match_dist_thresh', type=float, default=2.0,
                        help='BEV center-distance threshold (m) for pred-gt matching')
    parser.add_argument('--dist_bin_width', type=float, default=5.0,
                        help='distance bin width (m) used for mu/sigma fitting')
    parser.add_argument('--min_samples_per_bin', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='output dir; defaults to output/<exp>/<tag>/perception_error')
    parser.add_argument('--records_npz', type=str, default=None,
                        help='if given, skip inference and re-fit from an existing perception_error_records.npz')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='save mu/sigma vs distance plot')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='override config keys, e.g. --set DATA_CONFIG.INFO_PATH.test ...')
    return parser.parse_args()


def save_plot(records, fit_results, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    variables = ['dx', 'dy', 'dvx', 'dvy']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()

    d_all = np.array([r['d'] for r in records])
    d_grid = np.linspace(max(d_all.min(), 0.0), d_all.max(), 200)

    for ax, v in zip(axes, variables):
        if fit_results is None or v not in fit_results:
            ax.set_title(v)
            continue
        bs = fit_results[v]['bin_stats']
        mu = fit_results[v]['mu']
        sg = fit_results[v]['sigma']
        dc = np.array(bs['distance_centers'])
        means = np.array(bs['means'])
        stds = np.array(bs['stds'])

        mu_fit = mu['a0'] + mu['a1'] * d_grid + mu['a2'] * d_grid ** 2
        sg_fit = sg['b0'] + sg['b1'] * d_grid + sg['b2'] * d_grid ** 2

        ax.plot(dc, means, 'o', color='tab:blue', label=r'$\mu(d)$ (binned)')
        ax.plot(d_grid, mu_fit, '-', color='tab:blue', label=r'$\mu(d)$ fit')
        ax.plot(dc, stds, 's', color='tab:red', label=r'$\sigma(d)$ (binned)')
        ax.plot(d_grid, sg_fit, '-', color='tab:red', label=r'$\sigma(d)$ fit')
        ax.set_xlabel('distance d (m)')
        ax.set_ylabel('error')
        ax.set_title(v)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = save_dir / 'perception_error_fit.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(Path(args.cfg_file).parts[1:-1])  # strip 'tools/cfgs' prefix
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    np.random.seed(1024)

    if args.save_dir is None:
        save_dir = Path(cfg.ROOT_DIR) / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / 'perception_error'
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_dir / ('log_extract_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file)
    logger.info('cfg_file: %s' % args.cfg_file)

    if args.records_npz is not None:
        raw = np.load(args.records_npz)
        records = [
            {'d': float(raw['d'][i]), 'dx': float(raw['dx'][i]), 'dy': float(raw['dy'][i]),
             'dvx': float(raw['dvx'][i]), 'dvy': float(raw['dvy'][i])}
            for i in range(len(raw['d']))
        ]
        logger.info('Loaded %d records from %s (skip inference)' % (len(records), args.records_npz))
    else:
        logger.info('ckpt: %s' % args.ckpt)
        if args.batch_size is None:
            args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

        test_set, test_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=False,
        )

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        logger.info('Start inference + perception error extraction ...')
        records = []
        progress_bar = tqdm.tqdm(total=len(test_loader), desc='extract', dynamic_ncols=True)
        with torch.no_grad():
            for batch_dict in test_loader:
                load_data_to_gpu(batch_dict)
                pred_dicts, _ = model(batch_dict)
                records.extend(eval_utils.extract_perception_errors(
                    batch_dict, pred_dicts, match_dist_thresh=args.match_dist_thresh
                ))
                progress_bar.update()
        progress_bar.close()
        logger.info('Matched objects: %d' % len(records))

    fit_results = eval_utils.fit_distance_dependent_gaussian(
        records, dist_bin_width=args.dist_bin_width, min_samples_per_bin=args.min_samples_per_bin
    )
    eval_utils._save_perception_error(records, fit_results, save_dir, logger)

    if args.plot:
        save_plot(records, fit_results, save_dir)

    logger.info('Done. Results saved to %s' % save_dir)


if __name__ == '__main__':
    main()
