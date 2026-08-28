import json
import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def _greedy_match_centers(gt_centers, gt_labels, pred_centers, pred_labels, max_dist):
    """
    Greedily match each predicted box to a ground-truth box of the same class
    whose BEV center lies within ``max_dist`` meters.

    Args:
        gt_centers: (N_gt, 2) xy in the lidar/ego frame
        gt_labels: (N_gt,)
        pred_centers: (N_pred, 2)
        pred_labels: (N_pred,)

    Returns:
        list of (gt_idx, pred_idx) pairs
    """
    n_gt, n_pred = gt_centers.shape[0], pred_centers.shape[0]
    if n_gt == 0 or n_pred == 0:
        return []

    dist = np.linalg.norm(gt_centers[:, None, :] - pred_centers[None, :, :], axis=2)
    dist[~(gt_labels[:, None] == pred_labels[None, :])] = np.inf
    dist[dist > max_dist] = np.inf

    pairs = []
    while np.isfinite(dist).any():
        gi, pi = np.unravel_index(np.argmin(dist), dist.shape)
        if not np.isfinite(dist[gi, pi]):
            break
        pairs.append((int(gi), int(pi)))
        dist[gi, :] = np.inf
        dist[:, pi] = np.inf
    return pairs


def extract_perception_errors(batch_dict, pred_dicts, match_dist_thresh=2.0):
    """
    Compare BEVFusion predictions against ground-truth annotations and extract
    distance-dependent perception errors.

    Box layout (nuScenes):
        gt_boxes   : [x, y, z, dx, dy, dz, heading, vx, vy, class]  (10)
        pred_boxes : [x, y, z, dx, dy, dz, heading, vx, vy]          (9)

    Returns:
        list of dicts, one per matched object:
            {'d': ego-to-target distance, 'dx': longitudinal position error,
             'dy': lateral position error, 'dvx': longitudinal velocity error,
             'dvy': lateral velocity error}
    """
    gt_boxes = batch_dict['gt_boxes']  # (B, max_gt, 10), zero-padded
    batch_size = batch_dict['batch_size']
    records = []

    for idx in range(batch_size):
        cur_gt = gt_boxes[idx]
        valid = cur_gt.abs().sum(dim=1) > 0
        cur_gt = cur_gt[valid]
        if cur_gt.shape[0] == 0:
            continue

        pred = pred_dicts[idx]
        pred_boxes = pred['pred_boxes'].detach().cpu().numpy()
        if pred_boxes.shape[0] == 0:
            continue

        pred_labels = pred['pred_labels'].detach().cpu().numpy().astype(np.int64)
        gt_np = cur_gt.detach().cpu().numpy()
        gt_labels = gt_np[:, 9].astype(np.int64)

        pairs = _greedy_match_centers(
            gt_np[:, 0:2], gt_labels, pred_boxes[:, 0:2], pred_labels, match_dist_thresh
        )

        for gi, pi in pairs:
            records.append({
                'd': float(np.hypot(gt_np[gi, 0], gt_np[gi, 1])),
                'dx': float(pred_boxes[pi, 0] - gt_np[gi, 0]),
                'dy': float(pred_boxes[pi, 1] - gt_np[gi, 1]),
                'dvx': float(pred_boxes[pi, 7] - gt_np[gi, 7]),
                'dvy': float(pred_boxes[pi, 8] - gt_np[gi, 8]),
            })

    return records


def fit_distance_dependent_gaussian(records, dist_bin_width=5.0, min_samples_per_bin=5):
    """
    Fit the perception error of each state variable as a distance-dependent
    Gaussian: mean mu(d) and std sigma(d) are both approximated by quadratic
    polynomials in the ego-to-target distance d.

    Returns:
        dict keyed by error variable ('dx', 'dy', 'dvx', 'dvy'):
            {'mu': {'a0','a1','a2'}, 'sigma': {'b0','b1','b2'}, 'bin_stats': {...}}
        or None when there are too few samples to fit.
    """
    if len(records) < min_samples_per_bin * 3:
        return None

    d = np.array([r['d'] for r in records], dtype=np.float64)
    variables = ['dx', 'dy', 'dvx', 'dvy']
    data = {v: np.array([r[v] for r in records], dtype=np.float64) for v in variables}

    bin_idx = np.floor(d / dist_bin_width).astype(np.int64)
    results = {}

    for v in variables:
        x = data[v]
        bins = {}
        for bi in np.unique(bin_idx):
            mask = bin_idx == bi
            cnt = int(mask.sum())
            if cnt < min_samples_per_bin:
                continue
            d_center = float((bi + 0.5) * dist_bin_width)
            bins[d_center] = (cnt, float(x[mask].mean()), float(x[mask].std(ddof=1)))

        if len(bins) < 3:
            continue

        d_centers = np.array(sorted(bins.keys()), dtype=np.float64)
        counts = np.array([bins[k][0] for k in d_centers], dtype=np.float64)
        means = np.array([bins[k][1] for k in d_centers], dtype=np.float64)
        stds = np.array([bins[k][2] for k in d_centers], dtype=np.float64)

        # np.polyfit returns [c2, c1, c0] for a degree-2 polynomial.
        mu_c = np.polyfit(d_centers, means, 2, w=np.sqrt(counts))
        sigma_c = np.polyfit(d_centers, stds, 2, w=np.sqrt(counts))

        results[v] = {
            'mu': {'a0': float(mu_c[2]), 'a1': float(mu_c[1]), 'a2': float(mu_c[0])},
            'sigma': {'b0': float(sigma_c[2]), 'b1': float(sigma_c[1]), 'b2': float(sigma_c[0])},
            'bin_stats': {
                'distance_centers': d_centers.tolist(),
                'counts': counts.tolist(),
                'means': means.tolist(),
                'stds': stds.tolist(),
            },
        }

    return results if results else None


def _save_perception_error(records, fit_results, result_dir, logger):
    save_dir = result_dir / 'perception_error'
    save_dir.mkdir(parents=True, exist_ok=True)

    if fit_results is not None:
        with open(save_dir / 'perception_error_fit.json', 'w') as f:
            json.dump(fit_results, f, indent=2)

        header = '{:>8} {:>12} {:>12} {:>12} | {:>12} {:>12} {:>12}'.format(
            'var', 'a0', 'a1', 'a2', 'b0', 'b1', 'b2')
        lines = [header, '-' * len(header)]
        for v in ['dx', 'dy', 'dvx', 'dvy']:
            if v not in fit_results:
                continue
            mu = fit_results[v]['mu']
            sg = fit_results[v]['sigma']
            lines.append('{:>8} {:>12.6f} {:>12.6f} {:>12.6f} | {:>12.6f} {:>12.6f} {:>12.6f}'.format(
                v, mu['a0'], mu['a1'], mu['a2'], sg['b0'], sg['b1'], sg['b2']))
        logger.info('Perception error fit (mu(d)=a0+a1*d+a2*d^2, sigma(d)=b0+b1*d+b2*d^2):\n'
                    + '\n'.join(lines))
    else:
        logger.warning('Too few matched objects to fit perception error distribution.')

    if records:
        np.savez(save_dir / 'perception_error_records.npz',
                 d=np.array([r['d'] for r in records]),
                 dx=np.array([r['dx'] for r in records]),
                 dy=np.array([r['dy'] for r in records]),
                 dvx=np.array([r['dvx'] for r in records]),
                 dvy=np.array([r['dvy'] for r in records]))
    logger.info('Perception error records saved to %s' % save_dir)


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None,
                   extract_error=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    error_records = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if extract_error:
            error_records.extend(extract_perception_errors(batch_dict, pred_dicts))
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')
        if extract_error:
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, error_records)
            error_records = [r for part in gathered for r in part]

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    if extract_error:
        fit_results = fit_distance_dependent_gaussian(error_records)
        _save_perception_error(error_records, fit_results, result_dir, logger)

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
