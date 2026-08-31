"""
Visualize a representative "perception-limited" dangerous test scenario.

Loads the HDF5 results produced by sce_gen_gpu.py / sce_gen_parallel.py,
selects the most typical dangerous episode (collision / low TTC with clear
perception underestimation), then renders:

  * a static PNG of the most dangerous moment, and
  * an MP4 (or GIF fallback) animation of the whole episode.

Bird's-eye view shows the 3-lane road (ego drives to the RIGHT, forward
distance on the x-axis, lateral position on the y-axis). Vehicles are drawn as
rectangles; the ego's *noisy* perception of the other vehicles is drawn as
dashed "ghost" boxes linked to their true positions. A second panel plots
true-TTC vs perceived-TTC over time, marking the danger window.

Assumes the fixed (aligned) HDF5 data, where perception_data true columns
(4:8) match trajectories at the same timestep.
"""

import os
import glob
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ===================== config (edit as needed) =====================
DATA_DIR = "/workspace/asw-shared/dlp/training_tasks/aoh6szh/n260827-174817-scegen-ppo"
H5_GLOB = os.path.join(DATA_DIR, "ppo_logs_gpu*/vae-ppo_vehicle_trajectories_*.h5")
OUT_IMAGE = "typical_scenario.png"
OUT_VIDEO = "typical_scenario.mp4"
FPS = 10            # video playback frames per second
VIEW_AHEAD = 30.0   # meters ahead of ego shown in the bird's-eye view
VIEW_BEHIND = 10.0  # meters behind ego
TTC_YLIM = 15.0     # y-axis cap for the TTC panel (values above = "no conflict")

# ===================== constants (must match sce_gen_gpu.py) =====================
LANE_WIDTH = 4.0
LANES = 3
CAR_LENGTH = 5.0
CAR_WIDTH = 2.0
TIME_STEP = 0.2
SAME_LANE_THRESH = 2.2
TTC_LOW = 1.0
TTC_HIGH = 4.0
TTC_CAP = 100.0  # value used for "no conflict on this frame" (matches env default)

NAMES = ['ego', 'adversary', 'bg2', 'bg3', 'bg4']
COLORS = ['#1f77b4', '#d62728', '#7f7f7f', '#7f7f7f', '#7f7f7f']


# ===================== TTC computation =====================
def min_ttc_vectorized(ego, others):
    """Per-timestep minimum TTC between ego and all other vehicles.

    ego:    (T, 4)  [pos, lane_pos, speed, 0]
    others: (T, M, 4)
    Returns (T,) array of min TTC (inf where no conflict).
    """
    ego_pos = ego[:, 0][:, None]
    ego_lane = ego[:, 1][:, None]
    ego_spd = ego[:, 2][:, None]
    o_pos = others[:, :, 0]
    o_lane = others[:, :, 1]
    o_spd = others[:, :, 2]

    dx = o_pos - ego_pos
    rel = ego_spd - o_spd
    lane_ok = np.abs(ego_lane - o_lane) < SAME_LANE_THRESH

    with np.errstate(divide='ignore', invalid='ignore'):
        fwd = (dx - CAR_LENGTH) / rel     # ego catching a slower leader
        rear = (-dx - CAR_LENGTH) / (-rel)  # follower catching ego

    fwd = np.where(lane_ok & (dx > 0) & (rel > 0) & (fwd > 0), fwd, np.inf)
    rear = np.where(lane_ok & (dx < 0) & (rel < 0) & (rear > 0), rear, np.inf)
    return np.minimum(fwd, rear).min(axis=1)


# ===================== data loading =====================
def _load_episode_dict(path, key, g):
    """Build the episode dict expected by draw_frame from one h5 group.

    `g` is an already-open h5 group for `key` (e.g. 'episode_42'). Returns None
    if the trajectory is empty.
    """
    traj = g['trajectories'][:]            # (T, 5, 4) true state
    perc = g['perception_data'][:, :, 0:4]  # (T, 5, 4) ego's noisy view
    if traj.shape[0] == 0:
        return None
    ttc_true = min_ttc_vectorized(traj[:, 0, :], traj[:, 1:, :])
    ttc_perc = min_ttc_vectorized(perc[:, 0, :], perc[:, 1:, :])
    ttc_true = np.where(np.isinf(ttc_true), TTC_CAP, ttc_true)
    ttc_perc = np.where(np.isinf(ttc_perc), TTC_CAP, ttc_perc)
    return dict(
        path=path, key=key, traj=traj, perc=perc,
        ttc_true=ttc_true, ttc_perc=ttc_perc,
        collision=bool(g.attrs.get('collision', False)),
        reward=float(g.attrs.get('episode_reward', 0.0)),
    )


def collect_episodes(files):
    """Return a list of episode dicts with trajectory + perception + TTC data."""
    eps = []
    for path in files:
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                e = _load_episode_dict(path, k, f[k])
                if e is not None:
                    eps.append(e)
    return eps


def load_one_episode(path, episode):
    """Load a single episode by (h5 path, episode id). Returns the episode dict."""
    key = f"episode_{int(episode)}"
    with h5py.File(path, 'r') as f:
        if key not in f:
            raise SystemExit(f"Group {key} not found in {path}; "
                             f"available: {list(f.keys())[:10]} ...")
        e = _load_episode_dict(path, key, f[key])
    if e is None:
        raise SystemExit(f"Empty trajectory for {key} in {path}")
    return e


def select_episode(eps):
    """Pick the most 'typical' dangerous scenario.

    Prefers an actual collision (clearest danger outcome); within that pool it
    favors a lower minimum true TTC plus a larger perception underestimation at
    the danger moment (the ego thinks it is safer than it really is -- the core
    perception-limited effect). Falls back to near-miss episodes if no collision
    exists.
    """
    dangerous = [e for e in eps if e['ttc_true'].min() < TTC_HIGH]
    if not dangerous:
        dangerous = eps

    collisions = [e for e in dangerous if e['collision']]
    pool = collisions if collisions else dangerous

    def score(e):
        i = int(np.argmin(e['ttc_true']))
        mttc = e['ttc_true'][i]
        under = max(0.0, e['ttc_perc'][i] - mttc)  # >0 means underestimated danger
        return (TTC_LOW - mttc) + 0.5 * min(under, 10.0)  # cap extreme misses

    e = max(pool, key=score)
    i = int(np.argmin(e['ttc_true']))
    mttc = e['ttc_true'][i]
    under = e['ttc_perc'][i] - mttc
    print(f"Selected {e['path']} :: {e['key']}")
    print(f"  collision={e['collision']}  min_true_TTC={mttc:.2f}s  "
          f"underestimation_at_danger={under:.2f}s  danger_frame={i}")
    return e, i


# ===================== rendering =====================
def draw_frame(ax_main, ax_ttc, e, idx, collision):
    traj = e['traj']   # (T, 5, 4)  [pos, lane_pos, speed, 0]
    perc = e['perc']   # (T, 5, 4)
    T = traj.shape[0]
    t = idx
    time_s = t * TIME_STEP

    road_right = LANES * LANE_WIDTH
    ego_x = traj[t, 0, 0]   # longitudinal position of ego

    # ---- main bird's-eye panel (ego drives to the right) ----
    ax_main.clear()
    ax_main.set_facecolor('#1a1e24')

    # road surface background across all lanes
    ax_main.axhspan(0, road_right, color='#2c313a', zorder=0)

    # lane markings
    for b in range(LANES + 1):
        y = b * LANE_WIDTH
        is_edge = b in (0, LANES)
        ax_main.axhline(y, color='white' if is_edge else '#aab', lw=2.4 if is_edge else 1.2,
                        ls='-' if is_edge else '--', alpha=0.85 if is_edge else 0.5, zorder=1)

    ax_main.set_xlim(ego_x - VIEW_BEHIND, ego_x + VIEW_AHEAD)
    ax_main.set_ylim(-2.0, road_right + 2.0)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('Longitudinal position (m)')
    ax_main.set_ylabel('Lateral position (m)')

    # vehicles: solid = true, dashed ghost = ego's noisy perception
    for i in range(traj.shape[1]):
        x = traj[t, i, 0]   # longitudinal
        y = traj[t, i, 1]   # lateral
        ax_main.add_patch(Rectangle(
            (x - CAR_LENGTH / 2, y - CAR_WIDTH / 2), CAR_LENGTH, CAR_WIDTH,
            facecolor=COLORS[i], edgecolor='black', lw=1.0, zorder=3, alpha=0.95))
        ax_main.text(x, y, NAMES[i], ha='center', va='center', fontsize=6,
                     color='white', zorder=4)
        if i != 0:  # ego perceives other vehicles with noise
            xp = perc[t, i, 0]
            yp = perc[t, i, 1]
            ax_main.add_patch(Rectangle(
                (xp - CAR_LENGTH / 2, yp - CAR_WIDTH / 2), CAR_LENGTH, CAR_WIDTH,
                fill=False, edgecolor=COLORS[i], lw=1.6, ls='--', alpha=0.85, zorder=2))
            ax_main.plot([x, xp], [y, yp], color=COLORS[i], lw=0.7, alpha=0.5, zorder=2)

    # collision marker (at the final frame of a collision episode)
    if collision and t == T - 1:
        ax_main.scatter([traj[t, 0, 0]], [traj[t, 0, 1]], marker='x', s=260,
                        color='#ffd166', lw=3.5, zorder=6)

    ax_main.legend(handles=[
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4',
               markersize=10, label='ego (true)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728',
               markersize=10, label='adversary (RL)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7f7f7f',
               markersize=10, label='background'),
        Line2D([0], [0], color='#d62728', lw=1.6, ls='--', label='ego perceived (noisy)'),
    ], loc='upper left', fontsize=7, framealpha=0.9)

    # ---- TTC panel ----
    ax_ttc.clear()
    tt = np.arange(T) * TIME_STEP
    ttc_true_disp = np.clip(e['ttc_true'], 0, TTC_YLIM)
    ttc_perc_disp = np.clip(e['ttc_perc'], 0, TTC_YLIM)
    ax_ttc.plot(tt, ttc_true_disp, color='black', lw=2.0, label='true TTC')
    ax_ttc.plot(tt, ttc_perc_disp, color='#d62728', lw=1.5, ls='--', label='perceived TTC')
    ax_ttc.axhline(TTC_LOW, color='red', lw=1.0, ls=':')
    ax_ttc.axhline(TTC_HIGH, color='orange', lw=1.0, ls=':')
    ax_ttc.axvline(time_s, color='gray', lw=1.0, alpha=0.6)
    ax_ttc.fill_between(tt, 0, TTC_LOW, color='red', alpha=0.10)
    ax_ttc.fill_between(tt, TTC_LOW, TTC_HIGH, color='orange', alpha=0.08)
    ax_ttc.set_ylim(0, TTC_YLIM)
    ax_ttc.set_xlim(0, (T - 1) * TIME_STEP)
    ax_ttc.set_xlabel('Time (s)')
    ax_ttc.set_ylabel('TTC (s)')
    ax_ttc.legend(loc='upper right', fontsize=8)
    ax_ttc.grid(True, alpha=0.3)

    cur_true = e['ttc_true'][t]
    cur_perc = e['ttc_perc'][t]
    danger = 'DANGER' if cur_true < TTC_LOW else ('RISK' if cur_true < TTC_HIGH else 'SAFE')
    ax_main.set_title(
        f"{e['key']}  t={time_s:.1f}s  true TTC={cur_true:.2f}s  "
        f"perceived TTC={cur_perc:.2f}s  [{danger}]", fontsize=11)


# ===================== main =====================
def parse_args():
    p = argparse.ArgumentParser(
        description=("Render a perception-limited dangerous scenario as a static "
                     "PNG (most dangerous frame) and an MP4/GIF animation of the "
                     "whole episode. Two modes: (a) auto-select the most typical "
                     "dangerous episode across all H5 files matching H5_GLOB "
                     "(default), or (b) render a specific episode given --path and "
                     "--episode."))
    p.add_argument('--path', type=str, default=None,
                   help="h5 file path. If set together with --episode, skip auto "
                        "selection and render this specific episode.")
    p.add_argument('--episode', type=int, default=None,
                   help="episode id (the integer in 'episode_<id>'). Required with "
                        "--path for specific-episode mode.")
    p.add_argument('--out-image', type=str, default=OUT_IMAGE,
                   help=f"output PNG path (default: {OUT_IMAGE})")
    p.add_argument('--out-video', type=str, default=OUT_VIDEO,
                   help=f"output MP4 path (default: {OUT_VIDEO})")
    p.add_argument('--list-episodes', action='store_true',
                   help="list all episode ids in the given --path and exit")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- list mode ----
    if args.list_episodes:
        if not args.path:
            raise SystemExit("--list-episodes requires --path")
        with h5py.File(args.path, 'r') as f:
            keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
            print(f"{len(keys)} episodes in {args.path}")
            print(keys[:20], "..." if len(keys) > 20 else "")
        return

    # ---- pick the episode to render ----
    if args.path:
        if args.episode is None:
            raise SystemExit("--path requires --episode (use --list-episodes to see ids)")
        e = load_one_episode(args.path, args.episode)
        print(f"Loaded {e['key']} from {args.path}")
    else:
        files = sorted(glob.glob(H5_GLOB))
        if not files:
            raise SystemExit(f"No HDF5 files found for: {H5_GLOB}")
        print(f"Found {len(files)} h5 files")
        eps = collect_episodes(files)
        print(f"Loaded {len(eps)} episodes")
        if not eps:
            raise SystemExit("No episodes with data")
        e, _ = select_episode(eps)

    danger_i = int(np.argmin(e['ttc_true']))
    T = e['traj'].shape[0]
    collision = e['collision']
    out_image = args.out_image
    out_video = args.out_video

    layout_kw = dict(figsize=(14, 8), gridspec_kw={'height_ratios': [2.2, 1]})

    # ---- static image at the most dangerous moment ----
    fig_img, (ax_m, ax_t) = plt.subplots(2, 1, **layout_kw)
    draw_frame(ax_m, ax_t, e, danger_i, collision)
    fig_img.suptitle("Perception-limited dangerous scenario (most dangerous moment)",
                     fontsize=13)
    fig_img.subplots_adjust(top=0.92, hspace=0.38)
    fig_img.savefig(out_image, dpi=150)
    plt.close(fig_img)
    print(f"Saved image: {out_image}")

    # ---- video of the whole episode ----
    fig, (ax_m2, ax_t2) = plt.subplots(2, 1, **layout_kw)

    def update(idx):
        draw_frame(ax_m2, ax_t2, e, idx, collision)
        return []

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / FPS, blit=False)

    try:
        anim.save(out_video, writer=animation.FFMpegWriter(fps=FPS, bitrate=2500))
        print(f"Saved video: {out_video}")
    except Exception as ex:  # fallback to GIF if ffmpeg unavailable
        gif_path = out_video.rsplit('.', 1)[0] + '.gif'
        anim.save(gif_path, writer=animation.PillowWriter(fps=FPS))
        print(f"FFmpeg unavailable ({ex}); saved GIF: {gif_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
