import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from activediff.meep_compute_fom import double_with_mirror

CHANNELS = os.path.expanduser('~/scratch/nanophoto/lowfom/nodata/fields/channels.npy')
IMAGE = os.path.expanduser('~/scratch/nanophoto/1image.npy')

def plot_channels():
    channels = np.load(CHANNELS)

    # Identify channel pixels (high refractive index)
    threshold = (channels.min() + channels.max()) / 2
    is_channel = channels > threshold

    # Find gap between top and bottom channel regions
    row_sums = is_channel.sum(axis=1)
    channel_rows = np.where(row_sums > 0)[0]
    row_gaps = np.where(np.diff(channel_rows) > 1)[0]
    top_channel_bottom = 44
    bot_channels_top = 190 - 45
    xmin = 12
    xmax = 205 - 12

    # Normalize then add rectangle of 1s in the gap region
    channels = (channels - channels.min()) / (channels.max() - channels.min())
    channels[top_channel_bottom:bot_channels_top + 1, xmin:xmax + 1] = 1.0

    print(f"Rectangle: y=[{top_channel_bottom}, {bot_channels_top}], x=[{xmin}, {xmax}]")
    fig, ax = plt.subplots()
    ax.imshow(channels, aspect='auto')
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])

    # Red contour around the design region
    rect = Rectangle((xmin - 0.5, top_channel_bottom - 0.5),
                      xmax - xmin + 1, bot_channels_top - top_channel_bottom + 1,
                      linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Inner rectangle: same height, half the width, right edge aligned with outer
    inner_width = (xmax - xmin + 1) / 2
    inner_x = xmax + 0.5 - inner_width
    inner_rect = Rectangle((inner_x, top_channel_bottom - 0.5),
                            inner_width, bot_channels_top - top_channel_bottom + 1,
                            linewidth=6, edgecolor='red', facecolor='none')
    ax.add_patch(inner_rect)

    # "Design Region" label centred in the inner rectangle
    ax.text(inner_x + inner_width / 2, (top_channel_bottom + bot_channels_top) / 2,
            'Design\nRegion', color='red', fontsize=40, fontweight='bold',
            ha='center', va='center')

    # Green arrow pointing right→left, segment centred on the outer rectangle
    center_y = (top_channel_bottom + bot_channels_top) / 2
    arrow_center_x = (xmin + xmax) / 2
    arrow_half_len = ((xmax - xmin + 1) / 6)/2
    ax.annotate('', xy=(arrow_center_x - arrow_half_len, center_y),
                xytext=(arrow_center_x + arrow_half_len, center_y),
                arrowprops=dict(arrowstyle='->', color='green', lw=6))

    # Downward arrows in channels
    arrow_kw = dict(arrowstyle='->', color='red', lw=4)

    # Top channel
    top_cols = np.where(is_channel[:top_channel_bottom + 1, :].any(axis=0))[0]
    top_cx = (top_cols[0] + top_cols[-1]) / 2
    top_rows = np.where(is_channel[:top_channel_bottom + 1, :].any(axis=1))[0]
    ax.annotate('', xy=(top_cx, top_rows[-1]), xytext=(top_cx, top_rows[0]),
                arrowprops=arrow_kw)

    # Bottom channels (left and right)
    bot_cols_all = np.where(is_channel[bot_channels_top:, :].any(axis=0))[0]
    col_gaps = np.where(np.diff(bot_cols_all) > 1)[0]
    left_cols = bot_cols_all[:col_gaps[0] + 1]
    right_cols = bot_cols_all[col_gaps[0] + 1:]
    bot_rows = np.where(is_channel[bot_channels_top:, :].any(axis=1))[0] + bot_channels_top
    for cols in [left_cols, right_cols]:
        cx = (cols[0] + cols[-1]) / 2
        ax.annotate('', xy=(cx, bot_rows[-1]), xytext=(cx, bot_rows[0]),
                    arrowprops=arrow_kw)

    plt.savefig('channels.pdf', bbox_inches='tight', pad_inches=0)


def plot_channels_with_optimized_design():
    channels = np.load(CHANNELS)
    channels = (channels - channels.min()) / (channels.max() - channels.min())
    print(channels.shape)

    image = np.load(IMAGE)
    image = (image - image.min()) / (image.max() - image.min())
    image = double_with_mirror(image)
    print(image.shape)
    channels[44:190-45,12:205-12] = image

    fig, ax = plt.subplots()
    ax.imshow(channels, aspect='auto')
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])
    plt.savefig('channels_with_optimized_design.pdf', bbox_inches='tight', pad_inches=0)
    # plt.show()

def plot_train_sample():
    image = np.load(IMAGE)
    image = (image - image.min()) / (image.max() - image.min())
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])
    plt.savefig('optimized_design.pdf', bbox_inches='tight', pad_inches=0)


plot_channels()
# plot_channels_with_optimized_design()
# plot_train_sample()
