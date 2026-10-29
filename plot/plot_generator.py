import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np


sns.set_theme(style = "whitegrid")

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def generate_plots(room_consumption, building_vavs):
    # Mapping rooms to their attributes
    room_meta = {}
    for vav in building_vavs:
        for room in vav.rooms_associated:
            room_meta[room.room_num] = {
                'type': room.room_type,
                'bucket': room.room_bucket,
                'weight': room.get_room_weight()
            }

    # Directories
    dirs = ['output/room_type', 'output/room_size', 'output/combined']
    for d in dirs:
        os.makedirs(d, exist_ok = True)

    # Group rooms into categories
    type_groups = defaultdict(set)
    size_groups = defaultdict(set)
    combined_groups = defaultdict(set)

    for rm, meta in room_meta.items():
        type_groups[meta['type']].add(rm)
        size_groups[meta['bucket']].add(rm)
        comb_key = f"{meta['type']}_{meta['bucket']}"
        combined_groups[comb_key].add(rm)

    # Sort all time keys to build x-axis (7 days * 24 hours = 168 timesteps)
    sorted_time_keys = sorted(room_consumption.keys())  # List of (day_idx, hour_idx)

    # Major tick mark positions for days (at hour 12 of each day)
    day_tick_positions = []
    day_tick_labels = []

    for idx, (day_idx, hr_idx) in enumerate(sorted_time_keys):
        # Label at the start of the day
        if hr_idx == 0:
            day_tick_positions.append(idx)
            day_tick_labels.append(DAYS_OF_WEEK[day_idx])

    # Add a final tick at the end of the x-axis for the end of Sunday
    day_tick_positions.append(len(sorted_time_keys))
    day_tick_labels.append("End")

    def plot_group_by_rooms(groups_dict, out_dir, category_label):
        for group_name, rooms in groups_dict.items():
            if not rooms:
                continue

            num_rooms = len(rooms)
            sorted_rooms = sorted(rooms)

            # Dynamic figure size scaling based on number of rooms
            # Increases plot width/height and legend column count for large groups
            fig_width = max(16, 12 + (num_rooms // 20) * 2)
            fig_height = max(8, 6 + (num_rooms // 15) * 1.5)
            legend_cols = max(1, min(4, int(np.ceil(num_rooms / 25))))

            # Define high-contrast, distinct color palette dynamically for many rooms
            palette = sns.color_palette("tab20", n_colors = min(20,
                                                                num_rooms)) if num_rooms <= 20 else sns.color_palette(
                "husl", n_colors = num_rooms)

            for energy_idx, energy_type in enumerate(["ILM", "NILM"]):
                fig, ax = plt.subplots(figsize = (fig_width, fig_height))

                for i, rm in enumerate(sorted_rooms):
                    series = [room_consumption[tk].get(rm, [0.0, 0.0])[energy_idx] for tk in
                              sorted_time_keys]
                    ax.plot(
                        range(len(sorted_time_keys)),
                        series,
                        marker = 'o',
                        markersize = 2.5 if num_rooms > 20 else 3.5,
                        linewidth = 1.2 if num_rooms > 20 else 1.5,
                        color = palette[i % len(palette)],
                        label = f"Room {rm}"
                    )

                # Day ticks (Major)
                ax.set_xticks(day_tick_positions)
                ax.set_xticklabels(day_tick_labels, fontsize = 11)

                # Hour ticks (Minor)
                ax.set_xticks(range(len(sorted_time_keys)), minor = True)
                ax.tick_params(axis = 'x', which = 'minor', length = 4, color = 'gray')
                ax.tick_params(axis = 'x', which = 'major', length = 8, color = 'black')

                ax.set_title(f"{group_name} - Average Daily {energy_type} Energy by Day of Week",
                             fontsize = 15, pad = 12)
                ax.set_xlabel("Day of Week", fontsize = 12)
                ax.set_ylabel(f"Average {energy_type} Energy (kBTU)", fontsize = 12)

                # Place legend outside plot area with multi-column support for many rooms
                ax.legend(
                    title = "Room",
                    bbox_to_anchor = (1.02, 1),
                    loc = 'upper left',
                    ncol = legend_cols,
                    fontsize = 9 if num_rooms > 30 else 10,
                    title_fontsize = 11
                )

                # Use bbox_inches='tight' on savefig instead of tight_layout() to avoid UserWarning
                plt.savefig(
                    f"{out_dir}/{group_name}_{energy_type}.png",
                    dpi = 300,
                    bbox_inches = 'tight'
                )
                plt.close(fig)

    print("Generating room_type plots...")
    plot_group_by_rooms(type_groups, 'output/room_type', 'Room Type')

    print("Generating room_size plots...")
    plot_group_by_rooms(size_groups, 'output/room_size', 'Room Size')

    print("Generating combined plots...")
    plot_group_by_rooms(combined_groups, 'output/combined', 'Combined')
