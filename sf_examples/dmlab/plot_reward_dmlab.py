import sys
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

matplotlib.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 18,
    }
)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
api = wandb.Api()

envs = ["dmlab_collect_good_objects", "dmlab_nonmatch", "dmlab_rooms_exploit_deferred_effects"]

dmlab_xppo_dict = {
    'dmlab_collect_good_objects': 'dmlab_collect_good_objects_sf_xppo_value_func',
    'dmlab_nonmatch': 'dmlab_nomatch_sf_xppo_value_func',
    'dmlab_rooms_exploit_deferred_effects': 'dmlab_rooms_exploit_deferred_effects_sf_xppo_value_func'
}

dmlab_appo_dict = {
    'dmlab_collect_good_objects': 'dmlab_collect_good_objects_sf_origin_v3',
    'dmlab_nonmatch': 'dmlab_nomatch_sf_origin_v3',
    'dmlab_rooms_exploit_deferred_effects': 'dmlab_rooms_exploit_deferred_effects_train_sf_origin_v3'
}

def load_groups(group_and_keys, relabel_dict, x_range, extra_filter):
    all_interp_data = []
    for group, x_key, y_key in group_and_keys:
        total_filters = {
            "$and": [
                {"group": group},
                {"$not": {"tags": "exclude-from-paper"}},
                extra_filter,
            ]
        }
        pprint(total_filters)
        runs = api.runs(
            path="resl-mixppo/stabilized-rl",
            filters=total_filters,
        )
        print(f"Got {len(runs)} runs for group {group}")
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        for r in runs:
            # h = r.history(samples=2000, keys=[x_key, y_key])
            h = pd.DataFrame(r.scan_history(keys=[x_key, y_key]))
            try:
                if np.max(h[x_key]) < 0.99 * x_range[1]:
                    print("Maximum x value of run", str(r), "is", np.max(h[x_key]))
                interp_y = np.interp(x_vals, h[x_key], h[y_key])
            except KeyError:
                print("Could not get keys in run", r)
                print(h)
            else:
                all_interp_data.append(
                    pd.DataFrame.from_dict(
                        {
                            relabel_dict.get(x_key, x_key): x_vals,
                            relabel_dict.get(y_key, y_key): interp_y,
                            relabel_dict.get("group", "group"): relabel_dict.get(
                                group, group
                            ),
                            "run": str(r),
                        }
                    )
                )
    return pd.concat(all_interp_data, ignore_index=True)


def make_plot(env):
    group_and_keys = [
        (dmlab_xppo_dict[env], "global_step", "reward/reward"),
        (dmlab_appo_dict[env], "global_step", "reward/reward"),
    ]
    relabels = {
        dmlab_xppo_dict[env]: "FixPO",
        dmlab_appo_dict[env]: "APPO",
        "global_step": "Total Environment Steps",
        "reward/reward": "reward/reward",
        "group": "Algorithm",
    }
    all_data = load_groups(
        group_and_keys,
        relabels,
        (0, 1e8),
        {
            "$and": [
                {"config.env": env},
                {"$or": [{"state": "finished"}, {"state": "running"}, {"state": "crashed"}]},
            ]
        },
    )
    return all_data


def main():
    for env in envs:
        plt.clf()
        all_data = make_plot(env=env)

        sns.lineplot(
            data=all_data,
            x="Total Environment Steps",
            y="reward/reward",
            hue="Algorithm",
            errorbar=('ci', 95),
            style="Algorithm",
            palette="viridis",
        )
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"FixPO_vs_ppo_{env}.pdf")


if __name__ == "__main__":
    sys.exit(main())
