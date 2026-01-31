import pandas as pd
import numpy as np
import os
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


from utils.signals import SIGNALS
from utils.simulate import simulate_series
from utils.errors import ERROR_MODELS
from statsmodels.tsa.stattools import acf as sm_acf

from utils.cv import naive_kfold_splits, block_kfold_splits, block_with_buffer_splits, WalkForwardSpec, walk_forward_splits




plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
    'font.family': 'serif',  
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'figure.dpi': 300
})

COLUMN_WIDTH = 5  


def load_data(FOLDERS, MODEL_FILES):
    df_list = []
    print("Loading data...")
    
    for error_label, folder_path in FOLDERS.items():
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}. Skipping {error_label}.")
            continue
            
        for model_label, filename in MODEL_FILES.items():
            file_path = os.path.join(folder_path, filename)
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    if 'cv_time_seconds' in df.columns:
                        df['error_type'] = error_label
                        df['model_display'] = model_label
                        df = df[df['cv_time_seconds'] > 0] 
                        df_list.append(df)
                    else:
                        print(f"Skipping {filename} in {error_label}: 'cv_time_seconds' column missing.")
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                print(f"Warning: File {filename} not found in {folder_path}")

    if not df_list:
        raise ValueError("No data loaded. Please check your FOLDERS paths.")

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Successfully loaded {len(full_df)} rows.")
    return full_df



def aggregate_times(full_df):
    scheme_map = {
        'naive': 'Naive',
        'block': 'Block',
        'block_buffered': 'Buffered',
        'walk_forward': 'Walk Forward'
    }
    
    df = full_df.copy()
    df['cv_scheme_display'] = df['cv_scheme'].map(scheme_map)
    
    grouped = df.groupby(['model_display', 'error_type', 'cv_scheme_display'])['cv_time_seconds'].mean().reset_index()
    return grouped




def draw_nested_heatmap(df, name):
    basename = os.path.basename(name)            
    filename_no_ext = os.path.splitext(basename)[0] 
    
    prefix = "heatmap_running_time_"
    if filename_no_ext.startswith(prefix):
        raw_signal = filename_no_ext[len(prefix):] 
    else:
        raw_signal = filename_no_ext 
        
    signal_title = raw_signal.replace('_', ' ').upper()


    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_box_aspect(1) 

    models = ['Kernel', 'Spline', 'XGBoost']
    errors = ['AR1', 'MA5', 'ARIMA', 'Seasonal']
    
    sub_pos = {
        'Naive':    (0.0, 0.5), 
        'Block':    (0.5, 0.5), 
        'Buffered': (0.0, 0.0), 
        'Walk Forward':   (0.5, 0.0) 
    }
    
    vmin = df['cv_time_seconds'].min()
    vmax = df['cv_time_seconds'].max()
    
    cmap = plt.cm.YlOrRd 
    norm = LogNorm(vmin=vmin, vmax=vmax)

    patches_list = []
    colors_list = []

    for x_idx, model in enumerate(models):
        for y_idx, error in enumerate(errors):
            
            cell_data = df[(df['model_display'] == model) & (df['error_type'] == error)]
            
            for scheme, (dx, dy) in sub_pos.items():
                center_x = x_idx + dx + 0.25
                center_y = y_idx + dy + 0.25
                
                val_row = cell_data[cell_data['cv_scheme_display'] == scheme]
                
                if not val_row.empty:
                    val = val_row['cv_time_seconds'].values[0]
                    rect = patches.Rectangle(
                        (x_idx + dx, y_idx + dy), 0.5, 0.5,
                        edgecolor='white', linewidth=0.5
                    )
                    patches_list.append(rect)
                    colors_list.append(val)
                    
                    label_text = f"{scheme}\n\n{val:.2f} s"
                    
                    text_obj = ax.text(
                        center_x, center_y, 
                        label_text, 
                        ha='center', va='center', 
                        fontsize=8,             
                        fontweight='bold',
                        color='black'
                    )
                    text_obj.set_path_effects([
                        pe.withStroke(linewidth=2.5, foreground='white')
                    ])
                    
                else:
                    rect = patches.Rectangle(
                        (x_idx + dx, y_idx + dy), 0.5, 0.5,
                        facecolor='#f0f0f0', edgecolor='white', linewidth=0.5
                    )
                    ax.add_patch(rect)
                    ax.text(center_x, center_y, "N/A", ha='center', va='center', 
                            fontsize=8, color='gray')

    if patches_list:
        collection = PatchCollection(patches_list, cmap=cmap, norm=norm)
        collection.set_array(np.array(colors_list))
        ax.add_collection(collection)
    
    ax.set_xlim(0, len(models))
    ax.set_ylim(0, len(errors))
    
    ax.set_xticks(np.arange(len(models)) + 0.5)
    ax.set_xticklabels(models, fontsize=14) 
    
    ax.set_yticks(np.arange(len(errors)) + 0.5)
    ax.set_yticklabels(errors, fontsize=14)
    
    ax.set_xlabel("Regression Method", fontsize=16, labelpad=15)
    ax.set_ylabel("Error Type", fontsize=16, labelpad=15)
    
    ax.set_title("Average Running Time", fontsize=18, fontweight='bold', pad=35)
    
    ax.text(0.5, 1.02, f"Signal: {signal_title}", 
            transform=ax.transAxes, 
            ha='center', va='bottom', 
            fontsize=14, fontweight='normal', color='#404040')

    for x in range(1, len(models)):
        ax.axvline(x, color='black', linewidth=1.5)
    for y in range(1, len(errors)):
        ax.axhline(y, color='black', linewidth=1.5)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)

    if patches_list:
        cbar = plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Running Time (seconds)", fontsize=14)
        
        exp_min = math.floor(math.log10(vmin))
        exp_max = math.ceil(math.log10(vmax))
        
        ticks = []
        for exp in range(exp_min, exp_max + 1):
            base = 10**exp
            for step in [1, 2, 5]:
                val = base * step
                if val >= vmin * 0.9 and val <= vmax * 1.1:
                    ticks.append(val)
        
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(name)
    print(f"Plot saved to {name}")
    plt.show()

    
def plot_signals_with_errors(n, noise_params):

    signals = list(SIGNALS.keys())
    noises = list(ERROR_MODELS.keys())
    
    n_rows = len(signals)
    n_cols = len(noises)
    
    fig_height = 2.0 * n_rows 
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(COLUMN_WIDTH * n_cols if n_cols < 3 else COLUMN_WIDTH * 2, fig_height),
                             sharex=True, squeeze=False, constrained_layout=True)

    for i, f_id in enumerate(signals):
        for j, err_id in enumerate(noises):
            ax = axes[i][j]
            params = noise_params.get(err_id, {})
            
            sim = simulate_series(
                n=n, f_id=f_id, error_model=err_id, params=params,
                x_kind="time", standardize_signal=True, seed=1234,
            )
            
            ax.plot(sim.x, sim.y, color="tab:blue", alpha=0.8, linewidth=1, label="Observed")
            ax.plot(sim.x, sim.f, color="tab:orange", linestyle="--", linewidth=1.5, label="Signal")

            if i == 0:
                ax.set_title(err_id.replace('_', ' ').upper(), fontweight='bold', fontsize=8)
            
            if j == 0:
                ylabel = f_id.replace('_', ' ').title()
                ax.set_ylabel(ylabel, fontweight='bold')
            
            if i == n_rows - 1:
                ax.set_xlabel("Time ($t$)")

            ax.grid(True, linestyle=':', alpha=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), 
               ncol=2, frameon=False, fontsize=8) 
    
    fig.suptitle("Simulated Signals with Error Processes", fontsize=15, fontweight='bold')
    plt.savefig("figures/signals_with_errors.pdf")
    
    plt.show()


def plot_acf(n_acf, noise_params):

    noises = list(ERROR_MODELS.keys())
    
    fig, axes = plt.subplots(2, 2, 
                             figsize=(COLUMN_WIDTH*1.75, COLUMN_WIDTH), 
                             sharex=True, sharey=True, constrained_layout=True)
    
    axes_flat = axes.flatten()

    lags = 40
    for j, err_id in enumerate(noises):
        if j >= len(axes_flat): break 
        ax = axes_flat[j]
        
        err_fn = ERROR_MODELS[err_id]
        params = noise_params.get(err_id, {})
        
        eps = err_fn(n=n_acf, rng=np.random.default_rng(1234), **params)
        acf_vals = sm_acf(eps, nlags=lags, fft=True)

        ci = 1.96 / np.sqrt(len(eps))
        ax.axhspan(-ci, ci, color="gray", alpha=0.2)

        markerline, stemlines, baseline = ax.stem(range(lags + 1), acf_vals)
        plt.setp(stemlines, linewidth=1)
        plt.setp(markerline, markersize=3)
        plt.setp(baseline, linewidth=0.5, color="black")
        
        clean_title = err_id.replace('_', ' ').upper()
        ax.text(0.95, 0.925, clean_title, transform=ax.transAxes, 
                ha='right', va='top', fontweight='bold', fontsize=7,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", alpha=0.8))
        
        if j % 2 == 0:
            ax.set_ylabel("ACF")
            
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Lag")

    fig.suptitle("Autocorrelation of Error Terms", fontsize=10, fontweight='bold')
    plt.savefig("figures/ACF.pdf")

    plt.show()

def plot_indices_4_schemes(N_SAMPLES):

    fig, axs = plt.subplots(2, 2, figsize=(COLUMN_WIDTH, COLUMN_WIDTH), constrained_layout=True) 

    gen_1 = naive_kfold_splits(n=N_SAMPLES, k=5, seed=42)
    plot_cv_indices(gen_1, axs[0, 0], N_SAMPLES, "Naive k-Fold")

    gen_2 = block_kfold_splits(n=N_SAMPLES, k=5)
    plot_cv_indices(gen_2, axs[0, 1], N_SAMPLES, "Block CV")

    gen_3 = block_with_buffer_splits(n=N_SAMPLES, k=5, l=5)
    plot_cv_indices(gen_3, axs[1, 0], N_SAMPLES, "Buffered CV", max_folds_to_plot=10)
    axs[1, 0].set_xlabel("Index")

    spec = WalkForwardSpec(initial_train_size=10, val_size=5, step=5)
    gen_4 = walk_forward_splits(n=N_SAMPLES, spec=spec)
    plot_cv_indices(gen_4, axs[1, 1], N_SAMPLES, "Walk-Forward")
    axs[1, 1].set_xlabel("Index")

    handles = [
        patches.Patch(color='tab:blue', label='Train'),
        patches.Patch(color='tab:red', label='Val'),
        patches.Patch(color='silver', label='Skip')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), 
               ncol=3, frameon=False, fontsize=8)
    
    fig.suptitle("Cross-Validation Splitting Schemes", fontsize=10, fontweight='bold')
    plt.savefig("figures/Cv_schemes.pdf")

    plt.show()

def plot_cv_indices(cv_split_gen, ax, n_samples, title, max_folds_to_plot=None):
    splits = list(cv_split_gen)
    if max_folds_to_plot and len(splits) > max_folds_to_plot:
        splits = splits[:max_folds_to_plot]
    n_splits = len(splits)

    c_train = 'tab:blue'
    c_val = 'tab:red'
    c_buff = 'silver'

    for i, (train, test) in enumerate(splits):
        for j in range(n_samples):
            rect = patches.Rectangle((j - 0.25, i - 0.25), 0.5, 0.5, edgecolor='none', facecolor=c_buff)
            ax.add_patch(rect)
        for j in train:
            rect = patches.Rectangle((j - 0.25, i - 0.25), 0.5, 0.5, edgecolor='none', facecolor=c_train)
            ax.add_patch(rect)
        for j in test:
            rect = patches.Rectangle((j - 0.25, i - 0.25), 0.5, 0.5, edgecolor='none', facecolor=c_val)
            ax.add_patch(rect)

    ax.set_title(title, pad=5, fontsize=9, fontweight='bold')
    
    if n_splits > 10:
        ax.set_yticks([0, n_splits-1])
        ax.set_yticklabels(["Fold 1", f"Fold {n_splits}"])
    else:
        ax.set_yticks(np.arange(n_splits))
        ax.set_yticklabels([f"F{i+1}" for i in range(n_splits)])
        
    ax.set_xlim(-1, n_samples)
    ax.set_ylim(-0.5, n_splits - 0.5)
    ax.invert_yaxis()
    ax.axis('on') 


def plot_hyperparameter_combined(df, model_name=None):

    df_subset = df[df['model'] == model_name].copy() if model_name else df.copy()
    
    scheme_map = {
        'Naive': 'Naive', 'Block': 'Block', 'Buffered': 'Buffered', 
        'Window': 'Wind.', 'Walk Forward': 'Walk-Forward'
    }
    df_subset['cv_short'] = df_subset['cv_scheme_display'].map(scheme_map).fillna(df_subset['cv_scheme_display'])

    df_subset['pct_deviation'] = df_subset.groupby('cv_scheme_display')['selected_param'].transform(
        lambda x: (x - x.median()) / (x.median() + 1e-9) * 100
    )

    raw_param = df_subset['param_name'].iloc[0] if 'param_name' in df_subset.columns else "Param"
    latex_map = {'lambda': r'$\lambda$', 'h': r'$h$', 'sigma': r'$\sigma$'}
    param_label = latex_map.get(raw_param, raw_param.capitalize())


    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 4.5), sharex=True, constrained_layout=True)

    sns.boxplot(
        data=df_subset, x='cv_short', y='selected_param', hue='cv_short',
        legend=False, palette='Set2', width=0.6, linewidth=1,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize": 3},
        ax=axes[0]
    )
    axes[0].set_title(f"Selected Parameter: {param_label}", fontweight='bold', fontsize=9)
    axes[0].set_ylabel("Value")
    axes[0].set_xlabel("")
    axes[0].grid(axis='y', linestyle=':', alpha=0.5)

    sns.boxplot(
        data=df_subset, x='cv_short', y='pct_deviation', hue='cv_short',
        legend=False, palette='Set2', width=0.6, linewidth=1,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize": 3},
        ax=axes[1]
    )
    axes[1].set_title("Stability (% Deviation from Median)", fontweight='bold', fontsize=9)
    axes[1].set_ylabel("% Deviation")
    axes[1].set_xlabel("CV Scheme") 
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    axes[1].grid(axis='y', linestyle=':', alpha=0.5)

    fig.suptitle(f"Hyperparameter Analysis: {model_name}", fontsize=10, fontweight='bold')
    plt.savefig("figures/hyperparameter_analysis.pdf")
    
    plt.show()

def plot_cv_reliability(df_kernel, df_spline, df_xgb):

    FULL_WIDTH = 7.5 
    HEIGHT = 3.5 
    
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, HEIGHT), 
                             sharex=True, sharey=True, constrained_layout=True)

    data_map = [
        (df_kernel, "Kernel Regression"),
        (df_spline, "Penalized Splines"),
        (df_xgb, "XGBoost")
    ]
    
    colors = {'Naive': 'tab:blue', 'Block': 'tab:orange', 'Buffered': 'tab:green', 
              'Window': 'tab:purple', 'Walk Forward': 'tab:red'}

    all_est = pd.concat([df['cv_est_error'] for df in [df_kernel, df_spline, df_xgb]])
    all_true = pd.concat([df['test_error'] for df in [df_kernel, df_spline, df_xgb]])
    
    g_min = min(all_est.min(), all_true.min())
    g_max = max(all_est.max(), all_true.max())
    
    limit_min = g_min * 0.8
    limit_max = g_max * 1.5

    for i, (ax, (df, title)) in enumerate(zip(axes, data_map)):
        
        present_schemes = [s for s in ['Naive', 'Block', 'Buffered', 'Window', 'Walk Forward'] 
                           if s in df['cv_scheme_display'].unique()]
        
        for scheme in present_schemes:
            subset = df[df['cv_scheme_display'] == scheme]
            ax.scatter(
                subset['cv_est_error'], 
                subset['test_error'], 
                alpha=0.6, 
                label=scheme,
                color=colors.get(scheme, 'gray'),
                edgecolor='w',
                linewidth=0.5,
                s=20
            )

        ax.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', linewidth=1)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5, which='both')
        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.set_aspect('equal', adjustable='box') 
        ax.tick_params(axis='both', labelsize=8)

        if i == 0:
            ax.set_ylabel("True Test Error (MSE)", fontsize=9)
        if i == 1:
            ax.set_xlabel("CV Estimated Error (MSE)", fontsize=9)


        ax.text(0.05, 0.95, "UNDERESTIMATION", transform=ax.transAxes,
                fontsize=5, color='darkred', ha='left', va='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkred", alpha=0.85, linewidth=0.5))

        ax.text(0.95, 0.05, "OVERESTIMATION", transform=ax.transAxes,
                fontsize=5, color='darkblue', ha='right', va='bottom', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkblue", alpha=0.85, linewidth=0.5))

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    fig.legend(by_label.values(), by_label.keys(), 
               loc='lower center', 
               bbox_to_anchor=(0.5, -0.12), 
               ncol=5, 
               fontsize=8, 
               frameon=False,
               title="CV Scheme", title_fontsize=8)
    fig.suptitle("Reliability Comparison Across Models", fontsize=11, fontweight='bold')

    plt.savefig("figures/cv_reliability_comparison.pdf", bbox_inches='tight')
    plt.show()


def plot_cv_reliability_comparison_grid(
    df_kernel_1, df_spline_1, df_xgb_1,
    df_kernel_2, df_spline_2, df_xgb_2,
    save_path="figures/cv_reliability_comparison_grid.pdf"
):

    FULL_WIDTH = 7.5
    HEIGHT = 6.0 
    
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, HEIGHT), 
                             sharex=True, sharey=True, constrained_layout=True)

    rows_config = [
        (0, "ARIMA (2,0,5)", [
            (df_kernel_1, "Kernel Regression"),
            (df_spline_1, "Penalized Splines"),
            (df_xgb_1, "XGBoost")
        ]),
        (1, "ARIMA (2,0,20)", [
            (df_kernel_2, "Kernel Regression"),
            (df_spline_2, "Penalized Splines"),
            (df_xgb_2, "XGBoost")
        ])
    ]
    
    colors = {'Naive': 'tab:blue', 'Block': 'tab:orange', 'Buffered': 'tab:green', 
              'Window': 'tab:purple', 'Walk Forward': 'tab:red'}

    all_dfs = [df_kernel_1, df_spline_1, df_xgb_1, df_kernel_2, df_spline_2, df_xgb_2]
    all_est = pd.concat([df['cv_est_error'] for df in all_dfs])
    all_true = pd.concat([df['test_error'] for df in all_dfs])
    
    g_min = min(all_est.min(), all_true.min())
    g_max = max(all_est.max(), all_true.max())
    
    limit_min = g_min * 0.8
    limit_max = g_max * 1.5

    for row_idx, row_title_text, models_data in rows_config:
        for col_idx, (df, model_title) in enumerate(models_data):
            ax = axes[row_idx, col_idx]
            
            present_schemes = [s for s in ['Naive', 'Block', 'Buffered', 'Window', 'Walk Forward'] 
                               if s in df['cv_scheme_display'].unique()]
            
            for scheme in present_schemes:
                subset = df[df['cv_scheme_display'] == scheme]
                ax.scatter(
                    subset['cv_est_error'], 
                    subset['test_error'], 
                    alpha=0.6, 
                    label=scheme,
                    color=colors.get(scheme, 'gray'),
                    edgecolor='w',
                    linewidth=0.5,
                    s=15
                )

            ax.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', linewidth=1)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(limit_min, limit_max)
            ax.set_ylim(limit_min, limit_max)
            ax.grid(True, alpha=0.3, linewidth=0.5, which='both')
            ax.set_aspect('equal', adjustable='box') 
            ax.tick_params(axis='both', labelsize=8)

            if row_idx == 0:
                ax.set_title(model_title, fontsize=10, fontweight='bold')

            if col_idx == 2:
                ax.text(1.05, 0.5, row_title_text, 
                        transform=ax.transAxes, 
                        ha='left', va='center', 
                        fontsize=10, fontweight='bold', rotation=0)

            ax.text(0.05, 0.95, "UNDERESTIMATION", transform=ax.transAxes,
                    fontsize=5, color='darkred', ha='left', va='top', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkred", alpha=0.85, linewidth=0.5))

            ax.text(0.95, 0.05, "OVERESTIMATION", transform=ax.transAxes,
                    fontsize=5, color='darkblue', ha='right', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkblue", alpha=0.85, linewidth=0.5))


    axes[0, 0].set_ylabel("True Test Error (MSE)", fontsize=9, fontweight='normal')
    axes[1, 0].set_ylabel("True Test Error (MSE)", fontsize=9, fontweight='normal')
    
    axes[1, 1].set_xlabel("CV Estimated Error (MSE)", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    fig.legend(by_label.values(), by_label.keys(), 
               loc='lower center', 
               bbox_to_anchor=(0.5, -0.08), 
               ncol=5, 
               fontsize=8, 
               frameon=False,
               title="CV Scheme", title_fontsize=8)

    fig.suptitle("Reliability Comparison: Effect of Error Correlation Depth", fontsize=11, fontweight='bold')

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_comparison_results(x_test, y_test, 
                        y_pred_std, y_pred_fast, 
                        lam_std, lam_fast, 
                        lam_grid, cv_scores_std, cv_scores_fast):

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 5.5), constrained_layout=True)


    ax0 = axes[0]

    ax0.scatter(x_test, y_test, alpha=0.3, s=5, label='Test Data', 
                color='gray', zorder=1)

    ax0.plot(x_test, y_pred_std, 
                linestyle='-', linewidth=2.5, color='tab:blue', alpha=0.5,
                label=f'Standard ($\lambda$={lam_std:.4f})', zorder=2)

    ax0.plot(x_test, y_pred_fast, 
                linestyle='--', linewidth=1, color='tab:red',
                label=f'FastNCV ($\lambda$={lam_fast:.4f})', zorder=3)

    ax0.set_xlabel('X', fontsize=9)
    ax0.set_ylabel('Y', fontsize=9)
    ax0.set_title('Predictive Equivalence', fontsize=10, fontweight='bold')

    ax0.legend(fontsize=7, loc='best', framealpha=0.9, edgecolor='gray')
    ax0.grid(True, alpha=0.3, linestyle=':')
    ax0.tick_params(labelsize=7)



    ax1 = axes[1]

    ax1.loglog(lam_grid, cv_scores_std, 'o-', 
                label='Standard CV', color='tab:blue', 
                linewidth=2, alpha=0.5, markersize=3)

    ax1.loglog(lam_grid, cv_scores_fast, 'x--', 
                label='FastNCV', color='tab:red', 
                linewidth=1, markersize=3)

    ax1.axvline(lam_std, color='black', linestyle=':', alpha=0.6, linewidth=1)


    ax1.set_xlabel('Lambda ($\lambda$)', fontsize=9)
    ax1.set_ylabel('CV Error (MSE)', fontsize=9)
    ax1.set_title('CV Score Surface Comparison', fontsize=10, fontweight='bold')

    ax1.legend(fontsize=7, loc='best', framealpha=0.9, edgecolor='gray')
    ax1.grid(True, alpha=0.3, linestyle=':', which='both')
    ax1.tick_params(labelsize=7)

    plt.savefig("figures/method_comparison_combined.pdf")
    plt.show()



def plot_time_comparison(df_results, log_scale=True):

    scale_type = 'log' if log_scale else 'linear'
    
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 5.5), 
                             sharex=True, constrained_layout=True)


    ax0 = axes[0]
    ax0.plot(df_results["n"], df_results["time_std"], 'o-', 
             label="Standard Leave-(2l+1)-Out", 
             color='tab:red', linewidth=1.5, markersize=4)
    
    ax0.plot(df_results["n"], df_results["time_fast"], 's-', 
             label="Fast NCV ", 
             color='tab:green', linewidth=1.5, markersize=4)

    scale_desc = "Log-Log" if log_scale else "Linear"
    ax0.set_title(f"Execution Time vs Sample Size", fontsize=10, fontweight='bold')
    ax0.set_ylabel("Time (seconds)", fontsize=9)
    
    ax0.set_yscale(scale_type)
    ax0.set_xscale(scale_type)
    
    ax0.tick_params(axis='both', which='major', labelsize=7)
    ax0.grid(True, alpha=0.3, linestyle=':', which='both')
    
    ax0.legend(fontsize=7, loc='upper left', framealpha=0.9, edgecolor='gray')


    ax1 = axes[1]
    ax1.plot(df_results["n"], df_results["speedup"], '^-', 
             color='tab:blue', linewidth=1.5, markersize=4)

    ax1.set_title("Speedup Factor (Std / Fast)", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Sample Size ($n$)", fontsize=9)
    ax1.set_ylabel("Speedup ($x$ times)", fontsize=9)
    
    ax1.set_xscale(scale_type) 
    

    ax1.set_yscale("linear") 

    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax1.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3, linestyle=':')

    fname = f"figures/time_comparison_{scale_type}.pdf"
    plt.savefig(fname)
    plt.show()



def aggregate_errors(full_df):

    scheme_map = {
        'naive': 'Naive',
        'block': 'Block',
        'block_buffered': 'Buffered',
        'walk_forward': 'Walk Forward'
    }
    
    df = full_df.copy()
    if 'cv_scheme' in df.columns:
        df['cv_scheme_display'] = df['cv_scheme'].map(scheme_map).fillna(df['cv_scheme'])
    
    grouped = df.groupby(['model_display', 'error_type', 'cv_scheme_display'])['test_error'].median().reset_index()
    return grouped


def draw_triplet_heatmap(df_sig1, df_sig2, df_sig3, signal_names, save_path):
    
    dfs = [df_sig1, df_sig2, df_sig3]
    
    all_vals = pd.concat([d['test_error'] for d in dfs])
    vmin = all_vals.min()
    vmax = all_vals.max()
    if vmin <= 0: vmin = 1e-6
    
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd 

    fig, axes = plt.subplots(1, 3, figsize=(27, 10), constrained_layout=True)
    
    models = ['Kernel', 'Spline', 'XGBoost']
    errors = ['AR1', 'MA5', 'ARIMA', 'Seasonal']
    
    sub_pos = {
        'Naive':        (0.0, 0.5), 
        'Block':        (0.5, 0.5), 
        'Buffered':     (0.0, 0.0), 
        'Walk Forward': (0.5, 0.0) 
    }

    for i, (ax, df, title) in enumerate(zip(axes, dfs, signal_names)):
        
        ax.set_box_aspect(1) 
        
        patches_list = []
        colors_list = []
        
        for x_idx, model in enumerate(models):
            for y_idx, error in enumerate(errors):
                
                cell_data = df[(df['model_display'] == model) & (df['error_type'] == error)]
                
                for scheme, (dx, dy) in sub_pos.items():
                    center_x = x_idx + dx + 0.25
                    center_y = y_idx + dy + 0.25
                    
                    val_row = cell_data[cell_data['cv_scheme_display'] == scheme]
                    
                    if not val_row.empty:
                        val = val_row['test_error'].values[0]
                        
                        rect = patches.Rectangle(
                            (x_idx + dx, y_idx + dy), 0.5, 0.5,
                            edgecolor='white', linewidth=0.5
                        )
                        patches_list.append(rect)
                        colors_list.append(val)
                        
                        label_text = f"{scheme}\n\n{val:.2f}"
                        
                        text_obj = ax.text(
                            center_x, center_y, 
                            label_text, 
                            ha='center', va='center', 
                            fontsize=12, 
                            fontweight='bold',
                            color='black'
                        )
                        text_obj.set_path_effects([
                            pe.withStroke(linewidth=2.5, foreground='white')
                        ])
                        
                    else:
                        rect = patches.Rectangle(
                            (x_idx + dx, y_idx + dy), 0.5, 0.5,
                            facecolor='#f0f0f0', edgecolor='white', linewidth=0.5
                        )
                        ax.add_patch(rect)
                        ax.text(center_x, center_y, "N/A", ha='center', va='center', 
                                fontsize=8, color='gray')

        if patches_list:
            collection = PatchCollection(patches_list, cmap=cmap, norm=norm)
            collection.set_array(np.array(colors_list))
            ax.add_collection(collection)
        
        ax.set_xlim(0, len(models))
        ax.set_ylim(0, len(errors))
        
        ax.set_xticks(np.arange(len(models)) + 0.5)
        ax.set_xticklabels(models, fontsize=16) 
        
        ax.set_yticks(np.arange(len(errors)) + 0.5)
        if i == 0:
            ax.set_yticklabels(errors, fontsize=16)
            ax.set_ylabel("Error Type", fontsize=18, labelpad=15)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.set_xlabel("Regression Method", fontsize=18, labelpad=15)
        
        clean_title = title.replace('_', ' ').upper()
        ax.set_title(clean_title, fontsize=20, fontweight='bold', pad=20)
        
        for x in range(1, len(models)):
            ax.axvline(x, color='black', linewidth=1.5)
        for y in range(1, len(errors)):
            ax.axhline(y, color='black', linewidth=1.5)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', length=0)

    cbar = fig.colorbar(collection, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    cbar.set_label("Median Test Error (MSE)", fontsize=18)
    
    exp_min = math.floor(math.log10(vmin))
    exp_max = math.ceil(math.log10(vmax))
    ticks = []
    for exp in range(exp_min, exp_max + 1):
        base = 10**exp
        for step in [1, 2, 5]:
            val = base * step
            if val >= vmin * 0.9 and val <= vmax * 1.1:
                ticks.append(val)
    
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=14)

    fig.suptitle("Performance Comparison: Median Test Error", fontsize=24, fontweight='bold', y=1.05)

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
