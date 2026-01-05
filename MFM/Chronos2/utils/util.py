import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_forecast(
    context_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,   # ground truth (y_true_df 역할)
    target_column: str,
    timeseries_id,
    id_column: str = "id",
    timestamp_column: str = "TimeStamp",
    history_length: int = 256,
    prediction_length: int = 196,
    title_suffix: str = "",
    ax=None
):
    # helpers
    def _find_time_col(df):
        for name in ("TimeStamp", "timestamp", "Time", "time", "date", "DateTime"):
            if name in df.columns:
                return name
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                return c
        raise KeyError("No timestamp-like column found in dataframe")

    def _ensure_ts(df, col):
        if col not in df.columns:
            alt = _find_time_col(df)
            if alt:
                df[col] = pd.to_datetime(df[alt])
            else:
                raise KeyError(f"No timestamp column found for dataframe and '{col}' is missing.")
        else:
            df[col] = pd.to_datetime(df[col])
        return col

    # 1) Timestamp normalize
    #ts_col_ctx  = timestamp_column if timestamp_column in context_df.columns else _find_time_col(context_df)
    ts_col_test = timestamp_column if timestamp_column in test_df.columns else _find_time_col(test_df)
    ts_col_pred = timestamp_column if timestamp_column in pred_df.columns else _find_time_col(pred_df)

    #context_df[ts_col_ctx]  = pd.to_datetime(context_df[ts_col_ctx])
    test_df[ts_col_test]    = pd.to_datetime(test_df[ts_col_test])
    pred_df[ts_col_pred]    = pd.to_datetime(pred_df[ts_col_pred])

    # 2) Ensure id column exists (create constant id if missing)
    if id_column not in context_df.columns:
        context_df[id_column] = 0
    if id_column not in test_df.columns:
        test_df[id_column] = 0
    if id_column not in pred_df.columns:
        pred_df[id_column] = 0

    # 3) Filter by timeseries id
    #ctx_sel  = context_df[context_df[id_column] == timeseries_id].copy()
    test_sel = test_df[test_df[id_column] == timeseries_id].copy()
    pred_sel = pred_df[pred_df[id_column] == timeseries_id].copy()

    #ctx_sel  = ctx_sel.set_index(ts_col_ctx).sort_index()
    test_sel = test_sel.set_index(ts_col_test).sort_index()
    pred_sel = pred_sel.set_index(ts_col_pred).sort_index()

    # 4) Trim context to last history_length points
    """
    if history_length is not None:
        ctx_vals = ctx_sel.index
        if len(ctx_vals) > 0:
            start_idx  = max(0, len(ctx_vals) - history_length)
            plot_cutoff = ctx_vals[start_idx]
            ctx_plot   = ctx_sel[ctx_sel.index >= plot_cutoff]
        else:
            ctx_plot = ctx_sel
    else:
        ctx_plot = ctx_sel
    """
    # 5) Identify prediction column
    pred_col = None
    for name in ("predictions", "prediction", "pred", "value"):
        if name in pred_sel.columns:
            pred_col = name
            break
    if pred_col is None:
        skip_q = {f"{q:.1f}" for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
        for c in pred_sel.columns:
            if pd.api.types.is_numeric_dtype(pred_sel[c]) and c not in skip_q:
                pred_col = c
                break
    if pred_col is None:
        raise KeyError(f"No predictions column found in `pred_df`. Available columns: {list(pred_sel.columns)}")

    # 6) 제한할 예측 구간(timestamp)
    unique_pred_times = pd.Index(sorted(pred_sel.index.unique()))
    if len(unique_pred_times) == 0:
        raise ValueError("pred_df contains no timestamps for the requested timeseries_id")
    recent_times = unique_pred_times[-prediction_length:]
    pred_plot = pred_sel[pred_sel.index.isin(recent_times)]
    test_plot = test_sel[test_sel.index.isin(recent_times)]

    # quantile interval 있는지 확인
    lower_q = "0.1" if "0.1" in pred_plot.columns else None
    upper_q = "0.9" if "0.9" in pred_plot.columns else None

    # 7) Series 준비
    #hist_series     = ctx_plot[target_column] if target_column in ctx_plot.columns else None
    gt_series_full  = test_sel[target_column] if target_column in test_sel.columns else None
    gt_series_hor   = test_plot[target_column] if target_column in test_plot.columns else None
    forecast_series = pred_plot[pred_col]

    # ====== 여기서부터: 정보 + metric 출력 ======
    #print(f"\n=== plot_forecast: timeseries_id = {timeseries_id} ===")
    #print(f"- history points (after trim): {0 if hist_series is None else len(hist_series)}")
    #print(f"- forecast horizon points (ground truth): {0 if gt_series_hor is None else len(gt_series_hor)}")
    #print(f"- forecast horizon points (prediction: '{pred_col}'):", len(forecast_series))

    # metric 계산 (예측 vs ground truth, horizon 구간만)
    metrics = {"MSE": None, "RMSE": None, "MAE": None}
    if gt_series_hor is not None and len(gt_series_hor) > 0:
        merged = pd.DataFrame({
            "gt":   gt_series_hor,
            "pred": forecast_series
        }).dropna()

        if len(merged) > 0:
            mse  = float(np.mean((merged["gt"] - merged["pred"]) ** 2))
            rmse = float(np.sqrt(mse))
            mae  = float(np.mean(np.abs(merged["gt"] - merged["pred"])))
            metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae}

        else:
            print("\n[Metrics] No overlapping non-NaN points between gt and pred in forecast horizon.")
    else:
        print("\n[Metrics] No ground truth available in forecast horizon for this id.")

    # ====== Plot ======
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    else:
        fig = ax.get_figure()

    # 1) 히스토리
    #if hist_series is not None and len(hist_series) > 0:
        #hist_series.plot(ax=ax, label=f"historical target", linewidth=1.2, color="xkcd:azure")

    # 2) horizon 구간 ground truth (진하게)
    if gt_series_hor is not None and len(gt_series_hor) > 0:
        gt_series_hor.plot(ax=ax, label=f"future target (ground truth)", linewidth=1.5, color="xkcd:grass green")

    # 3) 예측
    forecast_series.plot(ax=ax, label=f"forecast", linewidth=1.2, color="xkcd:violet")

    # 4) 예측 구간 band (0.1~0.9 quantile)
    if lower_q and upper_q:
        ax.fill_between(
            pred_plot.index,
            pred_plot[lower_q],
            pred_plot[upper_q],
            alpha=0.7,
            label="prediction interval",
            color="xkcd:light lavender",
        )

    # 6) 마지막 history 시점 vertical line
    #if len(ctx_sel.index) > 0:
        #last_date = ctx_sel.index.max()
        #ax.axvline(x=last_date, linestyle="--", alpha=0.5)

    ax.set_title(f"{target_column} forecast for a sample {title_suffix}")
    ax.set_xlabel("Time")
    ax.set_ylabel(target_column)
    ax.grid(True)

    # ★★★ legend를 플롯 밖으로 빼기 ★★★
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        #bbox_to_anchor=(1.02, 0.5),  # 오른쪽 바깥 중앙
        borderaxespad=0.0,
    )

    # 오른쪽에 legend 자리 확보 (rect의 오른쪽 여백 줄이기)
    # fig.tight_layout(rect=[0, 0, 0.8, 1])

    if ax is None:
        plt.show()

    return metrics

def plot_das_heatmap(importance_dict, save_path="das_heatmap.png"):
    """
    Plot DAS Importance Score Heatmap (Layers vs Heads).
    Aggregates importance from Q, K, V, O weights for each head in each layer.
    """
    num_layers = 12
    num_heads = 12
    d_model = 768
    d_head = 64
    
    heatmap_data = np.zeros((num_layers, num_heads))
    
    print("Generating DAS Importance Heatmap...")
    
    for l in range(num_layers):
        # We focus on the first self-attention layer in each block
        # Pattern: encoder.block.{l}.layer.0.self_attention.{q,k,v,o}.weight
        
        # Q, K, V weights: (d_model, d_model) -> (num_heads * d_head, d_model)
        # We want to split rows by head
        for kind in ['q', 'k', 'v']:
            key = f"encoder.block.{l}.layer.0.self_attention.{kind}.weight"
            if key in importance_dict:
                imp = importance_dict[key].float().numpy() # (768, 768)
                # Reshape to (num_heads, d_head, d_model)
                imp_reshaped = imp.reshape(num_heads, d_head, d_model)
                # Sum importance for each head
                head_imp = imp_reshaped.sum(axis=(1, 2)) # (num_heads,)
                heatmap_data[l] += head_imp
                
        # O weight: (d_model, d_model) -> (d_model, num_heads * d_head)
        # We want to split columns by head
        key = f"encoder.block.{l}.layer.0.self_attention.o.weight"
        if key in importance_dict:
            imp = importance_dict[key].float().numpy() # (768, 768)
            # Reshape to (d_model, num_heads, d_head)
            imp_reshaped = imp.reshape(d_model, num_heads, d_head)
            # Sum importance for each head
            head_imp = imp_reshaped.sum(axis=(0, 2)) # (num_heads,)
            heatmap_data[l] += head_imp

    # Normalize to [0, 1]
    if heatmap_data.max() > 0:
        heatmap_data = heatmap_data / heatmap_data.max()
        
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', origin='upper')
    plt.colorbar(label='Importance Score')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.title('DAS Importance Score (Layers vs Heads)')
    plt.xticks(range(num_heads), [f'H{i}' for i in range(num_heads)])
    plt.yticks(range(num_layers), [f'L{i}' for i in range(num_layers)])
    
    # Add text annotations
    for i in range(num_layers):
        for j in range(num_heads):
            color = "black" if heatmap_data[i, j] < 0.5 else "white"
            plt.text(j, i, f'{heatmap_data[i, j]:.2f}', 
                     ha="center", va="center", color=color, fontsize=8)
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved DAS heatmap to {save_path}")
