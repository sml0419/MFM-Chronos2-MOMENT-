"""
Evaluation functions for forecasting models.
"""

import numpy as np
import torch
from tqdm import tqdm

from .utils import clear_memory, expected_seq_len, pad_or_truncate_to_seq_len


def evaluate_forecasting(model, test_loader, device, target_idx, y_scaler=None):
    """Evaluate forecasting performance (optionally denormalized)

    Args:
        model: Trained forecasting model
        test_loader: DataLoader for test data
        device: torch device
        target_idx: Index of target variable
        y_scaler: Optional scaler for denormalization

    Returns:
        metrics: Dict with MSE, MAE, RMSE
        all_preds: Predictions array
        all_trues: Ground truth array
    """
    model.eval()

    all_preds = []
    all_trues = []
    oom_count = 0

    with torch.no_grad():
        for batch_idx, (timeseries, forecast, input_mask) in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)
                forecast = forecast.float().to(device)

                with torch.cuda.amp.autocast():
                    seq_len_expected = expected_seq_len(model)

                    timeseries, input_mask = pad_or_truncate_to_seq_len(timeseries, input_mask, seq_len_expected)

                    output = model(x_enc=timeseries, input_mask=input_mask)

                all_preds.append(output.forecast[:, target_idx:target_idx+1, :].cpu().numpy())
                all_trues.append(forecast[:, target_idx:target_idx+1, :].cpu().numpy())

                # Clean up
                del timeseries, input_mask, forecast, output

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n  WARNING: OOM in evaluation batch {batch_idx}. Skipping batch...")
                    clear_memory()
                    oom_count += 1
                    continue
                else:
                    raise e

    if oom_count > 0:
        print(f"Total OOM events during evaluation: {oom_count}")

    if len(all_preds) == 0:
        print("ERROR: No predictions were generated due to OOM errors.")
        return {"MSE": float('inf'), "MAE": float('inf'), "RMSE": float('inf')}, None, None

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    if y_scaler is not None:
        original_shape = all_preds.shape

        all_preds = y_scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(original_shape)
        all_trues = y_scaler.inverse_transform(all_trues.reshape(-1, 1)).reshape(original_shape)

    # Calculate metrics
    mse = np.mean((all_trues - all_preds) ** 2)
    mae = np.mean(np.abs(all_trues - all_preds))
    rmse = np.sqrt(mse)

    return {"MSE": mse, "MAE": mae, "RMSE": rmse}, all_preds, all_trues
