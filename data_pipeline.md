# MOMENT Data Pipeline (CSV → `moment_run.py` → Model)

Time-series foundation models often require different input conventions (e.g., channel-first vs. time-first, masks, padding rules).
Rather than enforcing a single “universal” format, this repo documents **the exact format produced by our preprocessing code** and **how it is passed into MOMENT**.

---

## 1) Where the dataset → tensors are built

The key preprocessing/packaging code lives in:

- `datasets.py`
  - `create_moment_dataloader(...)`
  - `MOMENTDatasetWrapper`
  - (continual pretraining) `PretrainDataset`

For supervised forecasting, `moment_run.py` constructs PyTorch dataloaders via `create_moment_dataloader(...)`, which internally builds a `Dataset_Custom` and wraps it for MOMENT. 

---

## 2) What a training batch looks like (right before the model call)

`create_moment_dataloader(...)` returns a `DataLoader` whose batch elements are:

```text
(timeseries, forecast, input_mask)
```

And the shapes are:

```text
timeseries : [batch_size, n_channels, seq_len]
forecast   : [batch_size, n_channels, pred_len]
input_mask : [batch_size, seq_len]
```

This is created by `MOMENTDatasetWrapper`, which takes the sliding-window output of `Dataset_Custom` and **transposes** it to MOMENT’s expected channel-first format:

- `context = torch.FloatTensor(seq_x.T)` → `[n_channels, seq_len]`
- `forecast = torch.FloatTensor(seq_y.T)` → `[n_channels, pred_len]`
- `input_mask = torch.ones(seq_len)` → `[seq_len]`

(Then DataLoader stacks them into batch-first tensors.)

---

## 3) How it is passed into MOMENT (very briefly)

Inside training/evaluation, the model is called with keyword arguments:

```text
output = model(x_enc=timeseries, input_mask=input_mask)
```

For forecasting, the trainer computes loss between:

```text
output.forecast  vs.  forecast
```

---

## 4) Notes for adding other models

Because each foundation model has its own conventions, the recommended approach is:

- Keep **this** pipeline as the “MOMENT adapter”
- Add new models with their own adapter/wrapper that consumes the same raw DataFrame / windows
- Or reuse `Dataset_Custom` and write a new wrapper that outputs the new model’s expected tensors.

A practical pattern is:

```text
Dataset_Custom (sliding windows, scaling) 
  → ModelSpecificWrapper (transpose/masks/padding)
  → DataLoader
  → model(...)
```
