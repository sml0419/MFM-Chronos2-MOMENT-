# MFM-Chronos2-MOMENT

This repository provides time-series forecasting experiments using **MOMENT** and **Chronos2**. **Please run each time series foundation model within its appropriate virtual environment.** 
Chronos is used with [Minjun’s chronos code](https://github.com/ssmjun/Industrial-Time-Series-Forecasting-with-Chronos-2) as-is, while MOMENT is based on [Juyoung’s code](https://github.com/Hasaero/Manufacturing-Foundation-Model) with minor modifications.


- **MOMENT**: Zero-shot, Fine-tuning, Continual Learning
- **Chronos2**: See: [Chronos2 README](/MFM/Chronos2/README.md)

---
## MOMENT
Before running experiments, please review the [default configuration options of MOMENT.](/MFM/moment/moment_cl/config.py)
Below is an example command for execution
### **1. Zero-shot (no training)**

```bash
python moment_run.py \
  --data_dir "/src/Dataset" \
  --dataset "SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment baseline \
  --result_dir result \
  --run_name moment_SAMYANG_zeroshot \
  --seed 2026 --model_name "AutonLab/MOMENT-1-Large"
```

---

### **2. "Target-only" Fine-tuning**

```bash
python moment_run.py \
  --data_dir "/src/Dataset" \
  --dataset "SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment baseline \
  --fine_tune \
  --ft_target_only \
  --ft_epochs 2 \
  --result_dir result \
  --run_name moment_SAMYANG_finetune_target \
  --seed 2026 \
  --model_name "AutonLab/MOMENT-1-Large"
```

---
### **3. Target & Covariate Fine-tuning**

```bash
python moment_run.py \
  --data_dir "/src/Dataset" \
  --dataset "SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment baseline \
  --fine_tune \
  --ft_epochs 2 \
  --result_dir result \
  --run_name moment_SAMYANG_finetune_all \
  --seed 2026 \
  --model_name "AutonLab/MOMENT-1-Large"
```

---
### **4. Continual Learning – Sequential Pretraining + Fine-tuning**

```bash
python moment_run.py \
  --data_dir "/src/Dataset" \
  --dataset "SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment sequential \
  --pretrain_files ai4i2020.csv IoT.csv Steel_industry.csv \
  --pt_epochs 2 \
  --fine_tune \
  --ft_epochs 2 \
  --result_dir result \
  --run_name moment_SAMYANG_clseq \
  --seed 2026 \
  --model_name "AutonLab/MOMENT-1-Large"

```

---

### **5. Continual Learning – Soft-masking + Fine-tuning**

```bash
python moment_run.py \
  --data_dir "/src/Dataset" \
  --dataset "SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment soft_masking \
  --pretrain_files ai4i2020.csv IoT.csv Steel_industry.csv \
  --pt_epochs 2 \
  --importance_samples 1000 \
  --fine_tune \
  --ft_epochs 2 \
  --result_dir result \
  --run_name moment_SAMYANG_clsoft \
  --seed 2026 \
  --model_name "AutonLab/MOMENT-1-Large"

```

---

### **6. Run All Experiments**

```bash
python moment_run.py \
  --data_dir "/src/Dataset" \
  --dataset "SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment all \
  --pretrain_files ai4i2020.csv IoT.csv Steel_industry.csv \
  --pt_epochs 2 \
  --fine_tune \
  --ft_target_only \
  --ft_epochs 2 \
  --result_dir result \
  --run_name moment_SAMYANG_all \
  --seed 2026 \
  --model_name "AutonLab/MOMENT-1-Large"

```


