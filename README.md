# MFM-Chronos2-MOMENT

This repository provides time-series forecasting experiments using **MOMENT** and **Chronos2**.

- **MOMENT**: Zero-shot, Fine-tuning, Continual Learning
- **Chronos2**: See: [Chronos2 README](/MFM/Chronos2/README.md)

---
## MOMENT
---

### **1. Zero-shot (no training)**

```bash
python moment_run.py \
  --data_path "/src/Dataset/SAMYANG_dataset.csv" \
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

### **2. Target-only Fine-tuning**

```bash
python moment_run.py \
  --data_path "/src/Dataset/SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment baseline \
  --fine_tune \
  --moment_epochs 2 \
  --result_dir result \
  --run_name moment_SAMYANG_finetune \
  --seed 2026 --model_name "AutonLab/MOMENT-1-Large"
```

---

### **3. Continual Learning – Sequential Pretraining + Fine-tuning**

```bash
python moment_run.py \
  --data_path "/src/Dataset/SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment sequential \
  --pretrain_files "/src/Dataset/ai4i2020.csv,/src/Dataset/IoT.csv,/src/Dataset/Steel_industry.csv" \
  --pretrain_epochs 2 \
  --fine_tune \
  --moment_epochs 2 \
  --result_dir result \
  --run_name moment_SAMYANG_clseq \
  --seed 2026 --model_name "AutonLab/MOMENT-1-Large"
```

---

### **4. Continual Learning – Soft-masking + Fine-tuning**

```bash
python moment_run.py \
  --data_path "/src/Dataset/SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment soft_masking \
  --pretrain_files "/src/Dataset/ai4i2020.csv,/src/Dataset/IoT.csv,/src/Dataset/Steel_industry.csv" \
  --pretrain_epochs 2 \
  --importance_samples 1000 \
  --layer_to_mask "head,mlp" \
  --fine_tune \
  --moment_epochs 2 \
  --result_dir result \
  --run_name moment_SAMYANG_clsoft \
  --seed 2026 --model_name "AutonLab/MOMENT-1-Large"
```

---

### **5. Run All Experiments**

```bash
python moment_run.py \
  --data_path "/src/Dataset/SAMYANG_dataset.csv" \
  --target "SATURATOR_ML_SUPPLY_F_PV.Value" \
  --minute_interval 15 \
  --seq_len 48 \
  --pred_len 6 \
  --experiment all \
  --pretrain_files "/src/Dataset/ai4i2020.csv,/src/Dataset/IoT.csv,/src/Dataset/Steel_industry.csv" \
  --pretrain_epochs 2 \
  --fine_tune \
  --moment_epochs 2 \
  --result_dir result \
  --run_name moment_SAMYANG_all \
  --seed 2026 --model_name "AutonLab/MOMENT-1-Large"
```


