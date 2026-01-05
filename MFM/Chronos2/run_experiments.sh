#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 헤더 출력 함수
print_header() {
    echo -e "\n${BLUE}############################################################${NC}"
    echo -e "${YELLOW}👉 Running Experiment: ${BOLD}$1${NC}"
    echo -e "${BLUE}############################################################${NC}\n"
}

# ==========================================
# 실험 시작
# ==========================================

# 0. Naive Forecast (Mean Value)
print_header "0. Naive Forecast (Mean Value)"
python chronos_run.py --evaluate_naive --seq_len 12

# 1. No Covariates, No Cross Learning
print_header "1. No Covariates, No Cross Learning"
python chronos_run.py --use_chronos --seq_len 12

# 2. With Covariates, No Cross Learning
print_header "2. With Covariates, No Cross Learning"
python chronos_run.py --use_chronos --use_covariates

# 3. With Covariates, With Cross Learning
print_header "3. With Covariates, With Cross Learning"
python chronos_run.py --use_chronos --use_covariates --use_cross_learning

# 4. Fine-tuning (No Cross Learning)
print_header "4. Fine-tuning - No Cross Learning"
python chronos_run.py --use_chronos --fine_tune --use_covariates \
                    --ft_learning_rate 1e-6 --num_steps 400

# 5. Fine-tuning (With Cross Learning)
print_header "5. Fine-tuning - With Cross Learning"
python chronos_run.py --use_chronos --fine_tune --use_covariates --use_cross_learning

# 6. Continual Pretrained Model
print_header "6. Continual Pretrained Model + Fine-tuning"
python chronos_run.py --use_chronos --use_covariates --continual_pretrain --fine_tune \
                    --ft_learning_rate 1e-6 \
                    --num_steps 400 --pretrain_steps 400

# 7. Continual Pretrained Model
print_header "7. Continual Pretrained Model + Fine-tuning + Cross Learning"
python chronos_run.py --use_chronos --use_covariates --continual_pretrain --fine_tune --use_cross_learning --seq_len 12

# 8. Continual Pretrained Model + Fine-tuning + Cross Learning + DAS
print_header "8. Continual Pretrained Model + Fine-tuning + Cross Learning + Soft-CL"
python chronos_run.py --use_chronos --use_covariates --continual_pretrain --fine_tune --use_cross_learning --use_das \
                    --ft_learning_rate 5e-6 --pt_learning_rate 5e-6 \
                    --num_steps 400 --pretrain_steps 400

# 종료 메시지
echo -e "\n${GREEN}${BOLD}✅ All experiments completed successfully!${NC}\n"
