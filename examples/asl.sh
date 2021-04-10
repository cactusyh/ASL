# #!/usr/bin/env bash
# # Office31
CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model none --aug_ind 4  --seed 2021 > benchmarks/asl/Office31_A2W_4aug4_w0_c05_s2021.txt
CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office31 -d Office31 -s D -t W -a resnet50  --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model none --aug_ind 4  --seed 2021 > benchmarks/asl/Office31_D2W_4aug4_w0_c05_s2021.txt
CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office31 -d Office31 -s W -t D -a resnet50  --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model none --aug_ind 4  --seed 2021 > benchmarks/asl/Office31_W2D_4aug4_w0_c05_s2021.txt
CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office31 -d Office31 -s A -t D -a resnet50  --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model none --aug_ind 4  --seed 2021 > benchmarks/asl/Office31_A2D_4aug4_w0_c05_s2021.txt
CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office31 -d Office31 -s D -t A -a resnet50  --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model none --aug_ind 4  --seed 2021 > benchmarks/asl/Office31_D2A_4aug4_w0_c05_s2021.txt
CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office31 -d Office31 -s W -t A -a resnet50  --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model none --aug_ind 4  --seed 2021 > benchmarks/asl/Office31_W2A_4aug4_w0_c05_s2021.txt

# Office-Home
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Ar2Cl_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Ar2Pr_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Ar2Rw_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Cl2Ar_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Cl2Pr_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Cl2Rw_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Pr2Ar_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Pr2Cl_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Pr2Rw_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Rw2Ar_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Rw2Cl_4aug4_w01_c05_30ep.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 30 --num_aug 4 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Rw2Pr_4aug4_w01_c05_30ep.txt

# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 20 --num_aug 6 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Rw2Ar_6aug4_w01_c05.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 20 --num_aug 6 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Rw2Cl_6aug4_w01_c05.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 20 --num_aug 6 --aug_coeff 0.5 --wr_model model --wr_param 0.1 --aug_ind 4 --seed 2021 > benchmarks/asl/OfficeHome_Rw2Pr_6aug4_w01_c05.txt

# VisDA-2017
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/visda-2017 -d VisDA2017 -s T -t V -a resnet50 --epochs 30 --seed 2021 > benchmarks/asl/VisDA2017.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/visda-2017 -d VisDA2017 -s T -t V -a resnet101 --print-freq 500 --epochs 10 --seed 2021 --num_aug 1 --aug_coeff 0.5 --wr_model model --wr_param 0.01 --aug_ind 3 --lambd 0.01 > benchmarks/asl/VisDA2017_resnet101_1aug3_w001_c05_s2021.txt
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/visda-2017 -d VisDA2017 -s T -t V -a resnet101 --print-freq 500 --epochs 10 --seed 2023 --num_aug 1 --aug_coeff 0.5 --wr_model model --wr_param 0.01 --aug_ind 3 --lambd 0.01 > benchmarks/asl/VisDA2017_resnet101_1aug3_w001_c05_s2023.txt

# # DomainNet
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s c -t i -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s c -t p -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s c -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s c -t s -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s i -t c -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s i -t p -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s i -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s i -t s -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s p -t c -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s p -t i -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s p -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s p -t s -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s r -t c -a resnet101  --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s r -t i -a resnet101  --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s r -t p -a resnet101  --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s r -t s -a resnet101  --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s s -t c -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s s -t i -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s s -t p -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0
# CUDA_VISIBLE_DEVICES=0 python examples/asl.py data/domainnet -d DomainNet -s s -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0