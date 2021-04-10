#!/usr/bin/env bash
# Office31
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office31 -d Office31 -s A -t A -a resnet50  --epochs 50 --seed 2021 > benchmarks/train_source/Office31_A.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office31 -d Office31 -s D -t D -a resnet50  --epochs 50 --seed 2021 > benchmarks/train_source/Office31_D.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office31 -d Office31 -s W -t W -a resnet50  --epochs 50 --seed 2021 > benchmarks/train_source/Office31_W.txt

# # Office-Home
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office-home -d OfficeHome -s Ar -t Ar -a resnet50 --epochs 50 --seed 2021 > benchmarks/train_source/OfficeHome_Ar.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office-home -d OfficeHome -s Cl -t Cl -a resnet50 --epochs 50 --seed 2021 > benchmarks/train_source/OfficeHome_Cl.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office-home -d OfficeHome -s Pr -t Rr -a resnet50 --epochs 50 --seed 2021 > benchmarks/train_source/OfficeHome_Pr.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/office-home -d OfficeHome -s Rw -t Rw -a resnet50 --epochs 50 --seed 2021 > benchmarks/train_source/OfficeHome_Rw.txt

# # # VisDA-2017
# # CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/visda-2017 -d VisDA2017 -s T -t T -a resnet50  --epochs 10 -i 500 --seed 2021 > benchmarks/train_source/VisDA2017.txt
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/visda-2017 -d VisDA2017 -s T -t T -a resnet101 --epochs 10 --print-freq 500 --lr 0.001 --seed 2021 > benchmarks/train_source/VisDA2017_resnet101.txt

# DomainNet
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s c -t i -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_c2i.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s c -t p -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_c2p.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s c -t r -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_c2r.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s c -t s -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_c2s.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s i -t c -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_i2c.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s i -t p -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_i2p.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s i -t r -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_i2r.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s i -t s -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_i2s.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s p -t c -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_p2c.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s p -t i -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_p2i.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s p -t r -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_p2r.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s p -t s -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_p2s.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s r -t c -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_r2c.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s r -t i -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_r2i.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s r -t p -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_r2p.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s r -t s -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_r2s.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s s -t c -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_s2c.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s s -t i -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_s2i.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s s -t p -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_s2p.txt
# CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s s -t r -a resnet101 --epochs 20 -i 2500 --seed 0 --lr 0.01 > benchmarks/train_source/DomainNet_s2r.txt
