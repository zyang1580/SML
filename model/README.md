
# MF.py 

MF model

# conv_transfer.py
CNN-based Transfer

# transfer.py
SML learning process

# baseline.py
Contain all MF-based baselines.
+ full-retrain
<!---
(`python baseline.py --method='full' --l2_u=1e-7 --l2_i=1e-7 --lr=1e-2 --epochs=100`)
note: epochs=20 can also get a comparable performance
-->
+ fine-tune
<!---
`python baseline.py --method='fine' --batch_size=64 --l2_u=0 --l2_i=0 --lr=1e-2 --epochs=5`
-->
+ SPMF
<!---
`python baseline.py --method='spmf' --batch_size=64  --l2_i=0.0 --l2_u=0.0, laten=64 --lr=0.01 --pool_size=7000`
-->

