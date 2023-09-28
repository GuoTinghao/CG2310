export CUDA_VISIBLE_DEVICES="0" 
for dataset in 'rmd17_ethanol' 'rmd17_malonaldehyde'
do
    python main.py --dataset $dataset --hidden_dim 256 --lr 3e-4 --n_epochs 10000 --wandb_project $dataset --avoid_same second
done

for dataset in 'rmd17_ethanol' 'rmd17_malonaldehyde'
do
    python main.py --dataset $dataset --hidden_dim 64 --lr 3e-4 --n_epochs 10000 --wandb_project $dataset --avoid_same second
done