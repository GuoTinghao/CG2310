export CUDA_VISIBLE_DEVICES="2" 
for dataset in 'rmd17_ethanol' 'rmd17_malonaldehyde'
do
    python main.py --dataset $dataset --hidden_dim 256 --lr 5e-4 --n_epochs 10000 --wandb_project $dataset --avoid_same second
done

for dataset in 'rmd17_ethanol' 'rmd17_malonaldehyde'
do
    python main.py --dataset $dataset --hidden_dim 64 --lr 5e-4 --n_epochs 10000 --wandb_project $dataset --avoid_same second
done