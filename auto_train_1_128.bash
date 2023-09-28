export CUDA_VISIBLE_DEVICES="0" 
for dataset in 'rmd17_aspirin' 'rmd17_azobenzene' 'rmd17_benzene' 'rmd17_ethanol' 'rmd17_malonaldehyde'
do
    python main.py --dataset $dataset --hidden_dim 128 --lr 3e-4 --n_epochs 10000 --wandb_project $dataset
    python main.py --dataset $dataset --hidden_dim 128 --lr 5e-4 --n_epochs 10000 --wandb_project $dataset
done