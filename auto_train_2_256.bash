export CUDA_VISIBLE_DEVICES="3" 
for dataset in 'rmd17_naphthalene' 'rmd17_paracetamol' 'rmd17_salicylic' 'rmd17_toluene' 'rmd17_uracil'
do
    python main.py --dataset $dataset --hidden_dim 256 --lr 3e-4 --n_epochs 10000 --wandb_project $dataset
    python main.py --dataset $dataset --hidden_dim 256 --lr 5e-4 --n_epochs 10000 --wandb_project $dataset
done