export CUDA_VISIBLE_DEVICES="1" 
for dataset in 'rmd17_ethanol' 'rmd17_malonaldehyde' 'rmd17_toluene' 'rmd17_aspirin' 'rmd17_azobenzene' 'rmd17_naphthalene' 'rmd17_paracetamol'
do
    python test.py --dataset $dataset --hidden_dim 256 --lr 3e-4
done

for dataset in 'rmd17_uracil' 'rmd17_benzene' 'rmd17_salicylic'
do
    python test.py --dataset $dataset --hidden_dim 256 --lr 5e-4
done