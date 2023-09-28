export CUDA_VISIBLE_DEVICES="0" 
for dataset in 'rmd17_aspirin' 'rmd17_azobenzene' 'rmd17_naphthalene' 'rmd17_paracetamol'
do
    python test.py --dataset $dataset --hidden_dim 256 --lr 3e-4
done