export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

python text_discriminator_pretrain.py  -c ./configs/config_text_discriminator_pretrain.yaml
