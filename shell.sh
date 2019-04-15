# data process
wget https://github.com/x39826/Pali_Tripitaka/blob/master/Data.tar.gz 

tar -xzvf Data.tar.gz

python sample.py 1

python shuffle.py sample/sample.0.src sample/sample.0.tgt

split -l 17997230 sample/sample.0.src.shuf
mv xaa Data/0.src
mv xab Data/0.src.val

split -l 17997230 sample/sample.0.tgt.shuf 
mv xaa Data/0.tgt
mv xab Data/0.tgt.val

mkdir data
python preprocess.py -train_src Data/0.src -train_tgt Data/0.tgt -valid_src Data/0.src.val -valid_tgt Data/0.tgt.val -save_data data/demo0 \
	-src_seq_length 70 -tgt_seq_length 70 --src_vocab Data/vocab_del --tgt_vocab Data/vocab_del --share_vocab 

# train translator
cd OpenNMT-py

CUDA_VISIBLE_DEVICES=0 python  train.py -data ../data/demo0 -save_model demo0-model \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 2048 -batch_type tokens -normalization tokens  -accum_count 10 \
        -optim sparseadam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0 -valid_steps 5000 -save_checkpoint_steps 10000 \
        -world_size 1 -gpu_ranks 0  -adasoftmax -valid_batch_size 1 --share_embeddings \
        -tgt_vocab_cutoff  16347 47348 76893 107895 138888 169890 200211 231216 260775 291771 322776 353777 383633 414637 445632 476640

