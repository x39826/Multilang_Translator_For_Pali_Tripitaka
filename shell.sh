wget https://github.com/x39826/Pali_Tripitaka/blob/master/Data.tar.gz 
tar -xzvf Data.tar.gz

python preprocess.py -train_src Data/0.src -train_tgt Data/0.tgt -valid_src Data/0.src.val -valid_tgt Data/0.tgt.val -save_data data/demo0 \
	-src_seq_length 70 -tgt_seq_length 70 --src_vocab Data/vocab_del --tgt_vocab Data/vocab_del --share_vocab 
