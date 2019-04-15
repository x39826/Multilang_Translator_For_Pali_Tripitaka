

cd OpenNMT-py


python translate.py  -model demo0-model_step_50000.pt -src ../Data/0.src.val --gpu 0 -output pred.txt -beam_size 10 -batch_size 20  

sed -i 's/ //g' pred.txt
sed -i 's/▁/ /g' pred.txt
cp ../Data/0.tgt.val gold.txt
sed -i 's/ //g' gold.txt
sed -i 's/▁/ /g' gold.txt
sacrebleu -i pred.txt gold.txt --metrics chrf --echo both

# translate to fixed language
python translate.py  -model demo0-model_step_50000.pt -src ../Data/0.src.val --gpu 0 -output pred.txt -beam_size 10 -batch_size 20   -trans_to tibt

python translate.py  -model demo0-model_step_50000.pt -src ../Data/0.src.val --gpu 0 -output pred.txt -beam_size 10 -batch_size 20   -trans_to beng

# Server
mv demo0-model_step_50000.pt  available_models/
python server.py
python test_sever.py
