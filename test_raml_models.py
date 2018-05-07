import os

train_src="../dynet_nmt/data/train.de-en.de.wmixerprep"
train_tgt="../dynet_nmt/data/train.de-en.en.wmixerprep"
dev_src="../dynet_nmt/data/valid.de-en.de"
dev_tgt="../dynet_nmt/data/valid.de-en.en"
test_src="../dynet_nmt/data/test.de-en.de"
test_tgt="../dynet_nmt/data/test.de-en.en"

for temp in [0.5]:
    job_name = 'iwslt14.raml.corrupt_ngram.t%.3f' % temp
    train_log = 'train.' + job_name + '.log'
    model_name = 'model.' + job_name
    decode_file = 'iwslt14.test.en.raml.corrupt_ngram.t%.3f' % temp
    job_file = 'scripts/train.%s.sh' % job_name
    
    with open(job_file, 'w') as f:
        f.write("""#!/bin/sh

python nmt.py \
    --cuda \
    --mode test \
    --load_model models/{model_name}.bin \
    --beam_size 5 \
    --decode_max_time_step 100 \
    --save_to_file decode/{decode_file} \
    --test_src {test_src} \
    --test_tgt {test_tgt}

echo "test result" >> logs/{train_log}
perl multi-bleu.perl {test_tgt} < decode/{decode_file} >> logs/{train_log}
      
""".format(model_name=model_name, temp=temp, 
    train_src=train_src, train_tgt=train_tgt, 
    dev_src=dev_src, dev_tgt=dev_tgt,
    test_src=test_src, test_tgt=test_tgt,
    train_log=train_log, decode_file=decode_file))

    os.system('bash submit_job.sh %s' % job_file)
