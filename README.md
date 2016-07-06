
## Download inception cnn graph

```
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
```

Unzip it, and move the `tensorflow_inception_graph.pb` to `./data/models/graph.pb`

## Normalize dataset format

`coco2tsv.py` can be used to transform coco dataset into the following format. For the other datasets, please figure it out by yourself.

1. `trainval.tsv` of the format: `image-id \t image-file-name \t sentence`. Note that the different sentences of the same image should be in different lines
2. `test.tsv` of the same format as `trainval.tsv`
3. `images` directory

Here is an example of tsv file. For `test.tsv`, the sentences can be just single "."
```
318937417       318937417.jpg   aa alumni in the making
178613239       178613239.jpg   aa bathroom final review week
1425946864      1425946864.jpg  aa battery bottom form punch
9411126647      9411126647.jpg  aa board front and back of plane
500241682       500241682.jpg   aa canary spring mammoth hot springs
8314270221      8314270221.jpg  aa colors socks rainbow clothes
1039683274      1039683274.jpg  aa delicious moment as usual
2698595696      2698595696.jpg  aa float switch controlling the level of water in the seeping pots
2733477297      2733477297.jpg  aa gets the waitress number
2829369771      2829369771.jpg  aa gun at entrance to lion battery on signal hill
```

For the example of `images` directory, the file `images/318937417.jpg` should be found for the above `tsv` example

## Generate vocabulary

`cat *.tsv | python gen-vocab.py $PATH_DATA_FOLDER`

The file `$PATH_DATA_FOLDER/vocab` will be generated

## Compile tsv files and images into database

`python compile_data.py $PATH_TO_TSV $VOCAB_PATH $IMAGES_DIR`

The directory `$PATH_TO_TSV.tf` will be generated

## Training

`CUDA_VISIBLE_DEIVCES=$gpu python cnn2lm.py $PATH_TO_TRAINVAL_TSV.tf $VOCAB_PATH $MODEL_PATH`

The file `$MODEL_PATH` will be generated per 1000 iterations

## Evaluation

`CUDA_VISIBLE_DEIVCES=$gpu python cnn2lm.py $PATH_TO_TEST_TSV.tf $VOCAB_PATH $MODEL_PATH $SENTENCE_SAVE_PATH`

The file `$SENTENCE_SAVE_PATH-iteration` will be generated of the following format:

`image-id \t generated-caption`

Here are some examples:
```
301065891       the crowd at the end of the race
7976767557      the <UNK> of the <UNK> <UNK>
2392614879      the last of the sun is coming
4056258509      the kids and their parents
8984240335      the flaming lips off festival
109962269       the books are the ones that you can be
5378563712      the most beautiful cake ever
8764830058      the <UNK> of the <UNK> of the <UNK>
8539686511      the <UNK> of the <UNK> of the <UNK>
2190866953      the sun sets on the sea
```
