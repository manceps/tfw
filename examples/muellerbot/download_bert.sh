# Download official BERT models:

# [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip):
#  12-layer, 768-hidden, 12-heads, 110M parameters
# [BERT-Large, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip):
#  24-layer, 1024-hidden, 16-heads, 340M parameters
# [BERT-Base, Cased: 12-layer](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip):
#  768-hidden, 12-heads , 110M parameters
# [BERT-Large, Cased: 24-layer](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip):
#  1024-hidden, 16-heads, 340M parameters
# [BERT-Base, Multilingual Cased](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip):
#  (New, recommended) 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
# BERT-Base, Multilingual Uncased](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip):
#  (Orig, not recommended) (Not recommended, use Multilingual Cased instead): 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
# [BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip):
#  Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters


export BERT_MODEL_CASED=True
export BERT_MODELS_DIR=~/midata/public/models/bert

if [ "$BERT_MODEL_CASED" == "False" ]; then
    # https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    export BERT_MODEL_DATE=2018_10_18
    export BERT_MODEL_NAME=uncased_L-12_H-768_A-12
else
    # https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
    export BERT_MODEL_DATE=2018_10_18
    export BERT_MODEL_NAME=cased_L-12_H-768_A-12
fi

export BERT_MODEL_DIR="$BERT_MODELS_DIR/$BERT_MODEL_NAME"
export BERT_MODEL_ZIP="$BERT_MODEL_DIR.zip"
export UNZIPPED_MODEL_PATH="$BERT_MODELS_DIR/$BERT_MODEL_NAME"
export CONFIG_PATH="$UNZIPPED_MODEL_PATH/bert_config.json"
export CHECKPOINT_PATH="$UNZIPPED_MODEL_PATH/bert_model.ckpt"
export DICT_PATH="$UNZIPPED_MODEL_PATH/vocab.txt"

# multilingual cased model (recommended):
# https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip

# old multilingual uncased model:
# https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip

# uncased models:
# https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

# cased models:
# https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
# https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip

mkdir -p $BERT_MODELS_DIR

if [ ! -f $BERT_MODEL_ZIP ]; then
    # -c continues a partial download, but this isn't useful with the above if statement
    wget -c -O "$BERT_MODEL_ZIP" "https://storage.googleapis.com/bert_models/$BERT_MODEL_DATE/$BERT_MODEL_NAME.zip"
                                # https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
fi

# cd "$BERT_MODELS_DIR"
# -f freshens files that already exist, only if they are older, but doesn't create if not there
# -u updates or creates files as needed
unzip -u -d "$BERT_MODELS_DIR" "$BERT_MODEL_ZIP"
# mv $BERT_MODEL_DATE/ $BERT_MODELS_DIR/
