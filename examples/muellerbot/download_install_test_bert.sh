export BERT_MODEL=UNCASED
export BERT_MODELS_DIR=~/midata/public/models/bert/

if [ "$BERT_MODEL" == UNCASED ]; then
    export BERT_MODEL_DATE=2018_10_18
    export BERT_MODEL_NAME=uncased_L-12_H-768_A-12
else
    export BERT_MODEL_DATE=2018_11_23
    export BERT_MODEL_NAME=multi_cased_L-12_H-768_A-12
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
fi

# cd "$BERT_MODELS_DIR"
# -f freshens files that already exist, only if they are older, but doesn't create if not there
# -u updates or creates files as needed
unzip -u -d "$BERT_MODELS_DIR" "$BERT_MODEL_ZIP"
# mv $BERT_MODEL_DATE/ $BERT_MODELS_DIR/



conda activate bert || conda activate base
export CONDA_ENV_NAME="$(conda env list | egrep '(\W+)\s*\*' | cut -f1 -d ' ')"

if [ "bert" == "$CONDA_ENV_NAME" ] ; then
    echo "conda env: bert"
else
    conda create -n bert python=3.6
    conda activate bert
    conda install 'keras>=2.0.8'
    # conda install 'keras-gpu>=2.0.8'
    pip install 'keras-pos-embd>=0.10.0'
    pip install 'keras-transformer>=0.22.0'
    # on Ubuntu 18.04 need to upgrade to google's version of tf:
    # See: https://stackoverflow.com/a/51771078/623735
    pip uninstall tensorflow protobuf --yes
    find $CONDA_PREFIX -name "tensorflow" | xargs -Ipkg rm -rfv pkg
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl --no-cache-dir
fi



if [ -f "setup.py" ] ; then
    pip install -e .
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="1"

if [ -f "demo/load_model/load_and_predict.py" ] ; then
    cd demo/load_model/
    python load_and_predict.py
else
    if [ -f "load_and_predict.py" ] ; then
        python load_and_predict.py
    fi
fi
