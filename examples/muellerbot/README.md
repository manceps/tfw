# Muellerbot

Unredacts the mueller report (inaccurately) using BERT.

## Quickstart

```bash
git clone https://github.com/manceps/tfw
cd tfw
conda create --name tfw --file environment.yml --yes
conda activate tfw
pip install -e .
cd examples/muellerbot
source download_bert.sh
python load_and_predict.py
```

## Usage

At the `Text:` You can type or paste any text containing `unk` tokens as the redaction markers and muellerbot will try to fill in the blanks.
At the `Redaction marker:` You can type or paste any short text without spaces, like '[MASK]' or '[HOM]' to be used as the marker. Markers are assumed to hide a single word. So your text should contain multiple contiguous markers (like "unk unk unk") to predict/unredact multiple words. If you used `unk` (the default redaction marker) then you can just hit enter.
