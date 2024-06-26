# RADTTS + Vocos vocoder

🇺🇦 Join Ukrainian Text-to-Speech community: https://t.me/speech_synthesis_uk

<a target="_blank" href="https://colab.research.google.com/drive/1kkRzbInS1cPT7XfKlT1g36yHNUxpeiVN?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

Clone the code:

```bash
git clone https://github.com/egorsmkv/radtts-vocos
cd radtts-vocos
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download Ukrainian RADTTS model:

```bash
mkdir models
cd models

wget https://github.com/egorsmkv/radtts-istftnet/releases/download/v1.0/RADTTS-Lada.pt
```

Then you can inference own texts by the following command:

```bash
python3 inference.py -c config_ljs_dap.json -r models/RADTTS-Lada.pt -t test_sentences.txt -o results/
```
