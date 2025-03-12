毕业设计

## 项目结构

```
DeepMagPurge
├── LICENSE
├── README.md
├── Testscript.py
├── Trainscript.py
├── art_simulate.sh
├── checkpoints
├── data
│   ├── species.txt
│   ├── taxon2label.txt
│   └── token_12mer.txt
├── model
│   ├── AttentionLayer.py
│   ├── Dataset.py
│   ├── EmbeddingLayer.py
│   ├── LSTMLayer.py
│   └── TaxonClassifier.py
├── my_train_script.sh
├── other
│   ├── art_simulate_genomes.py
│   └── concat_all_fq.py
└── utils
    ├── DataProcess.py
    ├── LabelGenomes.py
    └── config.py# 基于深度语音模型的宏基因组去污
```