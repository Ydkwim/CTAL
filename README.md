# CTAL: Pre-training Cross-modal Transformer for Audio-and-Language Representations

## Installation
- **Python >= 3.6**
- **PyTorch version >= 1.7.0**
- For pre-traininng, please prepare the GPU with at least 16GB Memory (V100, RTX3080Ti)
- To develop locally, please follow th instruction below:

```shell
    git clone https://github.com/Ydkwim/CTAL.git
    cd CTAL
    pip install -r requirements.txt
```

******
## Preprocess

- Semantic Feature: please refer to the jupyter notebook: _nontebook/preprocess_text.ipynb_

- Acoustic Feature: please refer to the jupyter notebook: _notebook/preprocess_audio.ipynb_

-----
## Upstream Pre-training

After you prepare both the acoustic and semantic features, you can start to pre-training the model with executing following shell command:

```shell
    python run_m2pretrainn.py --run transformer \
    --config path/to/your/config.yaml --name model_name
```

The pre-trained model will be saved to the path: _result/transformer/model_name_. For the convenience of all the users, we make our pre-trained upstream model available:

- CTAL-Base: https://drive.google.com/file/d/1erCQplU9it9XBNrWDyLutekKthsZIi0q/view?usp=sharing

- CTAL-Large: https://drive.google.com/file/d/1L_QIZVRybJiiG2NywcX5xQQw8Y-3Vq5I/view?usp=sharing

----
## Downstream Finetune

It is very convient to use our pre-trained upstream model for different types of audio-and-language downstream tasks, including __Sentiment Analysis__, __Emotion Recognition__, __Speaker Verification__, etc. We prepare a sample fine-tuning script __m2p_finetune.py__ here for everyone. To start the fine-tuning process, you can run the following commands:

- Sentiment Regression:
```shell
    python m2p_finetune.py --config your/config/path \
    --task_name sentiment --epochs 10 --save_path your/save/path
```

- Emotionn Classification:
```shell
    python m2p_finetune.py --config your/config/path \
    --task_name emotion --epochs 10 --save_path your/save/path
```

- Speaker Verification:
```shell
    python m2p_finetune.py --config your/config/path \
    --task_name verification --epochs 10 --save_path your/save/path
```

****
## Contact

If you have any problem to the project, please feel free to report them as issues.