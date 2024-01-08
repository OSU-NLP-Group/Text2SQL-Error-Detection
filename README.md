# Text2SQL-Error-Detection
Code for our EMNLP 2023 paper [Error Detection for Text-to-SQL Semantic Parsing](https://aclanthology.org/2023.findings-emnlp.785/). An updated version is available on [arxiv](https://arxiv.org/abs/2305.13683).


## Setup
1. Install pytorch (1.12.1) and torch-geometric (2.1.0.post1) (https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). The code is tested with Python 3.8.
2. Install other required libraries.
    ```bash
    pip install -r requirements.txt
    ```
3. Download preprocessed data and model checkpoints
- Preprocessed data collected from the three base parsers is available at [url](https://drive.google.com/file/d/1E_ojxiX8m-8M4-o2wajL9o6FrWEHccuR/view?usp=drive_link).
    - Unzip the downloaded file and put the `datasets` folder in the `preprocessing` folder.
- Model checkpoints for simulated interactive evaluations for each base parser is available at [url](https://drive.google.com/file/d/1MAhIT87JxeLxfW7AIyb2KRs-RtPtvAlJ/view?usp=drive_link) (1 checkpoint each).
    - Unzip the downloaded file and put the folders in `experiments` folder.
    - `Parser_{parser}` folders are for parser-dependent baselines.

## Training
1. Prepare training data.
    
    In `preprocessing/dataset_beam.py`, choose indented data files `ed_{parser}_beam_train_sim2.json` and `ed_{parser}_beam_dev_sim2.json`. Then execute 'dataset_beam.py'. This will produce `.dat` files for training and dev sets, as well as `.pkl` files for indexers of non-terminal nodes.
    ```
    cd preprocessing
    python3 dataset_beam.py
    ```
2. Set the path to training and dev datasets, run `bash train.sh` for CodeBERT+GAT models and `train_no_graph.sh` for `CodeBERT` models.
    ```bash
    bash train.sh
    ```
## Evaluation
1. Prepare evaluation data.
    
    First choose the target evaluation dataset and source parser non-terminal node indexer in the `main()` function of `dataset_beam.py`. Then execute to obtain `{test_set}_sim2.dat`.
    ```bash
    cd preprocessing
    python3 dataset_beam.py
    ```
2. Set the path to evaluation dataset and model checkpoint, run `bash test.sh` for CodeBERT+GAT models and `test_no_graph.sh` for `CodeBERT` models.
    ```bash
    bash test.sh
    ```
    The prediction results `eval_{test_name}.json` can be found in the checkpoint folder.


## Citation
```
@inproceedings{chen-etal-2023-error,
    title = "Error Detection for Text-to-{SQL} Semantic Parsing",
    author = "Shijie Chen and Ziru Chen and Huan Sun and Yu Su",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.785",
    doi = "10.18653/v1/2023.findings-emnlp.785",
    pages = "11730--11743",
}
```
