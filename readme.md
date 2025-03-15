# MMolCoT

MMolCoT (Multi-Modal Chain-of-Thought) is a novel framework designed to enhance molecular property prediction by integrating multi-modal molecular data. The model leverages large language models (LLMs) and advanced fusion techniques to align and extract features from 1D SMILES strings, 2D molecular graphs, and molecular textual descriptions. MMolCoT introduces Chain-of-Thought (CoT) reasoning to improve interpretability and prediction accuracy, making it a powerful tool for real-world applications such as drug discovery.

## Installation
1.	Clone the repository:
```bash
git clone https://github.com/jinchang1223/MMolCoT.git
cd MMolCoT
```

2.	Install dependencies:
```bash
pip install -r requirements.txt
```

## Data
For details regarding the data used in this project, please refer to the MPP section in [M<sup>3</sup>-20M repository](https://github.com/bz99bz/M-3).

## Usage
You can modify your settings in the configure.py file.
### generating chain-of-thought texts
```bash
python main.py --generate_cot
```

### training and evaluating the model
```bash
python main.py 
```

## Citation
If you use MMolCoT in your work, please cite our paper:

