# GNN-PCST-R

Using Graph Neural Networks to optimize weights in the Prize Collecting Steiner Tree algorithm using the RADNET protocol.

## Environment

### Virtualenv

Access the project folder and create a Virtualenv:

```
python -m venv venv
```

After that, start the Virtualenv created earlier:

* For Unix
```
source venv/bin/active
```
* For Windows
```
venv/Scripts/activate
```

### Project dependencies

To install all project dependencies, simply run the command below:

```
$ pip install -r requirements.txt
```

Or, if you prefer, to install each project dependency manually, run the commands below:

```
pip install tqdm
pip install numpy
pip install pcst_fast
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Dataset 

the dataset used for pre-training the model is the [CAIDA](https://publicdata.caida.org/).

Download the dataset by clicking [here](https://publicdata.caida.org/datasets/as-relationships/serial-2/), then move the files to directory "inputs".

To generate the dataset, run the command below.

```
python main.py 
```