## Data

Ion binding proteins (IBPs) and non-IBPs are stored in `./datas/data.fasta` without labels. But first 114 sequences are IBPs, and other 207 samples are non-IBPs.

## Environment build

Anaconda (Anaconda3-2021.11-Linux-x86_64) virtual environment called `ibpred` on Linux system was used.

To build `ibpred`, please execute following commands in `base`  environment of Anaconda:

```bash
conda create -n ibpred -y
conda activate ibpred
conda install python==3.9.7 numpy==1.21.2 pandas==1.3.5 scipy==1.7.1 scikit-learn==1.0.1 tenacity==8.0.1 matplotlib==3.4.3 seaborn==0.11.1 ipykernel==6.4.1 traitlets==4.3.3 black -y
conda install -c conda-forge scikit-optimize==0.9.0 -y
```

If the user wants to run Jupyter notebook version-based workflow, all in `process.ipynb`, the `ipykernel` need to be installed.

> The Jupyter notebook in VS code editor required; `traitlets` is optional for solving some bugs that can't run 2+ notebooks in the same env 
>
> However, failed :(

Otherwise you can reproduce the paper results by run the following command line in code directory (It's better to use ipynb because our results were produced by running all codes in ipynb):

```bash
python process.py
```

The `matplotlib`, `seaborn`, `black` packages are alternative for model training, but `matplotlib` and `seaborn` are necessary for plot pictures, as well `black` can be used for formatting python code.



## How to use

`auto_pred.py` is the tool to identify IBPs. Users must confirm the model and the feature subset num used for prediction. Because the default parameter `model` is the `base_bayes_clf_PD` and default `set_num` is `193`.

Command line in the `ibpred` env like this:

```bash
python auto_pred.py --file 'input_file.fasta' --model 'base_bayes_clf_PD' --set_num 193 --out 'result.csv'
```

***Notice**:* 

- Don't use the same identifiers for different samples.
- Don't place the same sequences in the input file.
