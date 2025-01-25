# HiSLR (Hierarchical Sample-specific Label Relationships)
## Requirements
- Python>=3.6
- torch>=1.6.0
- transformers>=4.2.1
- numpy>=1.23.0

## Data
-  All datasets are publically available and can be accessed at [WOS](https://github.com/kk7nc/HDLTex), [RCV1-V2](https://trec.nist.gov/data/reuters/reuters.html), [BGC](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html)  [NYT](https://catalog.ldc.upenn.edu/LDC2008T19).
- After accessing the dataset, run the scripts in the folder `preprocess` for each dataset separately to obtain tokenized version of dataset and the related files. These will be added in the `data/x` folder where x is the name of dataset with possible choices as: wos, bgc, rcv and nyt.
-  For reference, we have added tokenized versions of the WOS, BGC, and NYT datasets along with their related files in the `data` folder. Due to size constraints in GitHub, we could not upload the tokenized version of the RCV1-V2 dataset, which exceeds 400 MB in size.
- Detailed steps regarding how to obtain and preprocess each dataset are mentioned in the readme file of `preprocess` folder 
##  Train HiSLR 
`python train.py --name t1_hislr --batch 10 --data wos --cl_loss 1  --cl_temp 0.07   --cl_wt 1` </br>
</br>
Some Important arguments: </br>
- `--name` name of directory in which your model will be saved. For e.g. the above model will be saved in `..HiSLR/data/wos/t1_hislr`
- `--batch` batch_size for training. We set it to 10 for all datasets.
- `--data` name of dataset directory which contains your data and related files. Possible choices are `wos`, `rcv`, `bgc`  and `nyt`.
- `--cl_loss` Set to 1 for using contrastive loss in Lorentz hyperbolic space
- `--cl_temp` Temperature for the contarstive loss. We use a value of 0.07 for all datasets
- `--cl_wt` weight for contrastive loss. We use the following weights for the contrastive loss across datasets: `WOS:0.3` , `RCV1-V2:0.4`, `BGC:0.4`,  and `NYT:1`


### For Exponential map transformation and Geodesic distance calculation in Lorentz hyeprbolic space
- The code for operations in lorentz geometry is in the script `lorentz.py` which is obtained from [meru](https://github.com/facebookresearch/meru/blob/main/meru/lorentz.py) repository.
- Specifically we use the functions  `exp_map0()` and `pairwise_dist()` provided in  the script `lorentz.py`

### For Contrastive loss in Lorentz hyperbolic space
- The code for the contrastive loss is in the script `criterion.py` 
### For multiple random runs 
In `train.py` set the `--rand_exp 1` to use a random seed for each run of the experiment
## Test
To run the trained model on test set run the script `test.py` </br> 
`python test.py --name t1_hislr --data wos --extra _macro` </br> </br>
Some Important arguments
- `--name` name of the directory which contains the saved checkpoint. The checkpoint is saved in `../HiSLR/data/wos/` when working with WOS dataset
- `--data` name of dataset directory which contains your data and related files
- `--extra` two checkpoints are kept based on macro-F1 and micro-F1 respectively. The possible choices are  `_macro` and `_micro` to choose from the two checkpoints
