## Data preparation
- All datasets are publically available and can be accessed at [WOS](https://github.com/kk7nc/HDLTex), [RCV1-V2](https://trec.nist.gov/data/reuters/reuters.html), [BGC](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html)  [NYT](https://catalog.ldc.upenn.edu/LDC2008T19).
- We followed the specific details mentioned in the  [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess) repository to preprocess the original datasets (WOS, RCV1-V2, BGC and NYT). 
### 1. WOS
- The original raw dataset is added to the WOS folder. Run the following scripts to obtain the preprocessed and tokenized version of the dataset
    ```bash
    cd preprocess/WOS
    python preprocess_wos.py
    python data_wos.py
### 2. BGC
 - The original raw dataset is added to the BGC folder. Run the following script to obtain the preprocessed and tokenized version of the dataset
    ```bash
    cd preprocess/BGC
    python data_bgc.py
### 3. RCV1-V2
- The original raw dataset is available after signing an [agreement](https://trec.nist.gov/data/reuters/reuters.html). Place the downloaded `rcv1.tarz.xz` file in the `preprocess\RCV` folder and run the following scripts to obtain the preprocessed and tokenized version of the dataset
    ```bash
    cd preprocess/RCV
    python preprocess_rcv1.py
    python data_rcv1.py
### 4. NYT
- The original raw dataset is available for a [fee](https://catalog.ldc.upenn.edu/LDC2008T19).  Place the downloaded files in the `preprocess\NYT` folder and run the following script to obtain the preprocessed and tokenized version of the dataset
    ```bash
    cd preprocess/NYT
    python data_nyt.py
## 
- After accessing the dataset, run the scripts in the folder `preprocess` for each dataset separately to obtain tokenized version of dataset and the related files. These will be added in the `data/x` folder where x is the name of dataset with possible choices as: wos, bgc,rcv, and nyt.
- For reference, we have added tokenized versions of the WOS, BGC, and NYT datasets along with their related files in the `data` folder. Due to size constraints in GitHub, we could not upload the tokenized version of the RCV1-V2 dataset, which exceeds 400 MB in size
