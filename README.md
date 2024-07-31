<h1 align="center"> Enhancing Precision Drug Recommendations via In-depth Exploration of Motif Relationships </h1>


## About Our Work

Update: 2023/11/29: We have created a repository for the paper titled *Enhancing Precision Drug Recommendations via In-depth Exploration of Motif Relationships*, which has been accepted by the *IEEE Transactions on Knowledge and Data Engineering (TKDE)*. In this repository, we offer the original sample datasets, preprocessing scripts, and algorithm files to showcase the reproducibility of our work.

![image-20230916163855435](https://s2.loli.net/2023/09/16/NfG4LQew7V53UEM.png)

## Requirements

- Python == 3.8.10
- Pytorch == 1.12.0
- rdkit == 2023.3.2
- dgl == 1.13.1
- pandas == 2.0.3
- matplotlib == 3.5.2

## Data Sets

The structure of the data set should be like,

```powershell
data
|_ raw
|  |_ ADDMISSIONS.csv
|  |_ DIAGNOSES_ICD.csv
|  |_ PATIENTS.csv
|  |_ PRESCRIPTIONS.csv
|  |_ PROCEDURES_ICD.csv
|  |_ RXCUIatc4.csv
|  |_ drug-atc.csv
|  |_ drug-DDI.csv
|  |_ drugbank_drugs_info.csv
|  |_ idx2SMILES.pkl
|  |_ ndc2atc_level4.csv
|  |_ ndc2RXCUI.txt
|_ raw2
|  |_ ...
```

After processing, the structure of the data set should be like,

```powershell
data
|_ ready
|  |_ atc3toSMILES.pkl
|  |_ ddi_A_final.pkl
|  |_ ddi_mask_H.pkl
|  |_ drug_smile.pkl
|  |_ ehr_adj_final.pkl
|  |_ records_final.pkl
|  |_ smile_sub_b.pkl
|  |_ smile_sub_degree_b.pkl
|  |_ smile_sub_recency_b.pkl
|  |_ smile_sub_voc_b.pkl
|  |_ smile_sub.pkl
|  |_ voc_final.pkl
|_ ready2
|  |_ ...
```

In accordance with the requirements of the **PhysioNet Clinical Databases**, we are unable to openly share the data without prior permission. Researchers interested in accessing the data can submit their requests through the following website, and a concise application process can be found here: [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/).

## RUN

```powershell
# unzip all files into the data directory
cd /root/DrugRec/data
python preprocess.py # preprocess
cd /root/DrugRec/src
python main.py # main file
```

## Contact

Please contact czhaobo@connect.ust.hk if you have any problems.
