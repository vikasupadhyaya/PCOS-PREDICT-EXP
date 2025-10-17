# PCOS-PREDICT-EXP:

project/
├─ data/ # raw or processed datasets (tracked with DVC)
│ └─ PCOS_data.csv.dvc
├─ models/ # trained models (tracked with DVC)
│ └─ model.pkl.dvc
├─ src/ # your Python scripts
│ ├─ train.py
│ ├─ evaluate.py
│ └─ utils.py
├─ experiments/ # optional experiment configs or notebooks
├─ .dvc/ # DVC metadata folder
├─ .gitignore
├─ requirements.txt
└─ dvc.yaml / dvc.lock # DVC pipeline files if using pipelines
