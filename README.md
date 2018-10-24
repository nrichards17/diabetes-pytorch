# diabetes-pytorch

**Please see [notebooks/PR-Diabetes-Results.ipynb](https://github.com/nrichards17/diabetes-pytorch/blob/master/notebooks/PR-Diabetes-Results.ipynb)
for full results!**

## Excerpts from Results:
From UCI ML Repository    
["Diabetes 130-US hospitals for years 1999-2008 Data Set"](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008#)

**Goal**: build model(s) to predict which patients will be re-hospitalized within 30 days

**Evaluate**: using AUROC

Notes:
- 'encounter_id' - unique admissions
- ignore 'patient_nbr' - treat all encounters independent
- 'readmitted' - treat 'NO' as '>30'
- Attributes: 55
- Samples: >100k
- Features: numerical, categorical


### Results

| Metric   | Train  | Val    | Test   |
|----------|--------|--------|--------|
| Accuracy | 0.6164 | 0.6074 | 0.6260 |
| AUROC    | 0.6570 | 0.6428 | 0.6650 |
| Loss     | 0.6584 | 0.6653 | 0.6579 |


### Model
Mixed-input deep neural network with categorical embeddings

```
Network(
  (embeddings): ModuleList(
    (0): Embedding(6, 3)
    (1): Embedding(2, 1)
    (2): Embedding(4, 2)
    (3): Embedding(22, 11)
    (4): Embedding(15, 8)
    (5): Embedding(9, 5)
    (6): Embedding(10, 5)
    (7): Embedding(10, 5)
    (8): Embedding(4, 2)
    (9): Embedding(4, 2)
    (10): Embedding(4, 2)
    (11): Embedding(4, 2)
    (12): Embedding(4, 2)
    (13): Embedding(4, 2)
    (14): Embedding(4, 2)
    (15): Embedding(4, 2)
    (16): Embedding(2, 1)
    (17): Embedding(2, 1)
  )
  (dropout_emb): Dropout(p=0.5)
  (bn_continuous): BatchNorm1d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc_layers): ModuleList(
    (0): FCUnit(
      (linear): Linear(in_features=67, out_features=512, bias=True)
      (batchnorm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout): Dropout(p=0.65)
    )
    (1): FCUnit(
      (linear): Linear(in_features=512, out_features=256, bias=True)
      (batchnorm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout): Dropout(p=0.65)
    )
    (2): FCUnit(
      (linear): Linear(in_features=256, out_features=64, bias=True)
      (batchnorm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout): Dropout(p=0.65)
    )
    (3): FCUnit(
      (linear): Linear(in_features=64, out_features=64, bias=True)
      (batchnorm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout): Dropout(p=0.65)
    )
    (4): FCUnit(
      (linear): Linear(in_features=64, out_features=64, bias=True)
      (batchnorm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout): Dropout(p=0.65)
    )
    (5): FCUnit(
      (linear): Linear(in_features=64, out_features=64, bias=True)
      (batchnorm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout): Dropout(p=0.65)
    )
    (6): FCUnit(
      (linear): Linear(in_features=64, out_features=64, bias=True)
      (batchnorm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout): Dropout(p=0.65)
    )
    (7): FCUnit(
      (linear): Linear(in_features=64, out_features=64, bias=True)
      (batchnorm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout): Dropout(p=0.65)
    )
    (8): FCUnit(
      (linear): Linear(in_features=64, out_features=32, bias=True)
      (batchnorm): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout): Dropout(p=0.65)
    )
  )
  (output_linear): Linear(in_features=32, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
```

