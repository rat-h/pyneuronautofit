# Scripts, templates, and helpers to fit neuron model

### Files 

|:---------------------------------:|:----------------:|
|├── README.md                      | This file        |
|├── pyneuronautofit                | module with extension of inspyred |
|│   ├── autofit.py                  |
|│   ├── evaluator.py                 |
|│   ├── fitter.py                    | main script for fitting|
|│   ├── __init__.py                   |            |
|│   ├── __main__.py -> fitter.py     | just link for `python -m pyneuronautofit` |
|│   └── runandtest.py                | runs and tests a model can be call independently |
|├── scripts                        | directory with useful scripts |
|│   ├── recovery-archive.sh          | recover archive if run was aborted |
|│   └── unique-in-ArXive.py          | collects only unique models |
|└── templates                       |  |
|    └── project.py                  | project file with all settings |

