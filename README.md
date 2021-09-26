# A set of tools for fitting single- or multi-compartment neuron models parameters, using NSGA2 or Krayzman's dynamically weighted multi-objective optimization


### Scripts, templates, and helpers to fit neuron model

|:---------------------------------:|:----------------:|
|├── README.md                      | This file        |
|├── pyneuronautofit                | module with extension of inspyred |
|│   ├── autofit.py                  | |
|│   ├── evaluator.py                 | evaluates a model
|│   ├── fitter.py                    | main script for fitting|
|│   ├── __init__.py                   |            |
|│   ├── __main__.py -> fitter.py     | just link for `python -m pyneuronautofit` |
|│   └── runandtest.py                | runs and tests a model can be call independently |
|├── scripts                        | directory with useful scripts |
|│   ├── recovery-archive.sh          | recover archive if run was aborted |
|│   └── unique-in-ArXive.py          | collects only unique models |
|└── templates                       |  |
|    └── project.py                  | project file with all settings |

