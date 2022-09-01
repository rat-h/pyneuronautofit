# A set of tools for fitting single- or multi-compartment neuron models parameters, using NSGA2 or Krayzman's dynamically weighted multi-objective optimization


### Scripts, templates, and helpers to fit neuron model

|                                    |                 |
|:--------------------------------- |:----------------|
|`├── README.md`                      | This file        |
|`├── pyneuronautofit`                | module with extension of inspyred |
|`│   ├── autofit.py`                  | helper functions |
|`│   ├── evaluator.py`                 | evaluates a model
|`│   ├── fitter.py`                    | main script for fitting|
|`│   ├── __init__.py`                   |            |
|`│   ├── __main__.py -> fitter.py`     |: just a link for `python -m pyneuronautofit` |
|`│   └── runandtest.py`                |:  runs and tests a model can be call independently |
|`├── scripts`                        |:  directory with useful scripts |
|`│   ├── recovery-archive.sh`          |:  recover archive if run was aborted |
|`│   └── unique-in-ArXive.py`          |: collects only unique models |
|`└── templates`                       |  |
|`    └── project.py`                  | templay of a project file with all settings |

### Components

#### Evaluator
The Evaluator can perform analysis of the data and copare two data set against each other
What kind of analysis it will perform is defined by `mod` variable.
The `mod` variable is a string with one or more upper-case letters, each for specific analysis.

|key| description |
|:-:|:------------|
| A| average spike shape during stimulus            | 
| C| distance between voltages during stimulus| 
| D| distance between voltages during after stimulus tails| 
| S| spike shapes during stimulus| 
| T| spike times| 
| R| resting potential| 
| L| post-stimulus tail statistics| 
| M| voltage stimulus statistics| 
| N| number of spikes| 
| O| Just total number of spikes| 
| P| difference in probability dencity on v,dv/dt plane weighted by 1 - target_dencity/sum(target_dencity)| 
| Q| the same as P but only during stimulus.  | 
| U| squared error of subthreshold voltage| 
| V| distance between voltages| 
| W| spike width during stimulus| 
| Z| distance between voltages with zooming weight on spikes| 


