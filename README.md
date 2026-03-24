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
|`.   └── project.py`                  | templay of a project file with all settings |

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

```bash
$python -m pyneuronautofit -h
Usage: __main__.py [flags] input_file_with_currents_and_target_stats (abf,npz,or json)

Options:
  -h, --help            show this help message and exit

  Fitting:
    Parameters related to Evolutionary Optimization

    -A ALGOR, --algorithm=ALGOR
                        Algorithm for multiobjective evaluation. It can be:
                        Krayzman - for Krayzman's fitness weighting; NSGA2 -
                        for Pareto nondominate selection; Max - for max scaled
                        summation; PsitiveCor - positive correlation (the same
                        as Krayzman's procedure, but with goal of make all
                        correlations positive).Algorithm can be given by first
                        letter K, N, M or P correspondingly. (Default is K)
    -P PSZ, --population-size=PSZ
                        population size (default 256). If it is a negative
                        number: the population size is the length of the
                        fitness vector multiple by absolute value of this
                        option.
    -G NGN, --number-generation=NGN
                        number of generation (default 256)
    -E ELITES, --number-elites=ELITES
                        number of elites in the replacement (default 32)
    -L, --off-log-scale
                        enable log scaling
    -I INITPOP, --init-population=INITPOP
                        file with a set of initial population
    -N KRTHR, --Krayzman-threshould=KRTHR
                        Threshould for Krayzman's iteration procedure of
                        weights adaptation (default 0.05)
    -U UPDATE, --scales-update=UPDATE
                        vector length * this scale is number of fitness
                        vectors before update Krayzman's weights or max
                        scalers (default 10)
    -y, --norm-space    normalize space under the curve
    -H, --hold-weights-normalization
                        hold weights without normalization in iteration
                        procedure (default disable)
    -b BOUNDKGA, --bound-Krayzman-weights=BOUNDKGA
                        bound weights by [1/x,x] (default disable)
    -M MRATE, --mutation-rate=MRATE
                        Basic mutation rate (default 10%%)
    -S AMSLOPE, --adaptive-mutation-slope=AMSLOPE
                        Adaptive mutation slope
    -Q VPVSIZE, --v-dvdt-hist-size=VPVSIZE
                        v dv/dt histogram size (default 12)
    -J, --inJect-elits  enable dynamic elits

  Model:
    Conditions for model running and evaluation

    -m EMODE, --eval-mode=EMODE
                        mode for evaluation T-spike time, S-spike shape,
                        U-subthreshould voltage dynamics, W-spike width, R -
                        resting potential, L - post-stimulus tail, M - voltage
                        stimulus statistics, A - average spike shape, N -
                        number of spikes (default RAMN)
    -k EMASK, --eval-mask=EMASK
                        mask to limit analysis
    -c ESPC, --spike-count=ESPC
                        number of spikes for evaluation (2)
    -t ETHSH, --spike-threshold=ETHSH
                        spike threshold (default 0.)
    -l ELEFT, --left-spike-samples=ELEFT
                        left window of spike (default 70)
    -r ERGHT, --right-spike-samples=ERGHT
                        right window of spike (default 140)
    -q TEMP, --temperature=TEMP
                        temperature (default 35)
    -z SPWTGH, --spike-Zoom=SPWTGH
                        if positive absolute weight of voltage diff during
                        spike; if negative relataed scaler
    -e, --collapse-diff
                        Collapse difference between a model and data in a
                        vector with size = number of tests (i.e. for  -m RAMNT
                        the diff vector will be length 5)

  Run:
    Options for entire EC running and logging

    -n NTH, --number-threads=NTH
                        number of threads (default None - autodetection)
    --dt=SIMDT          if positive absolute simulation dt; if negative scaler
                        for recorded dt
    -v LL, --log-level=LL
                        Level of logging.[CRITICAL, ERROR, WARNING, INFO, or
                        DEBUG] (default INFO)
    -u, --log-to-screen
                        log to screen
    -Z, --Krayzman-debug
                        enable debug dump for adaptation weight
    -p NCH, --printed-checkpoints=NCH
                        print out checkpoints every # generation (do not print
                        out if negative)
    -d DCH, --dump-checkpoints=DCH
                        dump out checkpoints into checkpoint file every #
                        generation (do not dump out if negative, default 8)
    -i RITER, --iteration=RITER
                        adds iteration number to the runs stamp
    -a RSTEMP, --run-stamp=RSTEMP
                        Use this run stamp instead of generated
    --slurm-id=SLURMID  Add SLURM ID into timestamp
    --log-population    record population into log file
    --log-archive       record archive into log file
    --dry-run           exit after init everything
```