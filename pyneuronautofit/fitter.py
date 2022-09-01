import sys,os,time,logging
from optparse import OptionParser, OptionGroup
from random import Random as rnd
from numpy import *
from multiprocessing import Lock

from inspyred import ec   # evolutionary algorithm
sys.path.append('.')
from project import getversion, param_nslh
try:
	from project import downsampler
except:
	downsampler = None

oprs = OptionParser("USAGE: %prog [flags] input_file_with_currents_and_target_stats (abf,npz,or json)")
#GA
EC_Algorithm = OptionGroup(oprs,"Fitting", "Parameters related to Evolutionary Optimization" )
EC_Algorithm.add_option("-A", "--algorithm",           dest="algor", default="Krayzman",       type='str',
    help="Algorithm for multiobjective evaluation. It can be: "+\
    "Krayzman - for Krayzman's fitness weighting; "+\
    "NSGA2 - for Pareto nondominate selection; "+\
    "Max - for max scaled summation; "+\
    "PsitiveCor - positive correlation (the same as Krayzman's procedure, but with goal of make all correlations positive)."+\
    "Algorithm can be given by first letter K, N, M or P correspondingly. (Default is K)")
EC_Algorithm.add_option("-P", "--population-size",     dest="psz",    default=256,             type='int',
    help="population size (default 256). If it is a negative number: the population size is the length of the fitness vector multiple by absolute value of this option." )
EC_Algorithm.add_option("-G", "--number-generation",   dest="ngn",    default=256,             type='int',
    help="number of generation (default 256)" )
EC_Algorithm.add_option("-E", "--number-elites",       dest="elites", default=32,              type='int',
    help="number of elites in the replacement (default 32)" )
EC_Algorithm.add_option("-L", "--off-log-scale",      dest="logs",    default=True,            action="store_false",
    help="enable log scaling")
EC_Algorithm.add_option("-I", "--init-population",    dest="initpop", default=None,            type='str',
    help="file with a set of initial population" )
EC_Algorithm.add_option("-N", "--Krayzman-threshould",dest="krthr",   default=0.05,           type='float',
    help="Threshould for Krayzman's iteration procedure of weights adaptation (default 0.05)" )
# EC_Algorithm.add_option("-N", "--no-negatives",       dest="nonegcor",default=False,           action="store_true",
    # help="Adjust negative correlation first" )
EC_Algorithm.add_option('-U', "--scales-update",      dest="update",default=10.,            type='float',
    help="vector length * this scale is number of fitness vectors before update Krayzman's weights or max scalers (default 10)") 
# EC_Algorithm.add_option('-M', "--enable-masking-KGA", dest="kga_mask",default=False,           action="store_true",
    # help="Enables masking in Krayzman's multiobjective optimization") 
EC_Algorithm.add_option("-y", "--norm-space",         dest="norms",  default=False,          action="store_true",
    help="normalize space under the curve")
EC_Algorithm.add_option("-H", "--hold-weights-normalization",dest="holdKGAweights",  default=False,          action="store_true",
    help="hold weights without normalization in iteration procedure (default disable)")
EC_Algorithm.add_option("-b", "--bound-Krayzman-weights",dest="boundKGA",  default=None,      type='float',
    help="bound weights by [1/x,x] (default disable)")
EC_Algorithm.add_option("-M",   "--mutation-rate",    dest="mrate",  default=0.1,            type='float',
    help="Basic mutation rate (default 10%%)")
EC_Algorithm.add_option("-S","--adaptive-mutation-slope",  dest="amslope",default=0.06,            type='float',
    help="Adaptive mutation slope")
EC_Algorithm.add_option("-Q","--v-dvdt-hist-size",    dest="vpvsize",   default=12,              type='int',
    help="v dv/dt histogram size (default 12)")
EC_Algorithm.add_option("-J", "--inJect-elits",       dest="enableDelits",  default=False,          action="store_true",
    help="enable dynamic elits")

oprs.add_option_group(EC_Algorithm)

#MODEL
Model_Cond = OptionGroup(oprs,"Model", "Conditions for model running and evaluation" )
Model_Cond.add_option("-m", "--eval-mode",          dest="emode",   default="RAMN",          type='str',
    help="mode for evaluation T-spike time, S-spike shape, U-subthreshould voltage dynamics, W-spike width, R - resting potential, L - post-stimulus tail, M - voltage stimulus statistics, A - average spike shape, N - number of spikes (default RAMN)")
Model_Cond.add_option("-k", "--eval-mask",          dest="emask",   default="None",          type='str',
    help="mask to limit analysis")
Model_Cond.add_option("-c", "--spike-count",        dest="espc",    default=2,               type='int',
    help="number of spikes for evaluation (2)")
Model_Cond.add_option("-t", "--spike-threshold",    dest="ethsh",   default=0.,              type='float',
    help="spike threshold (default 0.)")
Model_Cond.add_option("-l", "--left-spike-samples", dest="eleft",   default=110,             type='int',
    help="left window of spike (default 70)")
Model_Cond.add_option("-r", "--right-spike-samples",dest="erght",   default=220,             type='int',
    help="right window of spike (default 140)")
Model_Cond.add_option("-q", "--temperature",        dest="temp",    default=35.,             type='float',
    help="temperature (default 35)")
Model_Cond.add_option("-z", "--spike-Zoom",         dest="spwtgh",  default=None,            type="float",
    help="if positive absolute weight of voltage diff during spike; if negative relataed scaler")
Model_Cond.add_option("-e", "--collapse-diff",      dest="cdiff",   default=False,           action="store_true",
    help="Collapse difference between a model and data in a vector with size = number of tests (i.e. for  -m RAMNT the diff vector will be length 5)")
oprs.add_option_group(Model_Cond)    

#RUN
Run_ = OptionGroup(oprs,"Run", "Options for entire EC running and logging" )
Run_.add_option("-n", "--number-threads",      dest="nth",    default=os.cpu_count(),  type='int',
    help="number of threads (default None - autodetection)" )
Run_.add_option(       "--dt",                 dest="simdt",   default=None,            type="float",
    help="if positive absolute simulation dt; if negative scaler for recorded dt")
Run_.add_option("-v", "--log-level"   ,        dest="ll",     default="INFO",          type='str',
    help="Level of logging.[CRITICAL, ERROR, WARNING, INFO, or DEBUG] (default INFO)") 
Run_.add_option("-u", "--log-to-screen",       dest="lc",     default=False,           action="store_true",
    help="log to screen")
Run_.add_option('-Z', "--Krayzman-debug",      dest="Krdb",   default=False,           action="store_true",
    help="enable debug dump for adaptation weight")
Run_.add_option("-p", "--printed-checkpoints", dest="nch",    default=-1,              type='int',
    help="print out checkpoints every # generation (do not print out if negative)" )
Run_.add_option("-d", "--dump-checkpoints",   dest="dch",     default=8,               type='int',
    help="dump out checkpoints into checkpoint file every # generation (do not dump out if negative, default 8)" )
Run_.add_option("-i", "--iteration",           dest="riter",  default=-1,              type='int',
    help="adds iteration number to the runs stamp" )
Run_.add_option("-a", "--run-stamp",           dest="rstemp", default=None,            type='str',
    help="Use this run stamp instead of generated" )
Run_.add_option(      "--slurm-id",            dest="slurmid", default=None,            type='int',
    help="Add SLURM ID into timestamp" )
Run_.add_option(      "--log-population",      dest="logpop",  default=False,           action="store_true",
    help="record population into log file")
Run_.add_option(      "--log-archive",         dest="logarx",  default=False,           action="store_true",
    help="record archive into log file")
Run_.add_option(      "--dry-run",             dest="dryrun",  default=False,           action="store_true",
    help="exit after init everything")
oprs.add_option_group(Run_) 

opt, args = oprs.parse_args()

timestamp  = "-v"+getversion()+time.strftime("-%Y%m%d-%H%M%S")
timestamp += "-{:07d}-{}".format( random.randint(1999999) if opt.slurmid is None else opt.slurmid,opt.emode )
timestamp += "-{}{}".format("L" if opt.logs else "F",opt.algor[0])
if opt.riter >= 0:
    timestamp += f"-I{opt.riter:03d}"

if len(args) != 1:
    logging.error("Need only one ABF, NPZ, or JSAON input file with currents and target states")
    raise BaseException("Need only one ABF, NPZ, or JSAON input file with currents and target states")

algorithms = {"K": "Krayzman\'s weighted adaptation", "N":"Pareto nondominate selection", "M":"max scaled sum", "P":"Positive correlation" }
if not opt.algor[0] in algorithms:
    logging.error(f"Unknow algorithm {opt.algor}. Should be K, N, M, or P")
    raise BaseException(f"Unknow algorithm {opt.algor}. Should be K, N, M, or P")
        
try:
    opt.emask = eval(opt.emask)
except BaseException as e :
    sys.stderr.write(f"INFO: Cannot evaluate condition mask {opt.emask}:{e}\n")
    sys.stderr.write(f"INFO: Treating {opt.emask} as filename\n")
    try:
        with open(opt.emask) as fd:
            emask = fd.read()
        opt.emask = eval(emask)
    except BaseException as e :
        sys.stderr.write(f"ERROR: Cannot read condition mask from the {opt.emask}:{e}\n")
        sys.stderr.write( "ERROR: ====== !!! FULL STOP !!! ======\n")
        raise BaseException(f"Cannot read condition mask from the {opt.emask}:{e}")

if not opt.rstemp is None:
    timestamp = opt.rstemp
    if opt.riter >= 0: timestamp += f"-I{opt.riter:03d}"
    logname  = timestamp+".log.gz"
    finalrec = timestamp+"-final.json"
    chpntrec = timestamp+"-chpnt.json"
    arXive   = timestamp+"-arXive.json"
    Kr_log   = timestamp+"-KrWtLg.json" if opt.Krdb else None
    if not opt.lc:
        sys.stderr.write(f"Run stamp {timestamp}\n")

else:
    fname, fext = os.path.splitext(args[0])
    logname  = fname+timestamp+".log.gz"
    finalrec = fname+timestamp+"-final.json"
    chpntrec = fname+timestamp+"-chpnt.json"
    arXive   = fname+timestamp+"-arXive.json"
    Kr_log   = fname+timestamp+"-KrWtLg.json" if opt.Krdb else None
    if not opt.lc:
        sys.stderr.write(f"Run stamp: {fname+timestamp}\n >Options: "+" ".join(sys.argv)+"\n")

if opt.lc:
    logging.basicConfig(                  format='%(asctime)s:%(name)-10s:%(lineno)-4d:%(levelname)-8s:%(message)s', level=eval("logging."+opt.ll) )
else:
    import gzip
    logfd =  gzip.open(logname, mode='wt', encoding='utf-8')
    logging.basicConfig(stream = logfd,   format='%(asctime)s:%(name)-10s:%(lineno)-4d:%(levelname)-8s:%(message)s', level=eval("logging."+opt.ll) )


try:
    from .autofit    import *
    from .runandtest import RunAndTest
    from .evaluator  import Evaluator
except:
    from pyneuronautofit.autofit    import *
    from pyneuronautofit.runandtest import RunAndTest
    from pyneuronautofit.evaluator  import Evaluator
    

logging.info( '---------------------------------------------------')
logging.info(f'SCRIPT VERSION                      : {getversion()}')
logging.info(f'RUNNING PARAMETERS                  : '+" ".join(sys.argv))
logging.info( '---------------------------------------------------')
logging.info(f'Fitting to                          : {args[0]}')
logging.info(f'Final population                    : {finalrec}')
logging.info(f'Checkpoint Charly                   : {chpntrec}')
logging.info(f'Archive                             : {arXive}')
logging.info(f'Iteration                           : {opt.riter:03d}')
logging.info(f'Forced run stamp                    : {opt.rstemp}')
logging.info( '---------------------------------------------------')

# Test  evaluator
e = Evaluator(args[0],mode = opt.emode,mask = opt.emask,prespike = opt.eleft,\
        postspike = opt.erght,spikethreshold = opt.ethsh,spikecount = opt.espc,\
        collapse_tests = opt.cdiff,vpvsize=opt.vpvsize,\
        spikeweight=opt.spwtgh, downsampler = downsampler )
vl = e.diff(e,marks=True)
vm = "".join([m for m,v in vl])
vv = e.scores()
vl = len(vl)
with open(arXive,'w') as fd:
    fd.write('{\n')
    fd.write("\t\"markers\"   :" + json.dumps(list(vm))          +",\n")
    fd.write("\t\"bvalues\"   :" + json.dumps(vv)                +",\n")
    fd.write("\t\"parameters\":" + json.dumps(param_nslh)        +",\n")
    fd.write("\t\"version\"   :" + json.dumps(getversion())      +",\n")
    fd.write("\t\"target\"    :" + json.dumps(args[0])           +",\n")
    fd.write("\t\"evaluation\":" + json.dumps({
                        'mode'           : opt.emode,
                        'mask'           : opt.emask,
                        'prespike'       : opt.eleft,
                        'postspike'      : opt.erght,
                        'spikethreshold' : opt.ethsh,
                        'spikecount'     : opt.espc,
                        'collapse_tests' : opt.cdiff,
                        'spikeweight'    : opt.spwtgh,
                        'downsampler'    : downsampler,
                        'vpvsize'        : opt.vpvsize
                      })         +",\n")
    
    fd.write("\t\"cmd\"       :" + json.dumps(" ".join(sys.argv))+",\n")
    fd.write(f"\t\"records\":[\n")
if not Kr_log is None:
    with open(Kr_log,"w")as fd:
        fd.write("{\n")
if   opt.psz <  0 : opt.psz = abs(opt.psz)*vl
elif opt.psz == 0 : opt.psz = vl

numWadapPop = int(round(opt.update*vl/(opt.psz-opt.elites))) if opt.algor[0] != "N" else 0

logging.info( 'GA:')
logging.info(f' > population size                  : {opt.psz}')
logging.info(f' > number of generation             : {opt.ngn}')
logging.info(f' > number of elites in replacement  : {opt.elites}')
logging.info(f' > Algorithm                        : {algorithms[opt.algor[0]]}')
logging.info(f' > adapt weights every              : {numWadapPop} generations')
if opt.algor[0] == 'K':
    # logging.info(f' > mitigate negative correlation    : {opt.nonegcor}')
    logging.info(f' > Threshold for Krayzman\'s proc    : {opt.krthr}')
    logging.info(f' > Normlization                     : '+('space' if opt.norms else 'maximum') )
    logging.info(f' > weights are bound by             : {opt.boundKGA}')
    logging.info(f' > hold-on weights normalization    : {opt.holdKGAweights}')
    # logging.info(f' > masking                          : {opt.kga_mask}')
    logging.info(f' > basic mutation rate              : {opt.mrate}')
    logging.info(f' > adaptive mutation slope          : {opt.amslope}')    
logging.info(f' > use log scale                    : {opt.logs}')
logging.info(f' > initial population               : {opt.initpop}')
logging.info( 'EVALUATION:')
logging.info(f' > evaluation mode                  : {opt.emode}')
logging.info(f' > evaluation mask                  : {opt.emask}')
logging.info(f' > number of spikes in evaluation   : {opt.espc}')
logging.info(f' > spike threshold                  : {opt.ethsh}')
logging.info(f' > left and right spike boundry     : {opt.eleft} : {opt.erght}')
logging.info(f' > model temperature                : {opt.temp} C')
logging.info(f' > spike zoom                       : {opt.spwtgh}')
logging.info(f' > collapse vector                  : {opt.cdiff}')
logging.info(f' > vector length                    : {vl}')
logging.info(f' > vector components                : {vm}')
for c in opt.emode:
    logging.info(f' \-> {c}                              : {len([ p for p in vm if p==c])}')
   
logging.info( 'RUN:')
logging.info(f' > number of threads                : {opt.nth}')
logging.info(f' > simulation time step             : {opt.simdt}')
logging.info(f' > log level                        : {opt.ll}')
logging.info(f' > log on screen                    : {opt.lc}')
if opt.algor[0] == 'K':
    logging.info(f' > log for Krayzman\'s weight        : {opt.Krdb}')
    logging.info(f' > file for Krayzman\'s weight       : {Kr_log}')
logging.info(f' > print out check points every     : {opt.nch}')
logging.info(f' > dump  out check points every     : {opt.dch}')


logging.info( 'PARAMETERS:')
for n,s,l,h in param_nslh:
    tl,th=f"{l}",f"{h}"
    if type(n) is tuple:
        for m in n:
            logging.info(f' > {m:<33s}: {s} : [{tl:<9s}, {th:<9s}]')
    else:
        logging.info(f' > {n:<33s}: {s} : [{tl:<9s}, {th:<9s}]')

if opt.dryrun:
    exit(0)


def list2dict(lst,param_ranges):
    d = {}
    for p,(n,s,l,h) in zip(lst,param_ranges):
        d[n] = p
    return d

loglock = Lock()

def TCevaluator(candidates, args):
    fitness    = []
    modmode    = args.get('mode', False)
    prm_ranges = args.get('param_ranges', param_nslh)
    targetFile = args["expTarget"]
    evalPram   = args.get("evalparams",{})
    evaluator  = e#Evaluator(targetFile,**evalPram)
    fitness    = [ RunAndTest(evaluator,celsius=opt.temp,lock=loglock,logname=fname+timestamp)(params=list2dict(af_ec2mod(p,prm_ranges) if opt.logs else p,prm_ranges)) for p in candidates]
    # use_Pareto = args.get('use_Pareto', False)
    # if use_Pareto:
        # fitness   = [ ec.emo.Pareto(f) for f in fitness ]
    return fitness

def RecArXive(wfitness,fitness,candidates,args):
    # ----- ArXiving -----
    algorithm    = args.get("algorithm",'K')
    logscale     = args.get("logscale",False)
    param_ranges = args['param_ranges']
    chprint      = args.get("checkpoint_print",-1)
    arX_chpoint  = args.get("checkpoint_count",-1)


    xpop = sorted([ [w,f,af_ec2mod(c,param_ranges) if logscale else c] for w,f,c in zip(wfitness,fitness,candidates) ])
    if not hasattr(RecArXive, "chcount"): RecArXive.chcount  = 0
    else                                : RecArXive.chcount += 1
    
    if arX_chpoint > 0:
        if RecArXive.chcount%arX_chpoint == 0   :
            mm = args['algorithm'][0] != "N"
            mpop = \
                sorted([ 
                    [ c.fitness if mm else c.fitness.values,af_ec2mod(c.candidate,param_ranges) if logscale else c.candidate] for c in args["_ec"].archive 
                ])
            if "checkpoint_file" in args and type(args["checkpoint_file"]) is str:
                chpntrec = args["checkpoint_file"]
            with open(chpntrec,"w") as fd:
                fd.write("{\n\t\"Generation\":"+f"{RecArXive.chcount}"+",\n\t\"Population\":[\n")
                for il,(w,f,p) in enumerate(xpop):
                    fd.write("\t\t"+json.dumps({"fitness":f, "parameters":p, "weighted-fitness":w })+",\n" if il < len(xpop)-1 else "\n")
                fd.write("\t],\n\t\"Archive\":[\n")
                for il,(f,p) in enumerate(mpop):
                    fd.write("\t\t"+json.dumps({"fitness":f, "parameters":p})+",\n" if il < len(mpop)-1 else "\n")
                fd.write("\t]\n}\n")
                


    if chprint > 0:
        if RecArXive.chcount%chprint == 0       : print(json.dumps(xpop)+"\n")
    

    if "arXive_file" in args and type(args["arXive_file"]) is str:
        arXive = args["arXive_file"]
    with open(arXive,"a") as fd:
        for il,(w,f,p) in enumerate(xpop):
            fd.write("\t\t"+json.dumps({"fitness":f, "parameters":p, "weighted-fitness":w})+ ",\n")


    if algorithm[0] == 'K':
        krayzman_log  = args.get("krayzman_log",None)
        if not krayzman_log is None and type(krayzman_log) is str and KrayzmanGA.adap_cnt == 0:
            with open(krayzman_log,"a")as fd:
                fd.write("\t\"UPDATE\":"+json.dumps(KrayzmanGA.weights.tolist())+",\n")
    
def deNANificator(fitness):
    noneid = []
    vsize  = -1
    for fi,f in enumerate(fitness):
        if f is None:
            noneid.append(fi)
            continue
        if None in f:
            noneid.append(fi)
            continue
        if vsize < 0 : vsize = len(f)
        elif vsize != len(f):
            with loglock:
                logging.error(f"Vector sizes aren't equal {vsize}!={len(f)} :{fi}")
                raise RuntimeError(f"Vector sizes aren't equal {vsize}!={len(f)} :{fi}")
            exit(1)
    
    for nid in noneid:
        fitness[nid] = [nan for i in range(vsize) ]
    fmax = nanmax( array(fitness) )
    fitness =[ [fmax*1e3 if isnan(f) else f for f in p] for p in fitness ]
    return fitness
    
def ParetoGA(candidates=[], args={}):
    logscale       = args.get("logscale",False)
    norm_evaluator = args.get("norm_evaluator",ec.evaluators.parallel_evaluation_mp)
    use_Pareto     = args.get('use_Pareto', False)
    fitness        = norm_evaluator(candidates, args)
    enable_d_elists= args.get("enable_d_elists",False) 
    #DB>> TRAP #1
    fitness = deNANificator(fitness)
    #TRAP #1 <<DB
    RecArXive(['P' for _ in fitness],fitness,candidates,args)
    return [ ec.emo.Pareto(p) for p in fitness ]    
    
def KrayzmanGA(candidates=[], args={}):
    logscale       = args.get("logscale",False)
    norm_evaluator = args.get("norm_evaluator",ec.evaluators.parallel_evaluation_mp)
    use_Pareto     = args.get('use_Pareto', False)
    adap_intervals = args.get('adap_int',0)
    krayzman_th    = args.get('krayzman_threshold', 0.05)
    normspace      = args.get('normspace', False)
    weight_bound   = args.get('weight_bound',None)
    hold_weights   = args.get('hold_weights', False)
    vectormarks    = args.get('vecmarks',None)
    vecvalues      = args.get('vecvalues',None)
    krayzman_log   = args.get("krayzman_log",None)
    if type(krayzman_log) is not str: krayzman_log = None
    enable_d_elists= args.get("enable_d_elists",False) 

    ### Run evaluation
    fitness        = norm_evaluator(candidates, args)
    
    
    #DB>> TRAP #1
    fitness = deNANificator(fitness)
    #TRAP #1 <<DB
        
    def mydot(fit,w):
        wfit = array([ f[KrayzmanGA.masking]*w[KrayzmanGA.masking] for f in fit])
        return sum(wfit, axis = 1)
    def resetMASKEDweights(fitness):
        KrayzmanGA.weights  = var(fitness,axis=0)
        KrayzmanGA.masking  = where( (isfinite(KrayzmanGA.weights)) * (KrayzmanGA.weights > 1e-9) )[0]
        # KrayzmanGA.weights[   isnan(KrayzmanGA.weights)    ] = mean(KrayzmanGA.weights)
        # KrayzmanGA.weights[where(KrayzmanGA.weights < 1e-9)] = mean(KrayzmanGA.weights)
        KrayzmanGA.weights[KrayzmanGA.masking]  = 1./KrayzmanGA.weights[KrayzmanGA.masking]
        KrayzmanGA.weights  = KrayzmanGA.weights.T
        KrayzmanGA.weights[KrayzmanGA.masking] /= \
            sum(KrayzmanGA.weights[KrayzmanGA.masking])\
            if normspace else\
             amax(KrayzmanGA.weights[KrayzmanGA.masking])
        # wfitness = dot(fitness[:,KrayzmanGA.masking],KrayzmanGA.weights[KrayzmanGA.masking])
        wfitness = mydot(fitness,KrayzmanGA.weights)
        return wfitness
    def computerMASKEDcorrelation(acfit,wacfit):
        corrs              = array([ corrcoef(f,wacfit)[0,1] if m in KrayzmanGA.masking else nan for m,f in enumerate(acfit.T) ])
        KrayzmanGA.masking = array([ i for i in KrayzmanGA.masking if isfinite(corrs[i])])
        with loglock:
            logging.debug(f"MASK:{corrs} {corrs.shape}")
            logging.debug(f"CORRS:{KrayzmanGA.masking} {KrayzmanGA.masking.shape}")
        m2m                = amin(corrs[KrayzmanGA.masking])/amax(corrs[KrayzmanGA.masking])
        return corrs,m2m
    def computerMASKEDfittnes(fitness,acfit=None):
        if acfit is None:
            testvar = var(fitness,axis=0)
            KrayzmanGA.masking = array([ i for i in KrayzmanGA.masking if isfinite(testvar[i]) and testvar[i] > 1e-9 ])
            #DB>>
            with loglock:
                logging.debug(f"VARIANCE:{testvar.tolist()}")
                logging.debug(f"MASSK   :{KrayzmanGA.masking} {KrayzmanGA.masking.shape}")
            #<<DB
            #return dot(fitness[:,KrayzmanGA.masking],KrayzmanGA.weights[KrayzmanGA.masking])
            return mydot(fitness,KrayzmanGA.weights)
        else:
            # return (
                # dot(fitness[:,KrayzmanGA.masking],KrayzmanGA.weights[KrayzmanGA.masking]),
                # dot(  acfit[:,KrayzmanGA.masking],KrayzmanGA.weights[KrayzmanGA.masking])
            # )
            return (
                mydot(fitness,KrayzmanGA.weights),
                mydot(  acfit,KrayzmanGA.weights)
            )
    
    
    
    fitness  = array(fitness)
        

    if not hasattr(KrayzmanGA,'adap_cnt'):KrayzmanGA.adap_cnt = 0
    if not hasattr(KrayzmanGA,'adap_fit'):KrayzmanGA.adap_fit = []
    if not hasattr(KrayzmanGA,'masking' ):KrayzmanGA.masking  = array([ i for i in range(fitness.shape[1]) ])
    if not hasattr(KrayzmanGA,'weights' ):
        with loglock:
            logging.info("Generate new weights")
        wfitness = resetMASKEDweights(fitness)
        if not krayzman_log is None:
            with open(krayzman_log,'a') as fd:
                fd.write("\t\"SET\":{\n\t\t\"VARIANCE\":"+json.dumps(var(fitness,axis=0).tolist())+\
                ",\n\t\t\"MASKS\":"+json.dumps(KrayzmanGA.masking.tolist())+\
                ",\n\t\t\"WEIGHTS\":"+json.dumps(KrayzmanGA.weights.tolist())+"\n\t},\n" )
        RecArXive(wfitness.tolist(),fitness.tolist(),candidates,args)
        if enable_d_elists:
            args['_ec']._kwargs['num_elites'] = args.get('dynamics_elites',0)
        return wfitness.tolist()

    if enable_d_elists:
        args['_ec']._kwargs['num_elites'] = args.get('dynamics_elites',0)

    #DB>> TRAP #2
    try:
        wfitness = computerMASKEDfittnes(fitness)
    except BaseException as e:
        logging.error(f"Cannot compute masked fitness! fitness={fitness}, mask={KrayzmanGA.masking}:{e}")
        raise RuntimeError(f"Cannot compute masked fitness! fitness={fitness}, mask={KrayzmanGA.masking}:{e}")
    try:
        if any(isnan(fitness)):
            fmax = amax(wfitness[~isnan(wfitness)])
            with loglock:
                logging.info("Fitness is nan. Skip adjustment")
            return [ fmax*1e3 if isnan(f) else f for f in wfitness ]
    except:
        logging,error(f"TRAP #2, we should NOT be here: fitness = {fitness}")
        raise RuntimeError(f"TRAP #2, we should NOT be here: fitness = {fitness}")
        exit(1)
    #TRAP #2 <<DB
    corrs = zeros(KrayzmanGA.weights.shape[0])
    KrayzmanGA.adap_fit += fitness.tolist()
    KrayzmanGA.adap_cnt += 1
    if KrayzmanGA.adap_cnt < adap_intervals:
        with loglock:
            logging.info( " > Fitness Acucmulation")
            logging.info(f" |-> Iteration    : {KrayzmanGA.adap_cnt} of {adap_intervals}")
            logging.info(f" |-> Fitness size : {len(KrayzmanGA.adap_fit)}")
        wfitness = computerMASKEDfittnes(fitness)
    else:
        finiteweights       = KrayzmanGA.weights[isfinite(KrayzmanGA.weights)]
        minweight           = amin(finiteweights[where(finiteweights > 0.)])/2
        KrayzmanGA.weights  = array([KrayzmanGA.weights[i] if i in KrayzmanGA.masking else (minweight*(0.5+random.rand()))  for i in range(fitness.shape[1]) ])
        KrayzmanGA.masking  = array([ i for i in range(fitness.shape[1]) ])
        acfit     = array(KrayzmanGA.adap_fit)
        wacfit    = computerMASKEDfittnes(acfit)
        corrs,m2m = computerMASKEDcorrelation(acfit,wacfit)
        old_m2m   = m2m*2.
        cnt,st    = 0, time.time()
        with loglock:
            logging.info( " > Weights adaptation")
            logging.info(f" |-> BEFORE: mincor={amin(corrs[KrayzmanGA.masking]):0.6g} maxcor={amax(corrs[KrayzmanGA.masking]):0.6g} old={old_m2m:0.6g} m/m={m2m:0.6g} cnt={cnt:03d}")
        if not krayzman_log is None:
            with open(krayzman_log,'a') as fd:
                fd.write("\t\"ADAPTATION\":{\n\t\t\"FITNESS\":"+json.dumps(acfit.tolist())+",\n")

        while m2m  < krayzman_th:
            if enable_d_elists:
                args['_ec']._kwargs['num_elites'] = 0

            # if any(corrs < 0.) and nonegcor:
                # ncorid          = where((corrs < 0.)*(KrayzmanGA.weights<1e3 ))[0]
                # if not krayzman_log is None:
                    # with open(krayzman_log,'a') as fd:
                        # fd.write("\t\"NEGATIVE\":{\t\t\"CORRELATION\":"+json.dumps(corrs.tolist())+",\n\t\t\"WEIGHTS\":"+json.dumps(KrayzmanGA.weights.tolist())+"\n\t},\n" )
                # KrayzmanGA.weights[ncorid] = KrayzmanGA.weights[ncorid]*(1+2./float(acfit.shape[1]))
            # else:
            if weight_bound is None:
                upwghid = KrayzmanGA.masking
                dnwghid = KrayzmanGA.masking
            else:
                upwghid         = KrayzmanGA.masking[where(KrayzmanGA.weights[KrayzmanGA.masking]>1/weight_bound)]
                dnwghid         = KrayzmanGA.masking[where(KrayzmanGA.weights[KrayzmanGA.masking]<weight_bound )]
            upmaxid,dnminid = argmax(corrs[upwghid]),argmin(corrs[dnwghid])
            if not krayzman_log is None:
                with open(krayzman_log,'a') as fd:
                    fd.write("\t\t\"ITERATION\":{"+\
                    " \n\t\t\t\"MIN COR\":"+json.dumps([float(corrs[dnwghid[dnminid]]),float(KrayzmanGA.weights[dnwghid[dnminid]]),int(dnminid),dnwghid.tolist()])+\
                    ",\n\t\t\t\"MAX COR\":"+json.dumps([float(corrs[upwghid[upmaxid]]),float(KrayzmanGA.weights[upwghid[upmaxid]]),int(upmaxid),upwghid.tolist()])+\
                    ",\n\t\t\t\"CORRELATION\":"+json.dumps(corrs.tolist())+
                    ",\n\t\t\t\"MASKS\":"+json.dumps(KrayzmanGA.masking.tolist()) +\
                    ",\n\t\t\t\"WEIGHTS\":"+json.dumps(KrayzmanGA.weights.tolist())+\
                    ",\n\t\t\t\"FITNESS\":"+json.dumps(wacfit.tolist()) )
            KrayzmanGA.weights[dnwghid[dnminid]] *= 1+2./float(acfit.shape[1])
            KrayzmanGA.weights[upwghid[upmaxid]] *= 1-1./float(acfit.shape[1])
            if not krayzman_log is None:
                with open(krayzman_log,'a') as fd:
                    fd.write(",\n\t\t\t\"UPDATE\":"+json.dumps(KrayzmanGA.weights.tolist())+"\n\t\t},\n" )
            #KrayzmanGA.weights /= amax(KrayzmanGA.weights)
            if not hold_weights:
                KrayzmanGA.weights[KrayzmanGA.masking] /= \
                    sum(KrayzmanGA.weights[KrayzmanGA.masking])\
                    if normspace else\
                    amax(KrayzmanGA.weights[KrayzmanGA.masking])
            
            wfitness,wacfit          = computerMASKEDfittnes(fitness,acfit)
            old_m2m                  = m2m
            corrs,m2m                = computerMASKEDcorrelation(acfit,wacfit)
            
            if cnt >= 299 or time.time()-st > 120:
                with loglock:
                    logging.info(" |-> RESET : Weights are not converging: regenerate new weights")
                KrayzmanGA.masking  = array([ i for i in range(fitness.shape[1]) ])
                wfitness            = resetMASKEDweights(acfit)
                corrs,m2m           = computerMASKEDcorrelation(acfit,wacfit)
                with loglock:
                    logging.info(f" |-> NEW  : mincor={amin(corrs[KrayzmanGA.masking]):0.6g} maxcor={amax(corrs[KrayzmanGA.masking]):0.6g} m/m={m2m:0.6g} cnt={cnt:03d} time={time.time()-st}s")
                KrayzmanGA.adap_fit = []
                KrayzmanGA.adap_cnt = 0
                RecArXive(wfitness.tolist(),fitness.tolist(),candidates,args)
                if not krayzman_log is None:
                    with open(krayzman_log,'a') as fd:
                        fd.write("\t\t\"RESET\":{"+\
                        " \n\t\t\t\"MASKS\":"+json.dumps(KrayzmanGA.masking.tolist())+\
                        ",\n\t\t\t\"WEIGHTS\":"+json.dumps(KrayzmanGA.weights.tolist())+"\n\t\t}\n\t},\n" )
                return wfitness.tolist()
            if old_m2m >= m2m: cnt += 1
            elif cnt   >  0  : cnt -= 1
        if hold_weights:
            with loglock:
                logging.info(f" |-> PRENORM: mincor={amin(corrs[KrayzmanGA.masking]):0.6g} maxcor={amax(corrs[KrayzmanGA.masking]):0.6g} old={old_m2m:0.6g} m/m={m2m:0.6g} cnt={cnt:03d} time={time.time()-st}s")
            KrayzmanGA.weights[KrayzmanGA.masking] /= \
                sum(KrayzmanGA.weights[KrayzmanGA.masking])\
                if normspace else\
                amax(KrayzmanGA.weights[KrayzmanGA.masking])
            wfitness,wacfit          = computerMASKEDfittnes(fitness,acfit)
            old_m2m                  = m2m
            corrs,m2m                = computerMASKEDcorrelation(acfit,wacfit)
        
        with loglock:
            logging.info(f" |-> AFTER: mincor={amin(corrs[KrayzmanGA.masking]):0.6g} maxcor={amax(corrs[KrayzmanGA.masking]):0.6g} old={old_m2m:0.6g} m/m={m2m:0.6g} cnt={cnt:03d} time={time.time()-st}s")

        if not krayzman_log is None:
            with open(krayzman_log,'a') as fd:
                fd.write("\t\t\"COMPLETE\":{\n\t\t\t\"CORRELATION\":"+json.dumps(corrs.tolist())+\
                ",\n\t\t\t\"MASKS\":"+json.dumps(KrayzmanGA.masking.tolist())+\
                ",\n\t\t\t\"WEIGHTS\":"+json.dumps(KrayzmanGA.weights.tolist())+"\n\t\t}\n\t},\n" )
        KrayzmanGA.adap_fit = []
        KrayzmanGA.adap_cnt = 0

    with loglock:
        logging.info( " > Current weights, correlation, [ fitness ], target")    
        for z,(m,V, w,v,f,c) in enumerate(zip(vectormarks,vecvalues,KrayzmanGA.weights,wfitness,fitness.T,corrs) ):
            tw,tc,tfmin,tfmean,tfmax = f"{w:0.6g}", f"{c:0.6g}", f"{amin(f):0.6g}", f"{mean(f):0.6g}", f"{amax(f):0.6g}"
            logging.info(f" \-> {m} : "+("ON " if z  in KrayzmanGA.masking else "OFF")+f" : {tw:<15s} , {tc:<15s} : [ {tfmin:<15s}, {tfmean:<15s}, {tfmax:<15s} ] : {V:0.6g}")
            logging.debug(f"  `-> {f.tolist()}")
            

    #DB>>
    # wjs = json.dumps([fitness.tolist(), weights.tolist(), wfitness.tolist()])
    # logging.debug( wjs )
    #<<DB
    RecArXive(wfitness.tolist(),fitness.tolist(),candidates,args)

    return wfitness.tolist()
    
#################################################
def MaxNormalization(candidates=[], args={}):
    
    logscale       = args.get("logscale",False)
    norm_evaluator = args.get("norm_evaluator",ec.evaluators.parallel_evaluation_mp)
    use_Pareto     = args.get('use_Pareto', False)
    adap_intervals = args.get('adap_int',0)
    vectormarks    = args.get('vecmarks',None)
    vecvalues      = args.get('vecvalues',None)
    fitness        = norm_evaluator(candidates, args)
    
    
    #DB>> TRAP #1
    fitness = deNANificator(fitness)    
    #TRAP #1 <<DB
    fitness  = array(fitness)

    if not hasattr(MaxNormalization,'adap_cnt'):MaxNormalization.adap_cnt = 0
    if not hasattr(MaxNormalization,'adap_fit'):MaxNormalization.adap_fit = []
    if not hasattr(MaxNormalization,'scalers' ):MaxNormalization.scalers  = ones(len(vectormarks))


    args['_ec']._kwargs['num_elites'] = args.get('dynamics_elites',0)

    #DB>> TRAP #2
    wfitness = sum(fitness/MaxNormalization.scalers, axis = 1)
    try:
        if any(isnan(fitness)):
            fmax = amax(wfitness[~isnan(wfitness)])
            logging.info("Fitness is nan. Skip adjustment")
            return [ fmax*1e3 if isnan(f) else f for f in wfitness ]
    except:
        logging,error(f"TRAP #2 in MaxNormalization, we should NOT be here: fitness = {fitness}")
        raise RuntimeError(f"TRAP #2 in MaxNormalization, we should NOT be here: fitness = {fitness}")
        exit(1)
    #TRAP #2 <<DB
    MaxNormalization.adap_fit += fitness.tolist()
    MaxNormalization.adap_cnt += 1
    if MaxNormalization.adap_cnt < adap_intervals:
        logging.info( " > Max scaling : Fitness Acucmulation")
        logging.info(f" |-> Iteration    : {MaxNormalization.adap_cnt} of {adap_intervals}")
        logging.info(f" |-> Fitness size : {len(MaxNormalization.adap_fit)}")
    else:
        MaxNormalization.scalers    = amax( array(MaxNormalization.adap_fit), axis = 0 )
        MaxNormalization.scalers[where(MaxNormalization.scalers<1e-12)] = 1.
        KrayzmanGA.adap_fit = []
        KrayzmanGA.adap_cnt = 0
        args['_ec']._kwargs['num_elites'] = 0

    wfitness = sum(fitness/MaxNormalization.scalers, axis = 1)
    #DB>> TRAP #3
    if any( isnan(wfitness) ):
        logging.error(f"TRAP #3 in MaxNormalization, Hit NaN on normalization {wfitness.tolist()}: scalers: {MaxNormalization.scalers.tolist()}")
        raise RuntimeError(f"TRAP #3 in MaxNormalization, Hit NaN on normalization {wfitness.tolist()}: scalers: {MaxNormalization.scalers.tolist()}")
        exit(1)
    #TRAP #3 << DB
    sfitness = fitness/MaxNormalization.scalers
    logging.info( " > Max scaling : Current scaler, [ scaled fitness ][ fitness ], target")
    if not vectormarks is None:
        if vecvalues is None:
            for m,w,f,s in zip(vectormarks,MaxNormalization.scalers,fitness.T,sfitness.T):
                tw,tfmin,tfmean,tfmax = f"{w:0.6g}", f"{amin(f):0.6g}", f"{mean(f):0.6g}", f"{amax(f):0.6g}"
                tsmin,tsmean,tsmax = f"{amin(s):0.6g}", f"{mean(s):0.6g}", f"{amax(s):0.6g}"
                logging.info(f" \-> {m} : {tw:<15s} : [ {tsmin:<15s}, {tsmean:<15s}, {tsmax:<15s} ] [ {tfmin:<15s}, {tfmean:<15s}, {tfmax:<15s} ]")
                logging.debug(f"  `-> {f.tolist()}")
        else:
            for m,V, w,f,s in zip(vectormarks,vecvalues,MaxNormalization.scalers,fitness.T,sfitness.T):
                tw,tfmin,tfmean,tfmax = f"{w:0.6g}", f"{amin(f):0.6g}", f"{mean(f):0.6g}", f"{amax(f):0.6g}"
                tsmin,tsmean,tsmax = f"{amin(s):0.6g}", f"{mean(s):0.6g}", f"{amax(s):0.6g}"
                logging.info(f" \-> {m} : {tw:<15s} : [ {tsmin:<15s}, {tsmean:<15s}, {tsmax:<15s} ] [ {tfmin:<15s}, {tfmean:<15s}, {tfmax:<15s} ] : {V:0.6g}")
                logging.debug(f"  `-> {f.tolist()}")
            
    else:
        if vecvalues is None:
            for w,f,s in zip(MaxNormalization.scalers,fitness.T,sfitness.T):
                tw,tfmin,tfmean,tfmax = f"{w:0.6g}", f"{amin(f):0.6g}", f"{mean(f):0.6g}", f"{amax(f):0.6g}"
                tsmin,tsmean,tsmax = f"{amin(s):0.6g}", f"{mean(s):0.6g}", f"{amax(s):0.6g}"
                logging.info(f" \->  {tw:<15s} : [ {tsmin:<15s}, {tsmean:<15s}, {tsmax:<15s} ] [ {tfmin:<15s}, {tfmean:<15s}, {tfmax:<15s} ]")
                logging.debug(f"  `-> {f.tolist()}")
        else:
            for V,w,f,s in zip(vecvalues,MaxNormalization.scalers,fitness.T,sfitness.T):
                tw,tv,tfmin,tfmean,tfmax = f"{w:0.6g}", f"{amin(f):0.6g}", f"{mean(f):0.6g}", f"{amax(f):0.6g}"
                tsmin,tsmean,tsmax = f"{amin(s):0.6g}", f"{mean(s):0.6g}", f"{amax(s):0.6g}"
                logging.info(f" \->  {tw:<15s} : [ {tsmin:<15s}, {tsmean:<15s}, {tsmax:<15s} ] [ {tfmin:<15s}, {tfmean:<15s}, {tfmax:<15s} ] : {V:0.6g}")
                logging.debug(f"  `-> {f.tolist()}")
    logging.info( " > Max scaling : Current scaled fitness")
    for f0,f1,f2,f3 in zip(wfitness[:-3:4],wfitness[1:-2:4],wfitness[2:-1:4],wfitness[3::4]):
        logging.info(f" \->  {f0:0.3f} {f1:0.3f} {f2:0.3f} {f3:0.3f}")

    #DB>>
    # wjs = json.dumps([fitness.tolist(), weights.tolist(), wfitness.tolist()])
    # logging.debug( wjs )
    #<<DB
    RecArXive(wfitness.tolist(),fitness.tolist(),candidates,args)

    return wfitness.tolist()
    

#################################################
    
ea = ec.emo.NSGA2(rnd()) if opt.algor[0] == 'N' else ec.GA(rnd())

ea.observer = af_repoter
ea.variator = [af_crossover, af_mutation]
ea.terminator = ec.terminators.evaluation_termination

if   opt.algor[0]=='N': algor = ParetoGA
elif opt.algor[0]=='M': algor = MaxNormalization
elif opt.algor[0]=='K': algor = KrayzmanGA
else:
    logging.error(f"Unknow algorithm {opt.algor}. Should be K, N, M, or P")
    raise BaseException(f"Unknow algorithm {opt.algor}. Should be K, N, M, or P")

    
final_pop = ea.evolve(generator          = af_generator,
                      init_pop           = opt.initpop,
                      algorithm          = opt.algor[0],
                      evaluator          = algor,
                      krayzman_log       = Kr_log,
                      # krayzman_nonegcor  = opt.nonegcor,
                      krayzman_threshold = opt.krthr,
                      adap_int           = numWadapPop,
                      normspace          = opt.norms,
                      weight_bound       = opt.boundKGA,
                      hold_weights       = opt.holdKGAweights,
                      mutation_rate      = opt.mrate,
                      adaptive_mutation_slope = opt.amslope,
                      vecmarks           = vm,
                      vecvalues          = vv,
                      norm_evaluator     = ec.evaluators.parallel_evaluation_mp,
                      use_Pareto         = opt.algor[0] == 'N',
                      mp_evaluator       = TCevaluator, 
                      mp_nprocs          = int(opt.nth),
                      pop_size           = opt.psz,
                      bounder            = af_bounder(param_ranges=param_nslh, logscale=opt.logs),
                      param_ranges       = param_nslh,
                      logscale           = opt.logs,
                      maximize           = False,
                      max_evaluations    = opt.ngn*(opt.psz+opt.elites),
                      max_generations    = opt.ngn,
                      num_elites         = opt.elites,
                      dynamics_elites    = opt.elites,
                      enable_d_elists    = opt.enableDelits,
                      checkpoint_file    = chpntrec,
                      checkpoint_print   = opt.nch,
                      checkpoint_count   = opt.dch,
                      arXive_file        = arXive,
                      expTarget          = args[0],
                      logpop             = opt.logpop,
                      logarx             = opt.logarx,
                      evalparams         = {
                        'mode'           : opt.emode,
                        'mask'           : opt.emask,
                        'prespike'       : opt.eleft,
                        'postspike'      : opt.erght,
                        'spikethreshold' : opt.ethsh,
                        'spikecount'     : opt.espc,
                        'collapse_tests' : opt.cdiff,
                        'spikeweight'    : opt.spwtgh,
                        'downsampler'    : downsampler,
                        'vpvsize'        : opt.vpvsize
                      })
                          
final_pop = ea.population
xpop = sorted(
 [ [ c.fitness.values if opt.algor[0] == 'N' else c.fitness,'P',af_ec2mod(c.candidate,param_nslh) if opt.logs else c.candidate] for c in final_pop  ]+\
 [ [ c.fitness.values if opt.algor[0] == 'N' else c.fitness,'A',af_ec2mod(c.candidate,param_nslh) if opt.logs else c.candidate] for c in ea.archive ] 
)

with open(finalrec,'w') as fd:
    fd.write('{\n')
    fd.write("\t\"markers\"   :" + json.dumps(list(vm))          +",\n")
    fd.write("\t\"bvalues\"   :" + json.dumps(vv)                +",\n")
    fd.write("\t\"parameters\":" + json.dumps(param_nslh)        +",\n")
    fd.write("\t\"version\"   :" + json.dumps(getversion())          +",\n")
    fd.write("\t\"target\"    :" + json.dumps(args[0])           +",\n")
    fd.write("\t\"evaluation\":" + json.dumps({
                        'mode'           : opt.emode,
                        'mask'           : opt.emask,
                        'prespike'       : opt.eleft,
                        'postspike'      : opt.erght,
                        'spikethreshold' : opt.ethsh,
                        'spikecount'     : opt.espc,
                        'collapse_tests' : opt.cdiff,
                        'spikeweight'    : opt.spwtgh,
                        'downsampler'    : downsampler,
                        'vpvsize'        : opt.vpvsize
                      })         +",\n")
    fd.write("\t\"cmd\"       :" + json.dumps(" ".join(sys.argv))+",\n")
    fd.write("\t\"final\":[\n")
    for f,s,p in xpop:
        fd.write("\t\t"+json.dumps({'fitness':f, 'sours':s, 'parameters':p})+",\n")
    fd.write("\t\tnull\n\t]\n}\n")

with open(arXive,'a') as fd:
    fd.write("\t\tnull\n\t]\n}\n")
if not Kr_log is None:
    with open(Kr_log,"a")as fd:
        fd.write("\tnull\n}\n")

logging.info("DONE")
print(fname+timestamp)
