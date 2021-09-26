import sys,os,time,logging
from optparse import OptionParser
from random import Random as rnd
from numpy import *
from inspyred import ec   # evolutionary algorithm
sys.path.append('.')
from project import getversion, param_nslh

oprs = OptionParser("USAGE: %prog [flags] input_file_with_currents_and_target_stats (abf,npz,or json)")
#GA
oprs.add_option("-A", "--algorithm",           dest="algor", default="Krayzman",       type='str',
    help="Algorithm for multiobjective evaluation. It can be Krayzman - for Krayzman's fitness weighting; NSDA2 - for Pareto nondominate selection; and Max - for max scaled summation. Algorithm can be given by first letter K,N, or M correspondingly. (Default is K)")
oprs.add_option("-P", "--population-size",     dest="psz",    default=256,             type='int',
    help="population size (default 256). If it is a negative number: the population size is the length of the fitness vector multiple by absolute value of this option." )
oprs.add_option("-G", "--number-generation",   dest="ngn",    default=256,             type='int',
    help="number of generation (default 256)" )
oprs.add_option("-E", "--number-elites",       dest="elites", default=32,              type='int',
    help="number of elites in the replacement (default 32)" )
oprs.add_option("-L", "--off-log-scale",      dest="logs",    default=True,            action="store_false",
    help="enable log scaling")
oprs.add_option("-I", "--init-population",    dest="initpop", default=None,            type='str',
    help="file with a set of initial population" )
oprs.add_option("-N", "--no-negatives",       dest="nonegcor",default=False,           action="store_true",
    help="Adjust negative correlation first" )
oprs.add_option('-U', "--scales-update",      dest="update",default=10.,            type='float',
    help="vector length * this scale is number of fitness vectors before update Krayzman's weights or max scalers (default 10)") 

#MODEL
oprs.add_option("-m", "--eval-mode",          dest="emode",   default="RAMN",          type='str',
    help="mode for evaluation T-spike time, S-spike shape, U-subthreshould voltage dynamics, W-spike width, R - resting potential, L - post-stimulus tail, M - voltage stimulus statistics, A - average spike shape, N - number of spikes (default RAMN)")
oprs.add_option("-k", "--eval-mask",          dest="emask",   default="None",          type='str',
    help="mask to limit analysis")
oprs.add_option("-c", "--spike-count",        dest="espc",    default=2,               type='int',
    help="number of spikes for evaluation (2)")
oprs.add_option("-t", "--spike-threshold",    dest="ethsh",   default=0.,              type='float',
    help="spike threshold (default 0.)")
oprs.add_option("-l", "--left-spike-samples", dest="eleft",   default=110,             type='int',
    help="left window of spike (default 70)")
oprs.add_option("-r", "--right-spike-samples",dest="erght",   default=220,             type='int',
    help="right window of spike (default 140)")
oprs.add_option("-q", "--temperature",        dest="temp",    default=35.,             type='float',
    help="temperature (default 35)")
oprs.add_option("-e", "--collapse-diff",      dest="cdiff",   default=False,           action="store_true",
    help="Collapse difference between a model and data in a vector with size = number of tests (i.e. for  -m RAMNT the diff vector will be length 5)")
    
#RUN
oprs.add_option("-n", "--number-threads",      dest="nth",    default=os.cpu_count(),  type='int',
    help="number of threads (default None - autodetection)" )
oprs.add_option("-v", "--log-level"   ,        dest="ll",     default="INFO",          type='str',
    help="Level of logging.[CRITICAL, ERROR, WARNING, INFO, or DEBUG] (default INFO)") 
oprs.add_option("-u", "--log-to-screen",       dest="lc",     default=False,           action="store_true",
    help="log to screen")
oprs.add_option('-Z', "--Krayzman-debug",      dest="Krdb",   default=False,           action="store_true",
    help="enable debug dump for adaptation weight")
oprs.add_option("-p", "--printed-checkpoints", dest="nch",    default=-1,              type='int',
    help="print out checkpoints every # generation (do not print out if negative)" )
oprs.add_option("-d", "--dump-checkpoints",   dest="dch",     default=8,               type='int',
    help="dump out checkpoints into checkpoint file every # generation (do not dump out if negative, default 8)" )
oprs.add_option("-i", "--iteration",           dest="riter",  default=-1,              type='int',
    help="adds iteration number to the runs stamp" )
oprs.add_option("-a", "--run-stamp",           dest="rstemp", default=None,            type='str',
    help="Use this run stamp instead of generated" )
oprs.add_option(      "--slurm-id",            dest="slurmid", default=None,            type='int',
    help="Add SLURM ID into timestamp" )
oprs.add_option("-y", "--norm-space",          dest="norms",  default=False,           action="store_true",
    help="normalize space under the curve")

oprs.add_option(      "--log-population",      dest="logpop",  default=False,           action="store_true",
    help="record population into log file")
oprs.add_option(      "--log-archive",         dest="logarx",  default=False,           action="store_true",
    help="record archive into log file")

opt, args = oprs.parse_args()

timestamp  = "-v"+getversion()+time.strftime("-%Y%m%d-%H%M%S")
timestamp += "-{:07d}-{}".format( random.randint(1999999) if opt.slurmid is None else opt.slurmid,opt.emode )
timestamp += "-{}{}".format("L" if opt.logs else "F",opt.algor[0])
if opt.riter >= 0:
    timestamp += f"-I{opt.riter:03d}"

if len(args) != 1:
    logging.error("Need only one ABF, NPZ, or JSAON input file with currents and target states")
    raise BaseException("Need only one ABF, NPZ, or JSAON input file with currents and target states")

algorithms = {"K": "Krayzman\'s weighted adaptation", "N":"Pareto nondominate selection", "M":"max scaled sum" }
if not opt.algor[0] in algorithms:
    logging.error(f"Unknow algorithm {opt.algor}. Should be K, N, or M")
    raise BaseException(f"Unknow algorithm {opt.algor}. Should be K, N, or M")
        
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
    

# Test  evaluator
e = Evaluator(args[0],mode = opt.emode,mask = opt.emask,prespike = opt.eleft,postspike = opt.erght,spikethreshold = opt.ethsh,spikecount = opt.espc,collapse_tests = opt.cdiff)
vl = e.diff(e,marks=True)
vm = "".join([m for m,v in vl])
vv = e.scores()
vl = len(vl)
with open(arXive,'w') as fd:
    fd.write('{\n')
    fd.write("\t\"markers\"   :" + json.dumps(list(vm))          +",\n")
    fd.write("\t\"bvalues\"   :" + json.dumps(vv)                +",\n")
    fd.write("\t\"parameters\":" + json.dumps(param_nslh)        +",\n")
    fd.write("\t\"version\"   :" + json.dumps(getversion())          +",\n")
    fd.write("\t\"cmd\"       :" + json.dumps(" ".join(sys.argv))+",\n")
    fd.write(f"\t\"records\":[\n")
if not Kr_log is None:
    with open(Kr_log,"w")as fd:
        fd.write("{\n")
if   opt.psz <  0 : opt.psz = abs(opt.psz)*vl
elif opt.psz == 0 : opt.psz = vl

numWadapPop = int(round(opt.update*vl/(opt.psz-opt.elites))) if opt.algor[0] != "N" else 0

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
logging.info( 'GA:')
logging.info(f' > population size                  : {opt.psz}')
logging.info(f' > number of generation             : {opt.ngn}')
logging.info(f' > number of elites in replacement  : {opt.elites}')
logging.info(f' > Algorithm                        : {algorithms[opt.algor[0]]}')
logging.info(f' > adapt weights every              : {numWadapPop} generations')
if opt.algor[0] == 'K':
    logging.info(f' > mitigate negative correlation    : {opt.nonegcor}')
logging.info(f' > use log scale                    : {opt.logs}')
logging.info(f' > initial population               : {opt.initpop}')
logging.info( 'EVALUATION:')
logging.info(f' > evaluation mode                  : {opt.emode}')
logging.info(f' > evaluation mask                  : {opt.emask}')
logging.info(f' > number of spikes in evaluation   : {opt.espc}')
logging.info(f' > spike threshold                  : {opt.ethsh}')
logging.info(f' > left and right spike boundry     : {opt.eleft} : {opt.erght}')
logging.info(f' > model temperature                : {opt.temp} C')
logging.info(f' > collapse vector                  : {opt.cdiff}')
logging.info(f' > vector length                    : {vl}')
logging.info(f' > vector components                : {vm}')
for c in opt.emode:
    logging.info(f' \-> {c}                              : {len([ p for p in vm if p==c])}')
    
logging.info( 'RUN:')
logging.info(f' > number of threads                : {opt.nth}')
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

#DB>>
#exit(0)
#<<DB

def list2dict(lst,param_ranges):
    d = {}
    for p,(n,s,l,h) in zip(lst,param_ranges):
        d[n] = p
    return d



def TCevaluator(candidates, args):
    fitness    = []
    modmode    = args.get('mode', False)
    prm_ranges = args.get('param_ranges', param_nslh)
    targetFile = args["expTarget"]
    evalPram   = args.get("evalparams",{})
    evaluator  = Evaluator(targetFile,**evalPram)
    fitness    = [ RunAndTest(evaluator,celsius=opt.temp)(params=list2dict(af_ec2mod(p,prm_ranges) if opt.logs else p,prm_ranges)) for p in candidates]
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
        krayzma_log  = args.get("krayzman_log",None)
        if not krayzma_log is None and type(krayzma_log) is str and KrayzmanNormalization.adap_cnt == 0:
            with open(krayzma_log,"a")as fd:
                fd.write("\t\"UPDATE\":"+json.dumps(KrayzmanNormalization.weights.tolist())+",\n")
    
def KrayzmanNormalization(candidates=[], args={}):
    
   
    logscale       = args.get("logscale",False)
    norm_evaluator = args.get("norm_evaluator",ec.evaluators.parallel_evaluation_mp)
    use_Pareto     = args.get('use_Pareto', False)
    adap_intervals = args.get('adap_int',0)
    nonegcor       = args.get('krayzman_nonegcor', False)
    normspace      = args.get('krayzman_normspace', False)
    vectormarks    = args.get('vecmarks',None)
    vecvalues      = args.get('vecvalues',None)
    # db_adapt       = args.get('krayzman_dbadap',False)
    krayzma_log    = args.get("krayzman_log",None)
    fitness        = norm_evaluator(candidates, args)
    
    
    #DB>> TRAP #1
    noneid = []
    vsize  = -1
    for fi,f in enumerate(fitness):
        if f is None:
            noneid.append(fi)
            continue
        # if None in f:
            # noneid.append(fi)
            # continue
        if vsize < 0 : vsize = len(f)
        elif vsize != len(f):
            logging.info(f"Vector sizes aren't equal {vsize}!={len(f)} :{fi}")
            print(f"Vector sizes aren't equal {vsize}!={len(f)} :{fi}")
            exit(1)
    
    for nid in noneid:
        fitness[nid] = [nan for i in range(vsize) ]
    fmax = nanmax( array(fitness) )
    fitness =[ [fmax*1e3 if isnan(f) else f for f in p] for p in fitness ]
    #TRAP #1 <<DB
    if use_Pareto:
        RecArXive(['P' for _ in fitness],fitness,candidates,args)
        return [ ec.emo.Pareto(p) for p in fitness ]
        

    fitness  = array(fitness)

    if not hasattr(KrayzmanNormalization,'adap_cnt'):KrayzmanNormalization.adap_cnt = 0
    if not hasattr(KrayzmanNormalization,'adap_fit'):KrayzmanNormalization.adap_fit = []
    if not hasattr(KrayzmanNormalization,'weights' ):
        logging.info("Generate new weights")
        KrayzmanNormalization.weights  = var(fitness,axis=0)
        KrayzmanNormalization.weights[   isnan(KrayzmanNormalization.weights)    ] = mean(KrayzmanNormalization.weights)
        KrayzmanNormalization.weights[where(KrayzmanNormalization.weights < 1e-9)] = mean(KrayzmanNormalization.weights)
        KrayzmanNormalization.weights  = 1./KrayzmanNormalization.weights.T
        KrayzmanNormalization.weights /= sum(KrayzmanNormalization.weights) if normspace else amax(KrayzmanNormalization.weights)
        wfitness = dot(fitness,KrayzmanNormalization.weights)
        if not krayzma_log is None and type(krayzma_log) is str:
            with open(krayzma_log,'a') as fd:
                fd.write("\t\"SET\":{\n\t\t\"VARIANCE\":"+json.dumps(var(fitness,axis=0).tolist())+",\n\t\t\"WEIGHTS\":"+json.dumps(KrayzmanNormalization.weights.tolist())+"\n\t},\n" )
        RecArXive(wfitness.tolist(),fitness.tolist(),candidates,args)
        args['_ec']._kwargs['num_elites'] = 0
        return wfitness.tolist()

    args['_ec']._kwargs['num_elites'] = args.get('dynamics_elites',0)

    #DB>> TRAP #2
    wfitness = dot(fitness,KrayzmanNormalization.weights)
    try:
        if any(isnan(fitness)):
            fmax = amax(wfitness[~isnan(wfitness)])
            logging.info("Fitness is nan. Skip adjustment")
            return [ fmax*1e3 if isnan(f) else f for f in wfitness ]
    except:
        print(fitness)
        print(type(fitness))
        exit(1)
    #TRAP #2 <<DB
    corrs = zeros(KrayzmanNormalization.weights.shape[0])
    KrayzmanNormalization.adap_fit += fitness.tolist()
    KrayzmanNormalization.adap_cnt += 1
    if KrayzmanNormalization.adap_cnt < adap_intervals:
        logging.info( " > Fitness Acucmulation")
        logging.info(f" |-> Iteration    : {KrayzmanNormalization.adap_cnt} of {adap_intervals}")
        logging.info(f" |-> Fitness size : {len(KrayzmanNormalization.adap_fit)}")
    else:
        acfit    = array(KrayzmanNormalization.adap_fit)
        wacfit   = dot(acfit,KrayzmanNormalization.weights)
        corrs    = array([ corrcoef(f,wacfit)[0,1] for f in acfit.T ])
        m2m      = amin(corrs)/amax(corrs)
        old_m2m  = m2m*2.
        cnt,st   = 0, time.time()
        logging.info( " > Weights adaptation")
        logging.info(f" |-> BEFORE: mincor={amin(corrs):0.6g} maxcor={amax(corrs):0.6g} old={old_m2m:0.6g} m/m={m2m:0.6g} cnt={cnt:03d}")
        while m2m  < 0.3 and time.time()-st < 120:
            if any(corrs < 0.) and nonegcor:
                ncorid          = where((corrs < 0.)*(KrayzmanNormalization.weights<1e3 ))[0]
                if not krayzma_log is None:
                    with open(krayzma_log,'a') as fd:
                        fd.write("\t\"NEGATIVE\":{\t\t\"CORRELATION\":"+json.dumps(corrs.tolist())+",\n\t\t\"WEIGHTS\":"+json.dumps(KrayzmanNormalization.weights.tolist())+"\n\t},\n" )
                KrayzmanNormalization.weights[ncorid] = KrayzmanNormalization.weights[ncorid]*(1+2./float(acfit.shape[1]))
            else:
                upwghid         = where(KrayzmanNormalization.weights>1e-9)[0]
                dnwghid         = where(KrayzmanNormalization.weights<1e3 )[0]
                upmaxid,dnminid = argmax(corrs[upwghid]),argmin(corrs[dnwghid])
                if not krayzma_log is None:
                    with open(krayzma_log,'a') as fd:
                        fd.write("\t\"ADAPTATION\":{\n\t\t\"MIN\":"+json.dumps([dnwghid,dnminid])+",\n\t\t\"MAX\":"+json.dumps([upwghid,upmaxid])
                        +",\n\t\t\"CORRELATION\":"+json.dumps(corrs.tolist())+",\n\t\t\"WEIGHTS\":"+json.dumps(KrayzmanNormalization.weights.tolist())+"\n\t},\n" )
                KrayzmanNormalization.weights[dnwghid[dnminid]] *= 1+2./float(acfit.shape[1])
                KrayzmanNormalization.weights[upwghid[upmaxid]] *= 1-1./float(acfit.shape[1])
            #KrayzmanNormalization.weights /= amax(KrayzmanNormalization.weights)
            lwd           = where(KrayzmanNormalization.weights>1e-9)[0]
            KrayzmanNormalization.weights[lwd] /= sum(KrayzmanNormalization.weights[lwd]) if normspace else amax(KrayzmanNormalization.weights[lwd])
            wfitness      = dot(fitness,KrayzmanNormalization.weights)
            wacfit   = dot(acfit,KrayzmanNormalization.weights)
            corrs    = array([ corrcoef(f,wacfit)[0,1] for f in acfit.T ])
            old_m2m       = m2m
            m2m           = amin(corrs)/amax(corrs)
            if cnt >= 299:
                logging.info("Weights are not converging: regenerate new weights")
                KrayzmanNormalization.weights       = var(acfit,axis=0)
                KrayzmanNormalization.weights[   isnan(KrayzmanNormalization.weights)    ] = mean(KrayzmanNormalization.weights)
                KrayzmanNormalization.weights[where(KrayzmanNormalization.weights < 1e-9)] = mean(KrayzmanNormalization.weights)
                KrayzmanNormalization.weights       = 1./KrayzmanNormalization.weights.T
                lwd           = where(KrayzmanNormalization.weights>1e-9)[0]
                KrayzmanNormalization.weights[lwd] /= sum(KrayzmanNormalization.weights[lwd]) if normspace else amax(KrayzmanNormalization.weights[lwd])
                wfitness      = dot(fitness,KrayzmanNormalization.weights)
                logging.info(f" |-> NEW  : mincor={amin(corrs):0.6g} maxcor={amax(corrs):0.6g} old={old_m2m:0.6g} m/m={m2m:0.6g} cnt={cnt:03d}")
                KrayzmanNormalization.adap_fit = []
                KrayzmanNormalization.adap_cnt = 0
                RecArXive(wfitness.tolist(),fitness.tolist(),candidates,args)
                if not krayzma_log is None:
                    with open(krayzma_log,'a') as fd:
                        fd.write("\t\"RESET\":{\n\t\t\"CORRELATION\":"+json.dumps(corrs.tolist())+",\n\t\t\"WEIGHTS\":"+json.dumps(KrayzmanNormalization.weights.tolist())+"\n\t},\n" )
                return wfitness.tolist()
            if old_m2m >= m2m: cnt += 1
            elif cnt   >  0  : cnt -= 1
        logging.info(f" |-> AFTER: mincor={amin(corrs):0.6g} maxcor={amax(corrs):0.6g} old={old_m2m:0.6g} m/m={m2m:0.6g} cnt={cnt:03d}")
        KrayzmanNormalization.adap_fit = []
        KrayzmanNormalization.adap_cnt = 0
        args['_ec']._kwargs['num_elites'] = 0
        if not krayzma_log is None:
            with open(krayzma_log,'a') as fd:
                fd.write("\t\"COMPLETE\":{\n\t\t\"CORRELATION\":"+json.dumps(corrs.tolist())+",\n\t\t\"WEIGHTS\":"+json.dumps(KrayzmanNormalization.weights.tolist())+"\n\t},\n" )
    wfitness = dot(fitness,KrayzmanNormalization.weights)
    logging.info( " > Current weights, weighted fitness, correlation, [ fitness ], target")
    if not vectormarks is None:
        if vecvalues is None:
            for m,w,v,f,c in zip(vectormarks,KrayzmanNormalization.weights,wfitness,fitness.T,corrs):
                tw,tv,tc,tfmin,tfmean,tfmax = f"{w:0.6g}", f"{v:0.6g}", f"{c:0.6g}", f"{amin(f):0.6g}", f"{mean(f):0.6g}", f"{amax(f):0.6g}"
                logging.info(f" \-> {m} : {tw:<15s} , {tv:<15s}, {tc:<15s} : [ {tfmin:<15s}, {tfmean:<15s}, {tfmax:<15s} ]")
                logging.debug(f"  `-> {f.tolist()}")
        else:
            for m,V, w,v,f,c in zip(vectormarks,vecvalues,KrayzmanNormalization.weights,wfitness,fitness.T,corrs):
                tw,tv,tc,tfmin,tfmean,tfmax = f"{w:0.6g}", f"{v:0.6g}", f"{c:0.6g}", f"{amin(f):0.6g}", f"{mean(f):0.6g}", f"{amax(f):0.6g}"
                logging.info(f" \-> {m} : {tw:<15s} , {tv:<15s}, {tc:<15s} : [ {tfmin:<15s}, {tfmean:<15s}, {tfmax:<15s} ] : {V:0.6g}")
                logging.debug(f"  `-> {f.tolist()}")
            
    else:
        if vecvalues is None:
            for w,v,f,c in zip(KrayzmanNormalization.weights,wfitness,fitness.T,corrs):
                tw,tv,tc,tfmin,tfmean,tfmax = f"{w:0.6g}", f"{v:0.6g}", f"{c:0.6g}", f"{amin(f):0.6g}", f"{mean(f):0.6g}", f"{amax(f):0.6g}"
                logging.info(f" \->  {tw:<15s} , {tv:<15s}, {tc:<15s} : [ {tfmin:<15s}, {tfmean:<15s}, {tfmax:<15s} ]")
                logging.debug(f"  `-> {f.tolist()}")
        else:
            for V,w,v,f,c in zip(vecvalues,KrayzmanNormalization.weights,wfitness,fitness.T,corrs):
                tw,tv,tc,tfmin,tfmean,tfmax = f"{w:0.6g}", f"{v:0.6g}", f"{c:0.6g}", f"{amin(f):0.6g}", f"{mean(f):0.6g}", f"{amax(f):0.6g}"
                logging.info(f" \->  {tw:<15s} , {tv:<15s}, {tc:<15s} : [ {tfmin:<15s}, {tfmean:<15s}, {tfmax:<15s} ] : {V:0.6g}")
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
            logging.error(f"Vector sizes aren't equal {vsize}!={len(f)} :{fi}")
            print(f"Vector sizes aren't equal {vsize}!={len(f)} :{fi}")
            exit(1)
    if vsize != len(vectormarks) or vsize != len(vecvalues):
        logging.error(f"Vector size isn't equal  to Vector marks {vsize}!={len(vectormarks)} or Vector values {vsize}!={len(vecvalues)}")
        print(f"Vector size isn't equal  to Vector marks {vsize}!={len(vectormarks)} or Vector values {vsize}!={len(vecvalues)}")
        exit(1)
    for nid in noneid:
        fitness[nid] = [nan for i in range(vsize) ]    
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
        print(fitness)
        print(type(fitness))
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
        KrayzmanNormalization.adap_fit = []
        KrayzmanNormalization.adap_cnt = 0
        args['_ec']._kwargs['num_elites'] = 0

    wfitness = sum(fitness/MaxNormalization.scalers, axis = 1)
    #DB>> TRAP #3
    if any( isnan(wfitness) ):
        logging.error(f"Hit NaN on normalization {wfitness.tolist()}: scalers: {MaxNormalization.scalers.tolist()}")
        print(f"Hit NaN on normalization {wfitness.tolist()}: scalers: {MaxNormalization.scalers.tolist()}")
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
    
final_pop = ea.evolve(generator         = af_generator,
                      init_pop          = opt.initpop,
                      algorithm         = opt.algor[0],
                      evaluator         = MaxNormalization if opt.algor[0] == 'M' else KrayzmanNormalization,
                      krayzman_log      = Kr_log,
                      krayzman_nonegcor = opt.nonegcor,
                      adap_int          = numWadapPop,
                      normspace         = opt.norms,
                      vecmarks          = vm,
                      vecvalues         = vv,
                      norm_evaluator    = ec.evaluators.parallel_evaluation_mp,
                      use_Pareto        = opt.algor[0] == 'N',
                      mp_evaluator      = TCevaluator, 
                      mp_nprocs         = int(opt.nth),
                      pop_size          = opt.psz,
                      bounder           = af_bounder(param_ranges=param_nslh, logscale=opt.logs),
                      param_ranges      = param_nslh,
                      logscale          = opt.logs,
                      maximize          = False,
                      max_evaluations   = opt.ngn*opt.psz,
                      max_generations   = opt.ngn,
                      num_elites        = opt.elites,
                      dynamics_elites   = opt.elites,
                      checkpoint_file   = chpntrec,
                      checkpoint_print  = opt.nch,
                      checkpoint_count  = opt.dch,
                      arXive_file       = arXive,
                      expTarget         = args[0],
                      logpop            = opt.logpop,
                      logarx            = opt.logarx,
                      evalparams        = {
                        'mode'           : opt.emode,
                        'mask'           : opt.emask,
                        'prespike'       : opt.eleft,
                        'postspike'      : opt.erght,
                        'spikethreshold' : opt.ethsh,
                        'spikecount'     : opt.espc,
                        'collapse_tests' : opt.cdiff
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
