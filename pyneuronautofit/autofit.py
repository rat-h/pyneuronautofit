import time,logging,sys,os,gzip
from numpy import *
from numpy import random as nprnd
#from inspyred.ec.variators.mutators   import mutator 
from inspyred.ec.variators.crossovers import crossover 
import copy, json, os

def af_mod2ec(modprms,param_ranges):
    ecprms = []
    for modp,(pname,pscale,lo,hi) in zip(modprms,param_ranges):
        if   pscale == 'lin': 
            ecprms.append( modp )
        elif pscale == 'log':
            ecprms.append( log10(modp) )
        elif pscale == 'con':
            modprms.append( modp )
        else :
            logger.error('unknown scalier {}'.format(pscale))
            raise 
    return ecprms        

def af_ec2mod(ecprms,param_ranges):
    modprms = []
    for ecp,(pname,pscale,lo,hi) in zip(ecprms,param_ranges):
        if   pscale == 'lin': 
            modprms.append( ecp )
        elif pscale == 'log':
            modprms.append( 10.**ecp )
        elif pscale == 'con':
            modprms.append( ecp )
        else :
            logger.error('unknown scalier {}'.format(pscale))
            raise 
    return modprms        

__getV__ = lambda v,ndic:eval(v,ndic) if type(v) is str else v

class af_bounder(object):
    """
    Non linear bounder.
    Callable class with bounds
    """
    def __init__(self, param_ranges = None, logscale = False ):
        self.__set_bounds_(param_ranges, logscale)
    
    def __set_bounds_(self,param_ranges = None, logscale = False ):
        self.param_ranges = param_ranges
        self.logscale = logscale
        if param_ranges is None:
            self.lower_bound  = None
            self.upper_bound  = None
        else:
            self.lower_bound  = []
            self.upper_bound  = []
            dl,dh = {},{}
            if self.logscale:
                for pname,pscale,lo,hi in param_ranges:
                    if   pscale == 'log':
                        lb,hb = \
                            log10(__getV__(lo,dl)),\
                            log10(__getV__(hi,dh))
                        dl[pname] = 10.**lb
                        dh[pname] = 10.**hb
                    elif pscale == 'lin' or pscale == 'con':
                        lb,hb = \
                            __getV__(lo,dl),\
                            __getV__(hi,dh)
                        dl[pname] = lb
                        dh[pname] = hb
                    else:
                        logging.error(f"Unknow parameter scaler {pscale}. Can support only `log`, `lin`, or `con` ")
                        raise RuntimeError(f"Unknow parameter scaler {pscale}. Can support only `log`, `lin`, or `con` ")
                    self.lower_bound.append( lb )
                    self.upper_bound.append( hb )
            else:
                for pname,pscale,lo,hi in param_ranges:
                    lb,hb = \
                        __getV__(lo,dl),\
                        __getV__(hi,dh)
                    self.lower_bound.append( lb )
                    self.upper_bound.append( hb )
                    dl[pname] = lb
                    dh[pname] = hb

    def __call__(self, candidate, args):
        #We can reset parameter ranges list
        if ("logscale"     in args and self.logscale     != args["logscale"]) or \
           ("param_ranges" in args and self.param_ranges != args["param_ranges"]):
               self.__set_bounds_(args["param_ranges"], args["logscale"])

        # The default would be to leave the candidate alone
        if self.param_ranges is None : return candidate
        else:
            bounded_candidate = candidate
            d = {}
            if self.logscale:
                for i, (c, (pname, pscale, lo, hi)) in enumerate(zip(candidate, self.param_ranges)):
                    lo = __getV__(lo,d)
                    hi = __getV__(hi,d)
                    if   pscale == 'lin':
                        bounded_candidate[i] = max( min(c, hi), lo)
                    elif pscale == 'log':
                        bounded_candidate[i] = log10( max( min(10.**c, hi), lo) )
                    elif pscale == 'con':
                        bounded_candidate[i] = lo
                    else :
                        logger.error('unknown scalier {}'.format(pscale))
                        raise 
                    # bounded_candidate[i] = max( min(c, hi), lo)
                    d[pname] = bounded_candidate[i]
            else:
                for i, (c, (pname, pscale, lo, hi)) in enumerate(zip(candidate, self.param_ranges)):
                    lo = __getV__(lo,d)
                    hi = __getV__(hi,d)
                    if   pscale == 'lin' or pscale == 'log':
                        bounded_candidate[i] = max( min(c, hi), lo)
                    elif pscale == 'con':
                        bounded_candidate[i] = lo
                    else :
                        logger.error('unknown scalier {}'.format(pscale))
                        raise RuntimeError(f'unknown scalier {pscale}')
                    # bounded_candidate[i] = max( min(c, hi), lo)
                    d[pname] = bounded_candidate[i]
                
            return bounded_candidate
    def get_upper_bounds(self,candidate, args):
        bound,d = [], {}
        for i, (c, (pname, pscale, lo, hi)) in enumerate(zip(candidate, self.param_ranges)):
            hi = __getV__(hi,d)
            lo = __getV__(lo,d)
            if   pscale == 'lin' or pscale == 'log': bound.append(hi)
            elif pscale == 'con'                   : bound.append(lo)  
            else :
                logger.error(f'unknown scalier {pscale}')
                raise RuntimeError(f'unknown scalier {pscale}')
            d[pname] = c
        return bound
    def get_lower_bounds(self,candidate, args):
        bound,d = [], {}
        for i, (c, (pname, pscale, lo, hi)) in enumerate(zip(candidate, self.param_ranges)):
            lo = __getV__(lo,d)
            if   pscale == 'lin' or pscale == 'log': bound.append(lo)
            elif pscale == 'con'                   : bound.append(lo)  
            else :
                logger.error(f'unknown scalier {pscale}')
                raise RuntimeError(f'unknown scalier {pscale}')
            d[pname] = c
        return bound

def generator_with_resolve_strings(prm_ranges,logscale):      
    ret,d = [],{}
    for pname,pscale,lo,hi in prm_ranges:
        # print(pname,pscale,lo,hi)
        lo = __getV__(lo,d)
        hi = __getV__(hi,d)
        #DB>>
        # print(pname,pscale,lo,hi,":",lo,hi,)
        #<<DB
        if logscale:
            x = nprnd.uniform(lo, hi) if pscale == "lin" else (nprnd.uniform(log10(lo), log10(hi)) if pscale == 'log' else lo)
            d[pname] = 10.**x if pscale == 'log' else x
        else:
            x = lo if pscale == 'con' else nprnd.uniform(lo, hi)
            d[pname] = x
        ret.append(x)
        #DB>>
        # print(pname,pscale,lo,hi,logscale,"=>", (log10(lo), log10(hi)) if logscale else (lo,hi), '=> x=',x)
        #<<DB
    # print(ret)
    return  ret

def af_generator(random, args):
    #in args they may be different (just in case)
    prm_ranges   = args['param_ranges']
    logscale     = args.get("logscale",False)
    init_pop     = args.get("init_pop",None)
    if not init_pop is None and type(init_pop) is str:# 
        if os.path.isfile(init_pop):
            if not hasattr(af_generator,"counter"):
                af_generator.counter = 0
                logging.info("READING FROM INIT POPULATION")
                logging.info(f" > file                             : {init_pop}")
            fname, fext = os.path.splitext(init_pop)
            if   fext == ".gz":
                fname, fext = os.path.splitext(fname)
                if fext == ".json":
                    logging.info(f"Reading GZIP {init_pop}")
                    with gzip.open(init_pop,'r') as fd:
                        try:
                            arx = json.load(fd)
                        except BaseException as e:
                            logging.error(f" > Cannot read {init_pop}:{e}")
                            arx = None
                else:
                    logging.error(f"{init_pop} has no json[.gz] extension")
                    arx = None
            elif fext == ".json":
                logging.info(f"Reading JSON {init_pop}")
                with open(init_pop)as fd:
                    try:
                        arx = json.load(fd)
                        
                    except BaseException as e:
                        logging.error(f" > Cannot read {init_pop}:{e}")
                        arx = None
            else:
                logging.error(f" > Unknown extension {fext} in {init_pop}")
                arx = None
            if not arx is None:
                if not "parameters" in arx:
                    logging.error(f" > There aren't parameters in the starter {init_pop}'")
                    prm = None
                else :
                    prm = [ n for n,_,_,_ in arx['parameters'] ]
                pop  = [
                    p['parameters']
                    for r in 'final records unique'.split() if r in arx
                    for p in arx[r] if not p is None
                ]
                if len(pop) > af_generator.counter and not prm is None:
                    vect = pop[af_generator.counter]
                    rvec = af_ec2mod( generator_with_resolve_strings(prm_ranges,logscale),prm_ranges )
                    vect = [
                        (vect[prm.index(n[0])] if n[0] in prm else rvec[ni])
                        if type(n) is tuple or type(n) is list else 
                        (vect[prm.index(n)] if n in prm else rvec[ni])
                        for ni,(n,s,l,h) in enumerate(prm_ranges)
                    ]
                    logging.info(f" > a vector #{af_generator.counter:03d} has been read")
                    af_generator.counter += 1 
                    return af_mod2ec(vect,prm_ranges)

        else:
            logging.info(f" > Cannot read {init_pop} - file doesn't exist")
            sys.stderr.write(f" > Cannot read {init_pop}  - file doesn't exist\n")
            
    # if logscale:
        # return [nprnd.uniform(lo, hi) if pscale == "lin" else (nprnd.uniform(log10(lo), log10(hi)) if pscale == 'log' else lo) for pname,pscale,lo,hi in prm_ranges]
    # else:
        # return [lo if pscale == 'con' else nprnd.uniform(lo, hi)                                                               for pname,pscale,lo,hi in prm_ranges]
    
    return  generator_with_resolve_strings(prm_ranges,logscale)


@crossover
def af_crossover(random, mom, dad, args):
    crossover_rate = args.setdefault('crossover_rate', 1.0)
    num_crossover_points = args.setdefault('num_crossover_points', None)
    binary2continuous_ratio = args.setdefault('binary2continuous_ratio', 0.25)
    if num_crossover_points is None:
        num_crossover_points = random.randint(1,min(len(mom),len(dad))//2)
    children = []
    if random.random() < crossover_rate:
        bro = copy.copy(dad)
        sis = copy.copy(mom)
        if random.random() < binary2continuous_ratio:
            num_cuts = min(len(mom)-1, num_crossover_points)
            cut_points = random.sample(range(1, len(mom)), num_cuts)
            cut_points.sort()        
            normal = True
            for i, (m, d) in enumerate(zip(mom, dad)):
                if i in cut_points:
                    normal = not normal
                if not normal:
                    bro[i] = m
                    sis[i] = d
                    #normal = not normal
        else:
            for i, (m, d) in enumerate(zip(mom, dad)):
                bro[i] = d + (m-d)*random.random()
                sis[i] = d + (m-d)*random.random()
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children


def af_mutation(random, candidates, args):
    bounder       = args['_ec'].bounder
    prm_ranges    = args['param_ranges']
    adapt_mut_rt  = args.get('adaptive_mutation_rate', 0.9 )
    adapt_mut_sl  = args.get('adaptive_mutation_slope',0.06)
    mutation_rate = args.get('mutation_rate', 0.1)
    population    = [ p.candidate for p in args["_ec"].population ]
    aggregate     = candidates+population
    scalers       = [
        abs(array(bounder.get_upper_bounds(c,args)) - array(bounder.get_lower_bounds(c,args)) )
        for c in aggregate ]
    mutants       = []
    for i, (cs,css) in enumerate(zip(candidates,scalers[:len(candidates)]) ):
        mindist  = array( [ sqrt( sum( (array(cs) - array(ts))**2/css/tss) ) for ts,tss in zip(aggregate[i+1:],scalers[i+1:]) ] )/sqrt(float(len(cs)))
        mindist  = amin(mindist)
        mut_rate = mutation_rate + adapt_mut_rt * exp(-mindist/adapt_mut_sl)
        mutates  = nprnd.random(len(cs)) < mut_rate
        rnds     = nprnd.random(len(cs))
        mutant   = copy.copy(cs)
        for i, m in enumerate(cs):
            if mutates[i]:
                mutant[i] = bounder.lower_bound[i] + (bounder.upper_bound[i]-bounder.lower_bound[i])*rnds[i]
        mutants.append(mutant)
    return mutants

            
def af_repoter(population, num_generations, num_evaluations, args):
    param_ranges = args['param_ranges']
    # mymethod     = args['algorithm'][0] != "N"
    mymethod     = not args.get('use_Pareto',False)
    logscale     = args.get("logscale",False)
    logpop       = args.get("logpop",False)
    logarx       = args.get("logarx",False)
    logbest      = args.get("logbest",False)
    reverse      = args.get("maximize",False)
    logging.info( '---------------------------------------------------')
    logging.info(f' > Generation #{num_generations} of {args["max_generations"]}')
    if logpop:
        xpop = sorted([ [c.fitness if mymethod else c.fitness.values,af_ec2mod(c.candidate,param_ranges) if logscale else c.candidate] for c in population ],reverse = reverse)
        logging.info(f" P:{json.dumps(xpop)}")
    if logarx:
        xpop = sorted([ [c.fitness if mymethod else c.fitness.values,af_ec2mod(c.candidate,param_ranges) if logscale else c.candidate] for c in args["_ec"].archive ],reverse = reverse)
        logging.info(f" A:{json.dumps(xpop)}")
    if logbest:
        xpop = sorted([ [c.fitness if mymethod else c.fitness.values,af_ec2mod(c.candidate,param_ranges) if logscale else c.candidate] for c in population ],reverse = reverse)
        logging.info("=== Topmodels ===")
        logging.info(f" 1:{json.dumps(xpop[0])}")
        logging.info(f" 2:{json.dumps(xpop[1])}")
        logging.info(f" 3:{json.dumps(xpop[2])}")

# def af_archiver(random, population, archive, args):
    # logscale = args.get("logscale",False)
    # param_ranges = args['param_ranges']
    # if "checkpoint_file" in args:
        # xpop = sorted([ [c.fitness,":",af_ec2mod(c.candidate,param_ranges) if logscale else c.candidate] for c in population ]+\
                      # [ [c.fitness,":",af_ec2mod(c.candidate,param_ranges) if logscale else c.candidate] for c in archive ] )
        # with open(args["checkpoint_file"],"w") as fd:
            # json.dump(xpop,fd)
    # if "archive_file" in args:
        # xpop = sorted([ [c.fitness,":",af_ec2mod(c.candidate,param_ranges) if logscale else c.candidate] for c in population ])
        # with open(args["archive_file"],"a") as fd:
            # fd.write(json.dumps(xpop)+"\n")
    # return archive
