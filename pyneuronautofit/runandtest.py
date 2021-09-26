import sys, logging, os
from numpy import *
sys.path.append('.')
try:
    from .evaluator import Evaluator
except:
    from pyneuronautofit import Evaluator


from project import fitmeneuron, param_nslh, simulator
if   simulator == "neuron":
    from neuron import h
elif simulator == "brian2":
    import brian2 as br
    raise NotImplemented("Brian's runner isn't implemented yet")
else:
    raise RuntimeError(f"Unknown simulator {simulator}")

class RunAndTest():
    def __init__(self,evaluator:Evaluator,params:(dict,None)=None,celsius:(float,None)=None):
        self.eval      = evaluator
        self.params    = params        
        self.celsius   = celsius
        self.N         = len(self.eval.TestCurr)
        if simulator == "neuron":
            self.__sim_run__ = self.__nrn_run__
        
    @property
    def __run__(self): return self.__sim_run__
        

        
    def __nrn_SetPrm__(self, n:fitmeneuron, pname:(str,tuple), val:float)->None:
        if type(pname) is tuple:
            for pn in pname:
                exec(f"n.{pn} = {val}")
        else:
            exec(f"n.{pname} = {val}")
    
    def __nrn_run__(self, params:(dict,None)=None,view:(bool,int)=False)->list:
        logging.debug("Running evaluation")
        if params is None and self.params is None:
            logging.error(f"Parameters are not given")
            raise RuntimeError(f"Parameters are not given")
        elif params is None: params = self.params
        
        h.celsius = 36. if self.celsius is None else self.celsius
        
        pop   = [ fitmeneuron(nid=x+1) for x in range(self.N) ]
        stims = [
            [
                h.IClamp(0.5,sec=n.soma),
                h.Vector(self.eval.TestCurr[nid]*1e-3), # rec in pA and nrn in nA
                h.Vector(arange(self.eval.TestCurr[nid].shape[0])*self.eval.expdt )
            ] for nid,n in enumerate(pop)
        ]
            
        recs  = [ h.Vector() for n in pop ]
        
        for nid,(n,(ic,ival,itime),rc) in enumerate( zip(pop,stims,recs) ):
            for pname in params:
                self.__nrn_SetPrm__(n,pname,params[pname])
            pop[nid].setcable()
            ic.amp = self.eval.TestCurr[nid][0]*1e-3
            ic.delay = -1000.
            ic.dur   = 1e9
            itime.x[0] = -1000.
            ival.play(stims[nid][0]._ref_amp,stims[nid][2],1)
            rc.record(n.soma(0.5)._ref_v)
            
        h.celsius = 36. if self.celsius is None else self.celsius
        
        
        trec   = h.Vector()
        trec.record(h._ref_t)
        
    
        try:
            h.finitialize()
            h.fcurrent()
            h.frecord_init()
            h.t = -1000.   # 1000 ms for transient
        except BaseException as e :
            logging.error(f"STUCK IN PRERUN WITH PARAMETERS:{params}")
            logging.error(f"                      EXCEPTION:{e}")
            if view:raise
            else   :return self.eval.clone(None)
        
        try:    
            while h.t < self.eval.tmax :h.fadvance()
        except BaseException as e :
            logging.error(f"STUCK IN RUN WITH PARAMETERS:{params}")
            logging.error(f"                   EXCEPTION:{e}")
            if view:raise
            else   :return self.eval.clone(None)
    
    
    
        atrec    = copy(array(trec))
        zerot    = where(atrec > 0.)[0][0]-1
        atrec    = atrec[zerot:]
        arecs    = [ copy(array(v)[zerot:]) for v in recs ]
        if int(view) == 2:
            tscale  = int(round(self.eval.expdt/h.dt) )
            scldrecs= [ mean( v[:v.shape[0]-v.shape[0]%tscale].reshape((v.shape[0]//tscale,tscale)),axis=1) for v in arecs ]
            return self.eval.diff(self.eval.clone(scldrecs),marks=True),atrec,arecs
        if view: return atrec,arecs
        tscale  = int(round(self.eval.expdt/h.dt) )
        scldrecs= [ mean( v[:v.shape[0]-v.shape[0]%tscale].reshape((v.shape[0]//tscale,tscale)),axis=1) for v in arecs ]
        del pop,stims,recs,trec#,h,dLGN
        return self.eval.clone(scldrecs)

    def __call__(self, params=None)->list:
        return self.__run__(params=params,view=False) - self.eval
        
if __name__ == '__main__':
    import sys,os,importlib,gzip,json
    
    from optparse import OptionParser
    oprs = OptionParser("USAGE: %prog [flags] file_with_parameters")    
    oprs.add_option("-i", "--input",       dest="input", default=None,   help="input file should be abf,json,or npz. -v needs only abf",type="str")
    oprs.add_option("-M", "--Mode",        dest="mode",  default="TSUW", help="Mode for model evaluation (don't use)",                  type="str")
    oprs.add_option("-K", "--masK",        dest="mask",  default="{}",   help="mask for evaluation",                                    type="str")
    oprs.add_option("-T", "--Threshold",   dest="thrsh", default=0.,     help="spike threshold",                                        type="float")
    oprs.add_option("-L", "--Left",        dest="left",  default=10,     help="left sample for spike shape",                            type="int")
    oprs.add_option("-R", "--Right",       dest="rght",  default=20,     help="right sample for spike shape",                           type="int")
    oprs.add_option("-C", "--Count",       dest="count", default=-1,     help="number of spikes to analyze in spike shape and width",   type="int")
    
    oprs.add_option("-s", "--sort",        dest="sort",  default=False,  help="sort parameters file",                                   action="store_true")
    oprs.add_option("-n", "--neuron",      dest="nrnid", default=0,      help="neuron id from the top of the list",                     type="int")
    oprs.add_option("-t", "--temperature", dest="celsius", default=35.,  help="temperature in celsius",                                 type="float")
    
    oprs.add_option("-d", "--diff",        dest="diff",  default=False,  help="Print out differences",                                  action="store_true")
    oprs.add_option("-c", "--collapsed-diff",dest="cdiff",default=False, help="Print out collapsed differences",                        action="store_true")
    oprs.add_option("-v", "--view",        dest="view",  default=False,  help="Show graphs with voltages",                              action="store_true")
    oprs.add_option("-V", "--save-view",   dest="Gsave", default=None,   help="Save graphs in to a file instead of showing them",       type="str")
    oprs.add_option("-W", "--fig-size",    dest="Fsize", default=None,   help="The size of the figure in WxH format",                   type="str")
    oprs.add_option("-O", "--view-current",dest="showa", default=False,  help="Show figure with currents",                              action="store_true")
    oprs.add_option("-N", "--Number-records",dest="nrec", default=False,  help="Show number of records in archive and exit",            action="store_true")
    # oprs.add_option("-0", "--off-header",  dest="header",default=True,   help="Do not skip the header (for backward compatibility)",    action="store_false")
    oprs.add_option("-l", "--log-level"   ,        dest="ll",     default="INFO",          type='str',
        help="Level of logging.[CRITICAL, ERROR, WARNING, INFO, or DEBUG] (default INFO)") 
    oprs.add_option(     "--log-file"     ,       dest="lf",     default=None,           type='str',
        help="save log to file")

    oprs.add_option(      "--save-json",   dest="sjson", default=None,   help="save abf in json format",                                type="str")
    oprs.add_option(      "--save-numpy",  dest="snpz",  default=None,   help="save abf in numpy compressed format",                    type="str")
    oprs.add_option(      "--show-vector", dest="svec",  default=False,  help="print out vector and exit",                              action="store_true")
    opt, args = oprs.parse_args()
    
    if opt.lf is None:
        logging.basicConfig(format='%(asctime)s:%(name)-10s:%(lineno)-4d:%(levelname)-8s:%(message)s', level=eval("logging."+opt.ll) )
    else:
        logging.basicConfig(filename=opt.lf, format='%(asctime)s:%(name)-10s:%(lineno)-4d:%(levelname)-8s:%(message)s', level=eval("logging."+opt.ll) )
    
    try:
        opt.mask = eval(opt.mask)
    except BaseException as e :
        logging.error(f"INFO: Cannot evaluate condition mask {opt.mask}:{e}")
        logging.error(f"INFO: Treating {opt.mask} as filename")
        try:
            with open(opt.mask) as fd:
                mask = fd.read()
            opt.mask = eval(mask)
        except BaseException as e :
            logging.error(f"ERROR: Cannot read condition mask from the {opt.mask}:{e}")
            logging.error( "ERROR: ====== !!! FULL STOP !!! ======")
            raise BaseException(f"Cannot read condition mask from the {opt.mask}:{e}")




    if len(args) != 1 and opt.sjson is None and opt.snpz is None and not opt.svec:
        raise BaseException(f"Need ONE file with parameters for the neuron")
    if opt.input is None and not opt.nrec:
        raise BaseException(f"Need npz, json, or abf to compare")
    evaluator = Evaluator(opt.input,mode=opt.mode,mask=opt.mask,prespike=opt.left,postspike=opt.rght,spikethreshold=opt.thrsh,spikecount=opt.count,\
                        collapse_tests=opt.cdiff, savetruedata=opt.view) 
    if len(args) != 1: exit(0)
    BestSols = []
    fname, fext = os.path.splitext(args[0])
    if fext == ".py":
        try:
            mod = importlib.import_module(fname)
            BestSols = mod.selected
        except BaseException as e :
            print(f"Cannot import selected from {args[0]}")
            raise
    elif fext == ".json" or fext == ".gzip" or fext == ".gz":
        try:
            if fext == ".json":
                with open(args[0]) as fd:
                    BestSols = json.load(fd)
            else:
                with gzip.open(args[0]) as fd:
                    BestSols = json.load(fd)
        except BaseException as e :
            print(f"Cannot read json/json.gzip from {args[0]}:{e}")
            raise
        
        if type(BestSols) is list:
            BestSols = [ xline for xline in BestSols if type(xline[-1]) is list ]
        elif type(BestSols) is dict:
            if "parameters" in BestSols:
                param_nslh = BestSols["parameters"]
            BestSols = [
                [ p['fitness'],p['parameters'] ]
                for r in 'final records unique'.split() if r in BestSols
                for p in BestSols[r] if not p is None
            ]
        else:
            print(f"Incorrect type of arXive")
            raise
    elif fext == ".npz":
        
        BestSols = load(args[0])["arXive"]
        BestSols = list(BestSols)
    else:
        print(f"Unsupported fileformat {fext} ({args[0]})")
        raise

    if opt.sort: BestSols = sorted(BestSols)
    if opt.nrnid >= len(BestSols):
        print(f"NID={opt.nrnid} out of range [0,{len(BestSols)}] of {args[0]}")
        raise RuntimeError(f"NID={opt.nrnid} out of range [0,{len(BestSols)}] of {args[0]}")
    p = BestSols[opt.nrnid][-1]
    
    if opt.nrec:
        print(len(BestSols))
        exit(0)
        
    if not opt.snpz is None:
        evaluator.exportNPZ(opt.snpz)
    if not opt.sjson is None:
        evaluator.exportJSON(opt.sjson)
    if opt.svec:
        ev = evaluator.vector()
        print( ev )
        print( len(ev) )
        exit(0)
    
    if not opt.Fsize is None:
        try:
            opt.Fsize = [int(x) for x in opt.Fsize.split('x') ]
            opt.Fsize = tuple(opt.Fsize)
        except BaseException as e :
            logging.error(f"ERROR: Cannot convert figure size {opt.Fsize}:{e}")
            logging.error( "ERROR: ====== !!! IGNORE !!! ======")
            opt.Fsize = None

    if type(p) is dict:
        prms = p
    else:
        prms = {}
        for (pname,l,m,M),v in zip (param_nslh,p):
            if type(pname) is tuple or type(pname) is list:
                for ppname in pname:
                    prms[ppname]=v
            else:
                prms[pname]=v
    if opt.view:
        from matplotlib.pyplot import *
        if opt.diff or opt.cdiff:
            vdiff,trec,recs = RunAndTest(evaluator,params=prms,celsius=opt.celsius).__run__(view=2)
        else:
            trec,recs = RunAndTest(evaluator,params=prms,celsius=opt.celsius).__run__(view=True)
        nh = len(recs)//4 + (1 if len(recs)%4 else 0)
        if opt.showa:
            f2=figure(2,figsize=(16,9))
            for cid,c in enumerate(evaluator.TestCurr):
                subplot(nh,4,cid+1)
                plot(arange(c.shape[0])*evaluator.expdt,c)

        if opt.Fsize is None:
            f1=figure(1,figsize=(16,9)) if opt.Gsave is None else figure(1,figsize=(64,36))
        else:
             f1=figure(1,figsize=opt.Fsize)
        if   opt.diff  :
            suptitle(f"N #{opt.nrnid}:"+\
            "\n"+" ".join([f"{m}:{x:0.2g}" for m,x in vdiff[:len(vdiff)//2] ])+\
            "\n"+" ".join([f"{m}:{x:0.2g}" for m,x in vdiff[len(vdiff)//2:] ]),
            fontsize=18)
        elif opt.cdiff :
            suptitle(f"N #{opt.nrnid}:\n"+" ".join([f"{m}:{x:0.2g}" for m,x in vdiff]),fontsize=18)
        else:
            suptitle(f"N #{opt.nrnid}",fontsize=18)
        for pltid,rec in enumerate(recs):
            subplot(nh,4,pltid+1)
            plot(trec,array(rec))
            plot(arange(evaluator.TrueData[pltid].shape[0])*evaluator.expdt,evaluator.TrueData[pltid])
        if opt.Gsave is None:
            show()
        else:
            f1.savefig(opt.Gsave)
        if opt.diff or opt.cdiff:
            print(f"{opt.nrnid},",",".join([f"\'{m}:\',{x}" for m,x in vdiff]))
    elif opt.diff or opt.cdiff:
        res = RunAndTest(evaluator,celsius=opt.celsius)(params=prms)
        print(res)
        print("len=",len(res))
    else:
        ev  = RunAndTest(evaluator,celsius=opt.celsius).__run__(params=prms)
        print(ev.vector())
        
        
