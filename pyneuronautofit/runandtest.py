import sys, logging, os, importlib, gzip, json
from multiprocessing import Lock
from numpy import *
sys.path.append('.')
try:
    from .evaluator import Evaluator
except:
    from pyneuronautofit import Evaluator


from project import fitmeneuron, param_nslh, simulator
try:
	from project import downsampler
except:
	downsampler = None

if   simulator == "neuron":
    from neuron import h
elif simulator == "brian2":
    import brian2 as br
    raise NotImplemented("Brian's runner isn't implemented yet")
else:
    raise RuntimeError(f"Unknown simulator {simulator}")

class RunAndTest():
    def __init__(self,evaluator:Evaluator,
        params:(dict,None)   = None,
        celsius:(float,None) = None,
        init:(dict,None)     = None,
        dt:(float,None)      = None,
        lock:(Lock,None)=None,logname:(str,None)=None,loglevel:str="INFO"):
        self.eval      = evaluator
        self.params    = params        
        self.celsius   = celsius
        self.init      = init
        self.N         = len(self.eval.TestCurr)
        self.dt        = dt
        self.lock      = lock
        self.logger    = "RanAndTest.log" if logname is None else (logname+f"-RunAndTest-{os.getpid():09d}.log")
        self.logger    = "threadlog/"+self.logger
        self.loglevel  = loglevel
        if simulator == "neuron":
            self.__sim_run__ = self.__nrn_run__
        
    @property
    def __run__(self): return self.__sim_run__
    
    def log_error(self,message):
        if not self.loglevel in "ERROR WARNING INFO DEBUG".split(): return
        if self.lock is None: self.logger.error(message)
        else:
            with self.lock:
                #self.logger.error(message)
                with open(self.logger,"a") as fd:
                    fd.write(message+"\n")
    def log_info(self,message):
        if not self.loglevel in "INFO DEBUG".split(): return
        if self.lock is None: self.logger.info(message)
        else:
            with self.lock:
                #self.logger.info(message)
                with open(self.logger,"a") as fd:
                    fd.write(message+"\n")
    def log_debug(self,message):
        if not self.loglevel == "DEBUG": return
        if self.lock is None: self.logger.debug(message)
        else:
            with self.lock:
                #self.logger.debug(message)
                with open(self.logger,"a") as fd:
                    fd.write(message+"\n")

        
    def __nrn_SetPrm__(self, n:fitmeneuron, pname:(str,tuple), val:float)->None:
        if type(pname) is tuple:
            for pn in pname:
                exec(f"n.{pn} = {val}")
        else:
            exec(f"n.{pname} = {val}")
    
    def __nrn_run__(self, params:(dict,None)=None, init:(dict,None)=None, view:(bool,int)=False)->list:
        self.log_debug("Running evaluation")
        if params is None and self.params is None:
            self.log_error(f"Parameters are not given")
            raise RuntimeError(f"Parameters are not given")
        elif params is None: params = self.params
        if init is None : init = self.init
        
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
            if init is not None:
                for var in init:
                    self.__nrn_SetPrm__(n,var,init[var ])
            
            pop[nid].setcable()
            ic.amp = self.eval.TestCurr[nid][0]*1e-3
            ic.delay = -1000.
            ic.dur   = 1e9
            itime.x[0] = -1000.
            ival.play(stims[nid][0]._ref_amp,stims[nid][2],1)
            rc.record(n.soma(0.5)._ref_v)
            
        h.celsius = 36. if self.celsius is None else self.celsius
        
        if self.dt is not None:
            if self.dt > 0.:
                h.dt = self.dt
            else:
                h.dt = self.eval.expdt*abs(self.dt)

#DB>>
        # dbn=pop[0]
        # names = [ n for n in params]+[ n for n in "soma.diam soma.nseg axon.nseg soma.eca".split() ]
        # for n in names:
            # if type(n) is tuple or type(n) is list:
                # for xn in n:
                    # v = eval("dbn."+xn)
                    # print(f"{xn:<45s} = {v}")
            # else:
                # v = eval("dbn."+n)
                # print(f"{n:<45s} = {v}")
        # print(f"celcius = {h.celsius}")
        # print(f"dt = {h.dt}")
#<<DB

        trec   = h.Vector()
        trec.record(h._ref_t)
        
    
        try:
            h.finitialize()
            h.fcurrent()
            h.frecord_init()
            h.t = -1000.   # 1000 ms for transient
        except BaseException as e :
            self.log_error(f"STUCK IN PRERUN WITH PARAMETERS:{params}")
            self.log_error(f"                      EXCEPTION:{e}")
            if view:raise
            else   :return self.eval.clone(None)
        
        try:    
            while h.t < self.eval.tmax :h.fadvance()
        except BaseException as e :
            self.log_error(f"STUCK IN RUN WITH PARAMETERS:{params}")
            self.log_error(f"                   EXCEPTION:{e}")
            if view:raise
            else   :return self.eval.clone(None)
    
    
    
        atrec    = copy(array(trec))
        zerot    = where(atrec > 0.)[0][0]-1
        atrec    = atrec[zerot:]
        arecs    = [ copy(array(v)[zerot:]) for v in recs ]

        tscale   = int(round(self.eval.expdt/h.dt) )
        scldrecs = [ mean( v[:v.shape[0]-v.shape[0]%tscale].reshape((v.shape[0]//tscale,tscale)),axis=1) for v in arecs ]

        if   int(view) == 2:
            return self.eval.diff(self.eval.clone(scldrecs),marks=True,nummark=True),atrec,arecs
        elif int(view) == 3:
            e = self.eval.clone(scldrecs)
            return self.eval.diff(e,marks=True),atrec,arecs,e,scldrecs,tscale
        elif view: 
            return atrec,arecs
        else:
            del pop,stims,recs,trec#,h,dLGN
            return self.eval.clone(scldrecs)

    def __call__(self, params=None)->list:
        return self.__run__(params=params,view=False) - self.eval

def ReadArXive(fname,selection):
    def read_arxive(fd):
        arx = json.load(fd)
        if type(arx) is dict:
            markers    = arx['markers']    if 'markers'    in arx else None
            bvalues    = arx['bvalues']    if 'bvalues'    in arx else None
            parameters = arx['parameters'] if 'parameters' in arx else None
            target     = arx['target']     if 'target'     in arx else None
            evaluation = arx['evaluation'] if 'evaluation' in arx else None
            version    = arx['version']    if 'version'    in arx else None
            cmd        = arx['cmd']        if 'cmd'        in arx else None

            arXive  = [
                    [ p['fitness'],p['parameters'] ]
                    for r in 'final records unique model models'.split() if r in arx
                    for p in arx[r] if not p is None
            ]
        else:
            sys.stderr.write(f"Wrong format of arXive {type(arx)}\n")
            exit(1)
        return arXive,markers,bvalues,parameters,version,cmd,target,evaluation
    
    _, fext = os.path.splitext(fname)
    logging.info( "=========================")
    if fext == ".py":
        try:
            mod = importlib.import_module(fname)
            collections = mod.selected
        except BaseException as e :
            print(f"Cannot import selected from {args[0]}")
            raise
    elif fext == ".gz":
        logging.info(f"Reading GZIP {fname}")
        with gzip.open(fname,'r') as fd:
            collections,markers,bvalues,parameters,\
            version,cmd,target,evaluation = read_arxive(fd)
    elif fext == ".json":
        logging.info(f"Reading JSON {fname}")
        with open(fname,'r') as fd:
            collections,markers,bvalues,parameters,\
            version,cmd,target,evaluation = read_arxive(fd)
    else:
        sys.stderr.write(f"Unknown input file extension {fext}")
        exit(1)
    if selection is not None:
        logging.info( "=========================")
        logging.info(f"Filtering collections by selection {selection}")
        if type(selection) is str:
            try:
                selection = eval(selection)
            except BaseException as e :
                logging.error(f" Cannot convert selection {selection} into a python object:{e}")
                logging.error( " ====== !!! FULL STOP !!! ======")
                exit(1)
        if   type(selection) is int: selection = [selection]
        elif type(selection) is tuple and len(selection) == 2:
            if selection[0] >= len(collections):
                logging.error(f" Left boundary of selection{selection[0]} is bigger than arXive size {len(collection)}")
                logging.error( " ====== !!! FULL STOP !!! ======")
                exit(1)
            selection = [ i for i in range(selection[0],len(collections)) if i <= selection[1]]
        elif type(selection) is tuple and len(selection) == 3:
            if selection[0] >= len(collections):
                logging.error(f" Left boundary of selection{selection[0]} is bigger than arXive size {len(collection)}")
                logging.error( " ====== !!! FULL STOP !!! ======")
                exit(1)
            selection = [ i for i in range(selection[0],len(collections),selection[1]) if i <= selection[2]]
        elif type(selection) is list:
            proselection = []
            for x in selection:
                if   type(x) is int  :proselection.append(x)
                elif type(x) is list :
                    if len(x) == 2  or len(x) == 3:
                        for y in range(*x):
                            proselection.append(y)
                    else:
                        sys.stderr.write(f"Range selection should have 2 or 3 numbers {len(x)} is given\n")
                        exit(1)
                else:
                    sys.stderr.write(f"Unknown input type of selection {x}\n")
                    exit(1)
            selection = proselection
        else:
            logging.error(f" incorrect selector or selector size {selection}")
            logging.error( " ====== !!! FULL STOP !!! ======")
            exit(1)
        logging.info(f"Selection = {selection}")
        collections = [ collections[i] for i in selection]
    logging.info( "==================== DONE")
    
    return collections,markers,bvalues,parameters,\
            version,cmd,target,evaluation,selection


if __name__ == '__main__':
    import sys,os,importlib,gzip,json
    
    from optparse import OptionParser
    oprs = OptionParser("USAGE: %prog [flags] file_with_parameters")    
    oprs.add_option("-i", "--input",       dest="input",   default=None,   help="input file should be abf,json,or npz. -v needs only abf",type="str")
    oprs.add_option("-M", "--Mode",        dest="mode",    default=None,   help="Mode for model evaluation (don't use)",                  type="str")
    oprs.add_option("-K", "--masK",        dest="mask",    default=None,   help="mask for evaluation",                                    type="str")
    oprs.add_option("-T", "--Threshold",   dest="thrsh",   default=None,   help="spike threshold",                                        type="float")
    oprs.add_option("-L", "--Left",        dest="left",    default=None,   help="left sample for spike shape",                            type="int")
    oprs.add_option("-R", "--Right",       dest="rght",    default=None,   help="right sample for spike shape",                           type="int")
    oprs.add_option("-C", "--Count",       dest="count",   default=None,   help="number of spikes to analyze in spike shape and width",   type="int")
    oprs.add_option("-Z", "--spike-Zoom",  dest="spwtgh",  default=None,                                                                  type="float",\
        help="if positive absolute weight of voltage diff during spike; if negative relataed scaler")
    oprs.add_option("-Q","--v-dvdt-size",  dest="vpvsize", default=None,                                                                  type='int',\
        help="v dv/dt histogram size")
    oprs.add_option("-t", "--temperature", dest="celsius", default=35.,    help="temperature in celsius",                                 type="float")

    oprs.add_option(      "--dt",          dest="simdt",   default=None,                                                                  type="float",\
        help="if positive absolute simulation dt; if negative scaler for recorded dt")

    oprs.add_option("-s", "--sort",        dest="sort",    default=False,  help="sort parameters fist (it changes neurons IDs!)",         action="store_true")
    oprs.add_option("-n", "--neuron-ids",  dest="nrnid",   default=None,                                                                  type="str",\
        help="Neuron selector: if int-select neuron in the file; if tuple(int,int) - from,to selection; if tuple(int,int,int) - from,step,to selection; if list [int,....] -list of selected neurons. All ID from the top of the list.")
    oprs.add_option("-N", "--Num-threads", dest="nthrs",   default=os.cpu_count(),                                                        type="int",\
        help="Number of thread avalible for process. If not set all will be used. Set to 0 to stop multithreading")
    
    oprs.add_option("-d", "--diff",        dest="diff",  default=False,  help="Print out differences",                                  action="store_true")
    oprs.add_option("-c", "--collapsed-diff",dest="cdiff",default=False, help="Print out collapsed differences",                        action="store_true")
    oprs.add_option("-v", "--view",        dest="view",  default=False,  help="Show graphs with voltages",                              action="store_true")
    oprs.add_option("-V", "--save-view",   dest="Gsave", default=None,   help="Save graphs in to a file instead of showing them",       type="str")
    oprs.add_option("-W", "--fig-size",    dest="Fsize", default=None,   help="The size of the figure in WxH format",                   type="str")
    oprs.add_option("-O", "--view-current",dest="showa", default=False,  help="Show figure with currents",                              action="store_true")
    oprs.add_option("-B", "--numBer-records",dest="nrec", default=False,  help="Show number of records in archive and exit",            action="store_true")
    oprs.add_option(      "--save-json",   dest="sjson", default=None,   help="save abf in json format",                                type="str")
    oprs.add_option(      "--save-numpy",  dest="snpz",  default=None,   help="save abf in numpy compressed format",                    type="str")
    oprs.add_option(      "--show-vector", dest="svec",  default=False,  help="print out vector and exit",                              action="store_true")

    oprs.add_option("-l", "--log-level"   ,        dest="ll",     default="INFO",          type='str',
        help="Level of logging.[CRITICAL, ERROR, WARNING, INFO, or DEBUG] (default INFO)") 
    oprs.add_option(     "--log-file"     ,       dest="lf",     default=None,           type='str',
        help="save log to file")

    opt, args = oprs.parse_args()
    

    if opt.lf is None:
        logging.basicConfig(format='%(asctime)s:%(name)-10s:%(lineno)-4d:%(levelname)-8s:%(message)s', level=eval("logging."+opt.ll) )
    else:
        logging.basicConfig(filename=opt.lf, format='%(asctime)s:%(name)-10s:%(lineno)-4d:%(levelname)-8s:%(message)s', level=eval("logging."+opt.ll) )

       
    if len(args) != 1 and opt.sjson is None and opt.snpz is None and not opt.svec:
        logging.error(f" Need a json[.gz] file with arXive")
        logging.error( " ====== !!! FULL STOP !!! ======")
        exit(1)
    
    logging.info( "=========================")
    logging.info(f" Number of active threads : {opt.nthrs}")
    
    collections,markers,bvalues,parameters,version,\
        cmd,target,evaluation,selection = ReadArXive(args[0],opt.nrnid)

    ### Altering Parameters and Evaluation if it's needed >>>
    if parameters is not None: prm_nslh = parameters
    if opt.input is not None: target = opt.input
    if evaluation is None: evaluation = {}
    if opt.mode is not None:
        evaluation['mode'] = opt.mode
    if opt.mask is not None:
        try:
            evaluation['mask'] = eval(opt.mask)
        except BaseException as e :
            logging.info(f"Cannot evaluate condition mask {opt.mask}:{e}")
            logging.info(f"Treating {opt.mask} as filename")
            try:
                with open(opt.mask) as fd:
                    evaluation['mask'] = eval(fd.read())
            except BaseException as e :
                logging.error(f" Cannot read condition mask from the {opt.mask}:{e}")
                logging.error( " ====== !!! FULL STOP !!! ======")
                exit(1)
    if opt.thrsh is not None:
        evaluation['spikethreshold'] = opt.thrsh
    if opt.left is not None:
        evaluation['prespike'] = opt.left
    if opt.rght is not None:
        evaluation['postspike'] = opt.rght
    if opt.count is not None:
        evaluation['spikecount'] = opt.count
    if opt.cdiff: evaluation['collapse_tests'] = opt.cdiff    
    if opt.spwtgh is not None:
        evaluation['spikeweight'] = opt.spwtgh
    if opt.vpvsize is not None:
        evaluation['vpvsize'] = opt.vpvsize
    ### <<<--------------------------------------------------

    logging.info(f"Target file           :{target}")
    logging.info(f"Evaluation parameters :{evaluation}")

    if target is None:
        logging.error(f" Target file is not present in arXive and was not set by -i option")
        logging.error( " ====== !!! FULL STOP !!! ======")
        exit(1)

    evaluation['downsampler']=downsampler
    evaluator = Evaluator(target, savetruedata=opt.view, **evaluation) 
    if opt.sjson is not None or opt.snpz is not None:
        logging.infor( " ====== ================= ======")
        if opt.sjson is not None:
            logging.info(f" Exporting JSON into {opt.sjson}")
            evaluator.exportJSON(opt.sjson)
        if opt.snpz is not None:
            logging.info(f" Exporting NumPy archive into {opt.snpz}")
            evaluator.exportNPZ(opt.snpz)
        logging.infor( " ====== ===== DONE ====== ======")
        exit(0)    
    if opt.svec:
        print(evaluator.vector())
        exit(0)
    if prm_nslh is None:
        logging.error(f" Parameter ranges are not given in project.py or arXive")
        logging.error( " ====== !!! FULL STOP !!! ======")
        exit(1)


    ### Run Evaluation in parallel >>>
    import multiprocessing as mp
    def worker(p):
        if   type(p) is list:
            prm = {}
            for (n,s,l,h),v in zip(prm_nslh,p):
                if type(n) is list or type(n) is tuple:
                    for r in n:
                        prm[r] = v
                else:
                    prm[n]=v
        elif type(p) is dict:
            prm = p
        else:
            raise RuntimeError(f"Unsupported parameters type {type(p)}. It should be list or dictionary")
            exit(1)
        if opt.view:
            if opt.diff or opt.cdiff:
                fitness = RunAndTest(evaluator,celsius=opt.celsius,dt=opt.simdt, params=prm).__run__(view=2)
            else:
                fitness = RunAndTest(evaluator,celsius=opt.celsius,dt=opt.simdt, params=prm).__run__(view=opt.view)
        else:
            fitness = RunAndTest(evaluator,celsius=opt.celsius,dt=opt.simdt)(params=prm)
        return fitness
    if opt.nthrs > 0:
        pool = mp.Pool(processes=opt.nthrs)
        result = [pool.apply_async(worker,[p]) for _,p in collections]
        pool.close()
        pool.join()
        result = [r.get() for r in result]
    else:
        result = [ worker(p) for p in collections]
    ### <<<---------------------------


    if opt.view:
        from matplotlib.pyplot import *
        def keypass(event):
                if   event.key == "down"      : keypass.recid -= 1
                elif event.key == "up"        : keypass.recid += 1
                elif event.key == "home"      : keypass.recid  = 0
                elif event.key == "end"       : keypass.recid  = keypass.recid = len(result) - 1
                # elif event.key == "enter"      : 
                    # cand = candidates[keypass.recid] if modmode else af_ec2mod(candidates[keypass.recid],prm_ranges)
                    # print("Model # ",keypass.recid)
                    # print("Fitness ",fitness[keypass.recid])
                    # for p,(pname,pscale,lo,hi) in zip(cand,param_ranges):
                        # print(" > {:<33s}:{:g}".format("/".join(pname) if type(pname) is tuple else pname,p))
                    # print()
                    # return

                if keypass.recid < 0            : keypass.recid = 0
                if keypass.recid >= len(result) : keypass.recid = len(result) - 1
                
                if opt.diff or opt.cdiff:
                    vdiff,trec,recs = result[keypass.recid]
                    print(keypass.recid,vdiff)
                    t=arange(len(vdiff))
                    vdiff = array([v for m,v in vdiff ])
                    yline.set_xdata(t[where(vdiff > 0)])
                    yline.set_ydata(vdiff[where(vdiff > 0.)])

                else:
                    trec,recs = result[keypass.recid]
                    print(keypass.recid)
                suptit.set_text(f"N #{selection[keypass.recid]}")
                for xline,md in zip(xlines,recs):
                    xline.set_ydata(md)
                f1.canvas.draw()
                

        
        if opt.Fsize is not None:
            try:
                opt.Fsize = eval(opt.Fsize)
            except BaseException as e :
                logging.error(f" Cannot convert figure size {opt.Fsize} into python object:{e}")
                exit(1)
        
        nh =  len(evaluator.TestCurr)   //4 + (1 if len(evaluator.TestCurr)%4 else 0)
        if opt.showa:
            if opt.Fsize is None:
                f2=figure(2,figsize=(16,9)) if opt.Gsave is None else figure(2,figsize=(64,36))
            else:
                f2=figure(2,figsize=opt.Fsize)
            for cid,c in enumerate(evaluator.TestCurr):
                subplot(nh,4,cid+1)
                plot(arange(c.shape[0])*evaluator.expdt,c)
        if opt.Fsize is None:
            f1=figure(1,figsize=(16,9)) if opt.Gsave is None else figure(1,figsize=(64,36))
        else:
             f1=figure(1,figsize=opt.Fsize)
        
        suptit = suptitle(f"N #{selection[0]}",fontsize=18)

        if opt.diff or opt.cdiff:
            subplots = [ subplot2grid( (nh,7),(cid//4,cid%4) ) for cid,c in enumerate(evaluator.TestCurr) ]
        else:
            subplots = [      subplot(    nh,4,      cid+1     ) for cid,c in enumerate(evaluator.TestCurr) ]
        
        
        for sp,rec in zip(subplots,evaluator.TrueData):
            sp.plot(arange(rec.shape[0])*evaluator.expdt,rec)

        xlines = []
        if opt.diff or opt.cdiff:
            vdiff,trec,recs = result[0]
        else:
            trec,recs = result[0]

        for sp,rec in zip(subplots,recs):
            l, = sp.plot(trec,array(rec))
            xlines.append(l)
        keypass.recid = 0
        if opt.diff or opt.cdiff:
            adiff = array([ [ v for m,v in d ] for d,_,_ in result ])
            difmin = amin(adiff[where(adiff > 0)])
            difmax = amax(adiff[where(adiff > 0)])
            saxis = subplot2grid( (nh,7),(0,4), rowspan=nh, colspan=3 )
            t=arange(len(vdiff))
            adiff = array([v for m,v in vdiff ])
            yline, = saxis.semilogy(t[where(adiff > 0)],adiff[where(adiff > 0.)],"ko")
            saxis.set_ylim(difmin/2,difmax*2)
            saxis.set_xticks(t)
            saxis.set_xticklabels( [m for m,v in vdiff] )
            setp(saxis.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        
        
                
        if opt.Gsave is None:
            f1.canvas.mpl_connect('key_press_event', keypass)
            show()
        else:
            class Ek():
                def __init__(self, key):
                    self.key = key
            keypass(Ek("home"))
            for i in range(len(result)):
                f1.savefig(opt.Gsave.format(selection[i]))
                keypass(Ek("up"))

    elif opt.diff or opt.cdiff:
        res = RunAndTest(evaluator,celsius=opt.celsius,dt=opt.simdt)(params=prms)
        for i,r in enumerate(res):
            print(f"{i},{r.tolist()}".replace("[","").replace("]","").replace(" ",""))
            #print("  :", evaluator.spikezoomers[i])
        #print("len=",len(res))
        # from matplotlib.pyplot import *
        # ax1 = subplot(121)
        # ax2 = subplot(122,sharex=ax1)
        # for i,r in enumerate(res):
            # ax1.plot(r)
            # ax2.plot(evaluator.spikezoomers[i])
        # show()
        
    else:
        ev  = RunAndTest(evaluator,celsius=opt.celsius,dt=opt.simdt).__run__(params=prms)
        print(ev.vector())
        
        
