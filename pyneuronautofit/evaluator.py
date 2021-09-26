import sys, logging, os
from numpy import *
from scipy import stats as spstats


class Evaluator():
    def __init__(self, inputfile:(str,None), mode:str="RALMN", mask:(dict,list,tuple,int,None)=None,\
                       prespike:int=10, postspike:int=20, spikethreshold:float=0., spikecount:(int,None)=-1,\
                       pedant:bool=True, collapse_tests:bool=False, savetruedata:bool=False):
        """
        mode: (str)
            T - spike times
            S - spike shapes during stimulus
            U - squared error of subthreshold voltage
            W - spike width during stimulus
            R - resting potential
            L - post-stimulus tail statistics
            M - voltage stimulus statistics
            A - average spike shape during stimulus
            N - number of spikes
            C - distance between voltages during stimulus
            D - distance between voltages during after stimulus tails
            V - distance between voltages
            O - Just total number of spikes
        mask:
            if (int):
                use from all traces upto this index
            if (tuple):
                must be length 2: begin - end traces to use
            if (list):
                indices of traces to use
            if (dict)
                int, tuple or list for each mode
        """
        self.mode           = mode.upper()
        self.mask           = mask
        self.prespike       = prespike
        self.postspike      = postspike
        self.spikethreshold = spikethreshold
        self.spikecount     = spikecount
        if self.spikecount is None: self.spikecount = -1
        self.pedant         = pedant
        self.collapse       = collapse_tests
        self.savetruedata   = savetruedata
        
        
        if not inputfile is None:
            fname, fext = os.path.splitext(inputfile)
            if   fext == ".abf" or fext == ".ABF":
                self.readABF(inputfile)
            elif fext == ".npz" or fext == ".NPZ":
                self.readNPZ(inputfile)
            elif fext == ".json" or fext == ".JSON":
                self.readJSON(inputfile)
            else:
                logging.error(f"Unknown file type {fext}")
                raise BaseException(f"Unknown file type {fext}")
    

    def _detect_stims(self,currents:list,dt:float):
        more1std = [ where(abs(i-mean(i)) > std(i))[0] for i in currents ]
        mleft,mright = {},{}
        for m in more1std:
            if m[0] in mleft:mleft[m[0]] += 1
            else : mleft[m[0]] = 1
            if m[-1] in mright:mright[m[-1]] += 1
            else : mright[m[-1]] = 1
        self.stimLidx,self.stimRidx =\
            sorted([ [mleft[m] ,m] for m in mleft ])[-1][1],\
            sorted([ [mright[m],m] for m in mright])[-1][1]
        self.stimLidx,self.stimRidx = int(self.stimLidx),int(self.stimRidx)
        self.stimLtime,self.stimRtime = self.stimLidx*dt, self.stimRidx*dt


    def getspikeidx(self, rec:ndarray):
        idx = where(rec > self.spikethreshold)[0]
        if len(idx) == 0: return [],[]
        idu = [ idx[0] ] + [ x2 for x1,x2 in zip(idx[:-1],idx[1:]) if x2 > x1+1 ]
        idd = [ x1 for x1,x2 in zip(idx[:-1],idx[1:]) if x2 > x1+1 ] + [ idx[-1] ]
        return idu,idd        
    
    def _cmam(self,mod,io): #check mode and mask
        if not mod in self.mode: return False
        if self.mask is None: return True
        if type(self.mask) is int:
            return io <= self.mask
        if type(self.mask) is tuple:
            if len(self.mask) == 2: return self.mask[0] <= io <= self.mask[1]
            logging.error(f"Mask in tuple should have len 2, got {len(self.mask)}")
            raise BaseException(f"Mask in tuple should have len 2, got {len(self.mask)}")
        if type(self.mask) is list:
            return io in self.mask
        if type(self.mask) is dict:
            if not mod in self.mask: return True
            if self.mask[mod] is None: return True
            if type(self.mask[mod]) is int:
                return io <= self.mask[mod]
            if type(self.mask[mod]) is tuple:
                if len(self.mask[mod]) == 2: return  self.mask[mod][0] <= io <= self.mask[mod][1]
                logging.error(f"Mask in mode {mod} is tuple should have len 2, got {len(self.mask[mod])}")
                raise BaseException(f"Mask in mode {mod} is tuple should have len 2, got {len(self.mask[mod])}")
            if type(self.mask[mod]) is list:
                return io in self.mask[mod]
            logging.error(f"Unknown mask type {type(self.mask[mod])} for mode {mod}")
            raise BaseException(f"Unknown mask type {type(self.mask[mod])} for mode {mod}")
            
        logging.error(f"Unknown mask type {type(self.mask[mod])}")
        raise BaseException(f"Unknown mask type {type(self.mask[mod])}")
        
    
    def assess(self,obsdata:list)->list:
        rend = {}
        if 'S' in self.mode: rend['S'] = []
        if 'T' in self.mode: rend['T'] = []
        if 'W' in self.mode: rend['W'] = []
        if 'U' in self.mode: rend['U'] = []
        if 'R' in self.mode: rend['R'] = None
        if 'L' in self.mode: rend['L'] = []
        if 'M' in self.mode: rend['M'] = []
        if 'A' in self.mode: rend['A'] = []
        if 'N' in self.mode: rend['N'] = []
        if 'C' in self.mode: rend['C'] = []
        if 'D' in self.mode: rend['D'] = []
        if 'V' in self.mode: rend['V'] = []
        if 'O' in self.mode: rend['O'] = []

        for io,o in enumerate(obsdata):
            # if 'S' in self.mode: rend['S'].append([])
            # if 'T' in self.mode: rend['T'].append([])
            # if 'W' in self.mode: rend['W'].append([])
            # if 'U' in self.mode: rend['U'].append( copy(o) )
            for n in 'S T W'.split():
                if self._cmam(n,io): rend[n].append([])
            if self._cmam('U',io): rend['U'].append( copy(o) )
            if self._cmam('V',io): rend['V'].append( copy(o) )

            
            if self._cmam('R',io):
                rend['R'] = o[:self.stimLidx] if rend['R'] is None else concatenate((rend['R'],o[:self.stimLidx]))
            spkidx = self.getspikeidx(o)
            if self._cmam('N',io):
                # #DB>>
                # print(f"  N:{io}")
                # #<<DB
                rend['N'].append( len( spkidx[0]                                                                          ) ) # total number of spikes
                rend['N'].append( len([ idu for idu in spkidx[0] if self.stimLidx <= idu <= self.stimRidx                ]) ) # spikes during stimulus
                rend['N'].append( len([ idu for idu in spkidx[0] if self.stimRidx <= idu <= 2*self.stimRidx-self.stimLidx]) ) # spikes after stimulus
            if self._cmam('O',io):
                rend['O'].append( len( spkidx[0]                                                                          ) ) # total number of spikes
                
            if self._cmam('A',io):
                rend['A'].append( [None,0] )
            
            if self._cmam('C',io): rend['C'].append( copy(o[self.stimLidx:self.stimRidx] ) )
            if self._cmam('D',io): rend['D'].append( copy(o[self.stimRidx:             ] ) )
            for sid,(idu,idd) in enumerate( zip(*spkidx) ):
                lidx = (idu-self.prespike ) if idu-self.prespike  > 0          else 0
                ridx = (idu+self.postspike) if idu+self.postspike < o.shape[0] else None
                spshp = array( \
                    [ o[lidx                         ] for k in range(0,self.prespike-idu,1)]\
                    + o[lidx:ridx].tolist() + \
                    [ o[-1 if ridx is None else ridx ] for k in range(0,idu+self.postspike-o.shape[0],1) ] 
                )
                if self._cmam('S',io) and  self.stimLidx <= idu <= self.stimRidx:
                    rend['S'][-1].append(spshp)
                if self._cmam('A',io) and  self.stimLidx <= idu <= self.stimRidx:
                    if rend['A'][-1][1] == 0:
                        rend['A'][-1][0]  = spshp
                    else:
                        #DB>>
                        if rend['A'][-1][0].shape != spshp.shape:
                            print("#DB>>",lidx,ridx,self.prespike+self.postspike, rend['A'][-1][0].shape,spshp.shape)
                            exit(1)
                        #<<DB
                        rend['A'][-1][0] += spshp
                    rend['A'][-1][1] += 1
                if self._cmam('U',io):
                    rend['U' ][-1][lidx:ridx] = nan
                if self._cmam('T',io):
                    rend['T'][-1].append(float(idu)*self.expdt)
                if self._cmam('W',io) and  self.stimLidx <= idu <= self.stimRidx:
                    rend['W'][-1].append(float(idd-idu)*self.expdt)
            if self._cmam('T',io) and self.spikecount > 0:
                while len(rend['T'][-1]) < self.spikecount:
                    rend['T'][-1].append(self.tmax+1)
                if len(rend['T'][-1]) > self.spikecount and self.spikecount > 0:
                    rend['T'][-1] = rend['T'][-1][:self.spikecount]
            if self._cmam('W',io) and self.spikecount > 0:
                while len(rend['W'][-1]) < self.spikecount:
                    rend['W'][-1].append(self.tmax+1)
                if len(rend['W'][-1]) > self.spikecount and self.spikecount > 0:
                    rend['W'][-1] = rend['W'][-1][:self.spikecount]
            if self._cmam('S',io) and self.spikecount > 0:
                while len(rend['S'][-1]) < self.spikecount:
                    rend['S'][-1].append( ones(self.prespike+self.postspike)*1e3)
                if len(rend['S'][-1]) > self.spikecount and self.spikecount > 0:
                    rend['S'][-1] = rend['S'][-1][:self.spikecount]
            if self._cmam('L',io):
                vstd       = std(o[self.stimRidx:])
                xstat      = [ mean(o[self.stimRidx:]), vstd ]
                #Skewness and Kurtosis
                moments    = spstats.moment(o[self.stimRidx:],linspace(3,4,2))/power(vstd,linspace(3,4,2))
                rend['L'] += xstat+moments.tolist()
            if self._cmam('M',io):
                vstd       = std(o[self.stimLidx:self.stimRidx])
                xstat      = [ mean(o[self.stimLidx:self.stimRidx]), vstd ]
                #Skewness and Kurtosis
                moments    = spstats.moment(o[self.stimLidx:self.stimRidx],linspace(3,4,2))/power(vstd,linspace(3,4,2))
                rend['M'] += xstat+moments.tolist()
            if self._cmam('A',io):
                rend['A'][-1] = (ones(self.prespike+self.postspike)*1e3) if rend['A'][-1][0] is None else (rend['A'][-1][0]/rend['A'][-1][1])

        if 'R' in self.mode:
            rend['R'] = array( [mean(rend['R']), std(rend['R'])] )
        if 'L' in self.mode: rend['L'] = array(rend['L'])
        if 'M' in self.mode: rend['M'] = array(rend['M'])
        if 'N' in self.mode: rend['N'] = array(rend['N'])
        if 'O' in self.mode: rend['O'] = array(rend['O'])
        # #DB>>
        # print(rend['N'].shape)
        # #<<DB
                
        return rend

    def _gmi(self,mod,io): #get masked id
        if self.mask       is None:  return io
        if type(self.mask) is int:   return io
        if type(self.mask) is tuple: return io - self.mask[0]
        if type(self.mask) is list:  return self.mask.index(io)
        if type(self.mask) is dict:
            if mod in self.mask:
                if type(self.mask[mod]) is None:  return io 
                if type(self.mask[mod]) is int:   return io 
                if type(self.mask[mod]) is tuple: return io - self.mask[mod][0]
                if type(self.mask[mod]) is list:  return self.mask[mode].index(io)
        return io
        
    def __log_data_stats_(self):
        logging.debug(f"   > Stim time = [{self.stimLtime},{self.stimRtime}] ms")
        logging.debug(f"   > Spike wnd = [{self.prespike*self.expdt},{self.postspike*self.expdt}] ms")
        logging.debug(f"   > Spike thr = {self.spikethreshold}")
        logging.debug(f"   > Spike cnt = {self.spikecount}")
        logging.debug(f"   > dt        = {self.expdt} ms")
        logging.debug(f"   > tmax      = {self.tmax} ms")
        logging.debug(f"   > Data stat = [")
        for i in range(self.Nrec):
            logging.debug(f"     > {i:02d} = [")
            if self._cmam('U',i):
                x = self.cond['U'][self._gmi('U',i)]
                logging.debug(f"       > slike less v = {x}")
            if self._cmam('S',i):
                logging.debug(f"       > slike shapes = [")
                for k,s in enumerate(self.cond['S'][self._gmi('S',i)]):
                    logging.debug(f"          > {k:02d} = {s}")
                logging.debug(f"         ]")
            if self._cmam('T',i):
                x = self.cond['T'][self._gmi('T',i)]
                logging.debug(f"       > spike times = {x}")
            if self._cmam('W',i):
                x = self.cond['W'][self._gmi('W',i)]
                logging.debug(f"       > spike width = {x}")
            logging.debug(f"       ]")
        logging.debug(f"     ]")    
            
    
    def readABF(self,abf:str):
        import pyabf
        logging.debug(f"Reading ABF: {abf}")
        rec = pyabf.ABF(abf)
        logging.debug(f"   > Units     = {rec.adcUnits}")
        self.expdt = 1000./float(rec.dataRate)
        logging.debug(f"   > Chan N    = {rec.channelCount}")
        TrueData = []
        self.TestCurr = []
        self.tmax  = -1000
        for sweepNumber in rec.sweepList:
            rec.setSweep(sweepNumber = sweepNumber, channel=0)	
            TrueData.append( array(rec.sweepY) )
            rec.setSweep(sweepNumber = sweepNumber, channel=1)	
            self.TestCurr.append( array(rec.sweepY) )
            if self.tmax < TrueData[-1].shape[0]*self.expdt:
                self.tmax = TrueData[-1].shape[0]*self.expdt
        self._detect_stims(self.TestCurr,self.expdt)
        self.Nrec = len(TrueData)
        self.cond = self.assess(TrueData)
        if self.savetruedata: self.TrueData = TrueData
        self.__log_data_stats_()
        
    
    def readNPZ(self,npz:str):
        with load(npz) as npx:
            self.mode           = npx["mode"]
            self.mask           = npx["mask"]
            self.pedant         = npx["pedant"]
            self.collapse       = npx["collapse"]
            self.Nrec           = npx["nrec"]
            self.prespike       = npx["prespike"]
            self.postspike      = npx["postspike"]
            self.spikethreshold = npx["spikethreshold"]
            self.spikecount     = npx["spikecount"]
            self.expdt          = npx["expdt"]
            self.tmax           = npx["tmax"]
            self.stimLidx       = npx["stimLidx"]
            self.stimRidx       = npx["stimRidx"]
            self.stimLtime      = npx["stimLtime"]
            self.stimRtime      = npx["stimRtime"]
            self.cond           = npx["cond"]
            self.TestCurr       = npx["currents"]

        logging.debug(f"Reading NPZ: {npz}")
        self.__log_data_stats_()
        
        
    def exportNPZ(self,filename:str):
        if filename[-4:] != ".npz" : filename += ".npz"
        savez(filename,
            mode            = self.mode,
            mask            = self.mask,
            pedant          = self.pedant,
            collapse        = self.collapse,
            nrec            = self.Nrec,
            prespike        = self.prespike,
            postspike       = self.postspike,
            spikethreshold  = self.spikethreshold,
            spikecount      = self.spikecount,
            expdt           = self.expdt,
            tmax            = self.tmax,
            stimLidx        = self.stimLidx,
            stimRidx        = self.stimRidx,
            stimLtime       = self.stimLtime,
            stimRtime       = self.stimRtime,
            cond            = self.cond,
            currents        = self.TestCurr
        )


    def readJSON(self,jsonfile:str):
        import json
        with open(jsonfile) as fd:
            rend = json.load(fd)
        if not type(rend) is dict:
            logging.error(f"JSON object is not a dict: {type(rend)}")
            raise TypeError(f"JSON object is not a dict: {type(rend)}")
        for n,t in zip(\
            'mode nrec prespike postspike spikethreshold spikecount expdt tmax stimLidx stimRidx stimLtime stimRtime cond currents '.split(),\
            (str,  int,  int,     int,       float,         int,    float,float,  int,     int,     float,    float, dict, list,   ) ):
            if not n in rend:
                logging.error(f"There is no \'{n}\' in JSON object")
                raise RuntimeError(f"There is no \'{n}\' in JSON object")
            if not type(rend[n]) is t:
                logging.error(f"{n} has a wrong type in JSON object: have {type(rend[n])} but should be {t}")
                raise TypeError(f"{n} has a wrong type in JSON object: have {type(rend[n])} but should be {t}")

        for n,t in zip('U S T W R L M A N C D V O'.split(),(list,list,list,list,list,list,list,list,list,list,list,list,list) ):
            if not n in rend['mode']: continue
            if not n in rend['cond']:
                logging.error(f"There is no \'{n}\' in condition {i} ")
                raise RuntimeError(f"There is no \'{n}\' in condition {i} ")
            if not type(rend['cond'][n]) is t:
                logging.error(f"{n} has a wrong type in condition {i}: have {type(rend['cond'][n])} but should be {t}")
                raise  TypeError(f"{n} has a wrong type in condition {i}: have {type(rend['cond'][n])} but should be {t}")

        if 'U' in rend['mode']:
            rend['U'] = [ array(o)             for o in rend['U'] ]
        if 'V' in rend['mode']:
            rend['V'] = [ array(o)             for o in rend['V'] ]
        if 'S' in rend['mode']:
            rend['S']= [ [array(p) for p in o] for o in rend['S'] ]
        if 'A' in rend['mode']:
            rend['A']= [ array(o)              for o in rend['A'] ]
        for n in 'R L M N O'.split():
            if n in rend['mode']:
                rend['cond'][n] = array(rend['cond'][n])
        if 'C' in rend['mode']:
            rend['C']= [ array(o)             for o in rend['C'] ]
        if 'D' in rend['mode']:
            rend['D']= [ array(o)             for o in rend['D'] ]

        self.mode           = rend['mode']
        self.mask           = rend['mask']
        self.pedant         = rend['pedant']
        self.collapse       = rend['collapse']
        self.Nrec           = rend['nrec']
        self.prespike       = rend['prespike']
        self.postspike      = rend['postspike']
        self.spikethreshold = rend['spikethreshold']
        self.spikecount     = rend['spikecount']
        self.expdt          = rend['expdt']
        self.tmax           = rend['tmax']
        self.stimLidx       = rend['stimLidx']
        self.stimRidx       = rend['stimRidx']
        self.stimLtime      = rend['stimLtime']
        self.stimRtime      = rend['stimRtime']
        self.cond           = rend['cond']
        self.TestCurr       = [ array(c) for c in rend['currents'] ]
        logging.debug(f"Reading JSON: {jsonfile}")
        self.__log_data_stats_()
        del json

    def exportJSON(self,filename:str):
        if filename[-5:] != ".json" : filename += ".json"
        import json
        rend = {
            'mode'           : self.mode,
            'mask'           : self.mask,
            'pedant'         : self.pedant,
            'collapse'       : self.collapse,
            'nrec'           : self.Nrec,
            'prespike'       : self.prespike,
            'postspike'      : self.postspike,
            'spikethreshold' : self.spikethreshold,
            'spikecount'     : self.spikecount,
            'expdt'          : self.expdt,
            'tmax'           : self.tmax,
            'stimLidx'       : self.stimLidx,
            'stimRidx'       : self.stimRidx,
            'stimLtime'      : self.stimLtime,
            'stimRtime'      : self.stimRtime,
            'cond'           : {},
            'currents'       : [
                c.tolist() for c in self.TestCurr
            ]
        }
        if 'U' in self.mode:
            rend['cond']['U'] = [ o.tolist() for o in self.cond['U'] ]
        if 'V' in self.mode:
            rend['cond']['V'] = [ o.tolist() for o in self.cond['V'] ]
        if 'S' in self.mode:
            rend['cond']['S'] = [ [ p.tolist() for p in o] for o in self.cond['S'] ]
        if 'A' in self.mode:
            rend['cond']['A'] = [ o.tolist() for o in self.cond['A'] ]
        if 'T' in self.mode:
            rend['cond']['T'] = self.cond['T']
        if 'W' in self.mode:
            rend['cond']['W'] = self.cond['W']

        for n in 'R L M A N O'.split():
            if n in self.mode:
                rend['cond'][n] = self.cond[n].tolist()

        if 'C' in self.mode:
            rend['cond']['C']   = [ o.tolist() for o in self.cond['C'] ]
        if 'D' in self.mode:
            rend['cond']['D']   = [ o.tolist() for o in self.cond['D'] ]

        with open(filename,"w") as fd:
            json.dump(rend,fd)
        del json

    def clone(self,data):
        n                =  Evaluator(None)
        n.mode           =  self.mode
        n.mask           =  self.mask
        n.pedant         =  self.pedant
        n.collapse       =  self.collapse
        n.Nrec           =  self.Nrec if data is None else len(data)
        n.prespike       =  self.prespike
        n.postspike      =  self.postspike
        n.spikethreshold =  self.spikethreshold
        n.spikecount     =  self.spikecount
        n.expdt          =  self.expdt
        n.tmax           =  self.tmax
        n.stimLidx       =  self.stimLidx
        n.stimRidx       =  self.stimRidx
        n.stimLtime      =  self.stimLtime
        n.stimRtime      =  self.stimRtime
        n.cond           =  data if data is None else n.assess(data) 
        n.TestCurr       =  []
        return n
        

    def vector(self,marks=False)->list:
        vec = []
        if 'U' in self.mode:
            vec +=  [ o.tolist() for o in self.cond['U'] ]
        if 'V' in self.mode:
            vec +=  [ o.tolist() for o in self.cond['V'] ]
        if 'S' in self.mode:
            vec +=  [ [ p.tolist() for p in o] for o in self.cond['S'] ]
        if 'T' in self.mode:
            vec +=  self.cond['T']
        if 'W' in self.mode:
            vec +=  self.cond['W']
        if 'A' in self.mode:
            vec +=  [ o.tolist() for o in self.cond['A'] ]

        for n in 'R L M N O'.split():
            if n in self.mode:
                vec += self.cond[n].tolist()

        if 'C' in self.mode:
            vec += [ o.tolist() for o in self.cond['C'] ]
        if 'D' in self.mode:
            vec += [ o.tolist() for o in self.cond['D'] ]
        return vec

    
    def scores(self,marks=False)->list:
        clone0 = self.clone(None)
        clone0.cond = {}
        if 'U' in self.mode:
            clone0.cond['U'] =  [ zeros(o.shape) for o in self.cond['U'] ]
        if 'V' in self.mode:
            clone0.cond['V'] =  [ zeros(o.shape) for o in self.cond['V'] ]
        if 'S' in self.mode:
            clone0.cond['S'] =  [ [zeros(p.shape) for p in o] for o in self.cond['S'] ]
        if 'T' in self.mode:
            clone0.cond['T'] =  [ [ 0. for p in x ] for x in   self.cond['T'] ]
        if 'W' in self.mode:
            clone0.cond['W'] =  [ [ 0. for p in x ] for x in   self.cond['W'] ]
        if 'A' in self.mode:
            clone0.cond['A'] =  [ zeros(o.shape) for o in self.cond['A'] ]

        for n in 'R L M N O'.split():
            if n in self.mode:
                clone0.cond[n] = zeros(self.cond[n].shape)

        if 'C' in self.mode:
            clone0.cond['C'] = [ zeros(o.shape) for o in self.cond['C'] ]
        if 'D' in self.mode:
            clone0.cond['D'] = [ zeros(o.shape) for o in self.cond['D'] ]
        return self.diff(clone0,marks=marks)
    

    def _num_cond_(self,oo)->int:
        if self.pedant:
            if self.Nrec != oo.Nrec:
                logging.error("Numbers of conditions are different\n\n")
                raise RuntimeError("Numbers of conditions are different")
            return self.Nrec
        else:
            return min(self.Nrec,oo.Nrec)
    
    def _num_spikes_(self,oo,spsh1, spsh2)->int:
        if self.pedant:
            if oo.spikecount != self.spikecount:
                logging.error("Spike counts are different\n\n")
                raise RuntimeError("Spike counts are different")
        if self.spikecount < 0:
            if   spsh1 is None          : ns = len(spsh2)
            elif spsh2 is None          : ns = len(spsh1)
            elif len(spsh1) > len(spsh2): ns = len(spsh1)
            else                        : ns = len(spsh2)
        else:
            ns = self.spikecount
        return ns
            
    def subthreshold_error(self, oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('U',i) ]

        if len(self.cond['U']) != len(oo.cond['U']):
                logging.error("Numbers of U-conditions are different\n\n")
                raise RuntimeError("Numbers of U-conditions are different")
        rend = []
        for stv1, stv2 in zip(self.cond['U'],oo.cond['U']):
            if stv1.shape[0] == stv2.shape[0]:
                subdiff = (stv1-stv2)**2
            else:
                lstop = min(stv1.shape[0], stv2.shape[0])
                subdiff = (stv1[:lstop]-stv2[:lstop])**2
            subdiff = subdiff[~isnan(subdiff)]
            if self.collapse:
                rend.append( ( sum(subdiff),subdiff.shape[0] ) )
            else:
                rend.append(sqrt( sum(subdiff) )/subdiff.shape[0])
        if self.collapse:
            rend = array(rend)
            rend = rend[where(rend[:,1]>0)]
            rend = [sqrt( sum(rend[:,0]*rend[:,1])/sum(rend[:,1]) )]
        return rend
    def spiketimes_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('T',i)  for j in range(self.spikecount if self.spikecount > 0 else 1)]
        if len(self.cond['T']) != len(oo.cond['T']):
                logging.error("Numbers of T-conditions are different\n\n")
                raise RuntimeError("Numbers of T-conditions are different")
        rend = []
        for spt1,spt2 in zip(self.cond['T'],oo.cond['T']):
            ns = self._num_spikes_(oo,spt1, spt2)
            spt1 = array(spt1[:ns]+[self.tmax+1 for i in range(0,ns - len(spt1))])
            spt2 = array(spt2[:ns]+[  oo.tmax+1 for i in range(0,ns - len(spt2))])
            rend += (abs( spt1 - spt2  )).tolist()
        return [sum(rend)] if self.collapse else rend
    def spikeshape_error(self, oo)->list:
        if self.cond is None or oo.cond is None:
            return [ [None for j in range(self.spikecount if self.spikecount > 0 else 1)] for i in range(self.Nrec) if self._cmam('S',i) ]
        if len(self.cond['S']) != len(oo.cond['S']):
                logging.error("Numbers of S-conditions are different\n\n")
                raise RuntimeError("Numbers of S-conditions are different")
        rend = []
        for spsh1,spsh2 in zip(self.cond['S'],oo.cond['S']):
            ns = self._num_spikes_(oo,spsh1, spsh2)
            for spidx in range(ns):
                sp1  = (ones(self.prespike+self.postspike)*1e3) if spidx >= len(spsh1) else copy(spsh1[spidx])
                sp2  = (ones(self.prespike+self.postspike)*1e3) if spidx >= len(spsh2) else copy(spsh2[spidx])
                if sp1.shape[0] != sp2.shape[0]:
                    logging.error(f"Spike size isn't equal spike index {spidx}/{len(spsh1)}/{len(spsh2)}: {sp1.shape[0]},{sp2.shape[0]},{self.prespike+self.postspike}")
                    if sp1.shape[0] > sp2.shape[0]:sp1.shape[0] =sp1.shape[0][:sp2.shape[0]]
                    else                          :sp2.shape[0] =sp2.shape[0][:sp1.shape[0]]
                rend.append( sum( abs(sp1-sp2) ) )
        if self.collapse:
            rend = [sum(rend)]
        return rend
    def spikewidth_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ [ None for j in range(self.spikecount if self.spikecount > 0 else 1)] for i in range(self.Nrec) if self._cmam('W',i) ]
        if len(self.cond['W']) != len(oo.cond['W']):
                logging.error("Numbers of W-conditions are different\n\n")
                raise RuntimeError("Numbers of W-conditions are different")
        rend = []
        for spwd1,spwd2 in zip(self.cond['W'],oo.cond['W']):
            # ns = self._num_spikes_(oo,spwd1, spwd2 )
            if   len(spwd1) < len(spwd2):
                spw1  = array(spwd1+[0. for i in range(len(spwd2) - len(spwd1))])
                spw2  = array(spwd2)
            elif len(spwd2) < len(spwd2):
                spw2  = array(spwd1+[0. for i in range(len(spwd1) - len(spwd2))])
                spw1  = array(spwd1) 
            else:
                spw1  = array(spwd1) 
                spw2  = array(spwd2)
            rend += abs(spw1-spw2).tolist()              
        if self.collapse:
            rend = [ mean(array(rend)) ]
        return rend
    
    def restingpot_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [None, None]
        diff = abs(self.cond['R']-oo.cond['R'])
        return [sum(diff)] if self.collapse else diff.tolist()
    
    def poststtail_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('L',i) for j in range(4)]
        if len(self.cond['L']) != len(oo.cond['L']):
                logging.error("Numbers of L-conditions are different\n\n")
                raise RuntimeError("Numbers of L-conditions are different")
        diff = abs(self.cond['L']-oo.cond['L'])
        return [sum(diff)] if self.collapse else diff.tolist()
    
    def voltststat_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('M',i) for j in range(4)]
        rend =  abs(self.cond['M'] - oo.cond['M'])
        return [sum(rend)] if self.collapse else rend.tolist()
    
    def avspkshape_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('A',i) ]
        if len(self.cond['A']) != len(oo.cond['A']):
                logging.error("Numbers of A-conditions are different\n\n")
                raise RuntimeError("Numbers of A-conditions are different")
        diff = [ sum(abs( x - y ))/x.shape[0] for x,y in zip(self.cond['A'],oo.cond['A']) ]
        return [ sum(diff) ] if self.collapse else diff

    def spikenumbr_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('N',i) ]
        diff = abs( self.cond['N'] - oo.cond['N'] )
        return [sum(diff)] if self.collapse else diff.tolist()
    
    def stimvoltdf_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('C',i)]
        if len(self.cond['C']) != len(oo.cond['C']):
                logging.error("Numbers of C-conditions are different\n\n")
                raise RuntimeError("Numbers of C-conditions are different")
        rend = []
        for stvl1,stvl2 in zip(self.cond['C'],oo.cond['C']):
            if stvl1.shape != stvl2.shape:
                logging.error(f"Active voltage shapes are different{stvl1.shape} != {stvl2.shape}")
                if stvl1.shape[0] > stvl2.shape[0]:stvl1.shape =stvl1.shape[:stvl2.shape]
                else                              :stvl2.shape =stvl2.shape[:stvl1.shape]
            stvldiff = (stvl1-stvl2)**2
            if self.collapse:
                rend .append( (sum(stvldiff),stvldiff.shape[0]))
            else:
                stvldiff = sqrt(sum(stvldiff))/stvldiff.shape[0]
                rend .append(stvldiff)
        if self.collapse:
            rend = array(rend)
            rend = rend[where(rend[:,1]>0)]
            rend = [sqrt( sum(rend[:,0]*rend[:,1])/sum(rend[:,1]) )]
        return rend

    def tailvoltdf_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('D',i)]
        if len(self.cond['D']) != len(oo.cond['D']):
                logging.error("Numbers of D-conditions are different\n\n")
                raise RuntimeError("Numbers of D-conditions are different")
        rend = []
        for tlvl1,tlvl2 in zip(self.cond['D'],oo.cond['D']):
            if tlvl1.shape != tlvl2.shape:
                logging.error(f"Active voltage shapes are different{tlvl1.shape} != {tlvl2.shape}")
                if tlvl1.shape[0] > tlvl2.shape[0]:tlvl1.shape =tlvl1.shape[:tlvl2.shape]
                else                              :tlvl2.shape =tlvl2.shape[:tlvl1.shape]
            tlvldiff = (tlvl1-tlvl2)**2
            if self.collapse:
                rend.append( (sum(tlvldiff),tlvldiff.shape[0]) )
            else:
                tlvldiff = sqrt(sum(tlvldiff))/tlvldiff.shape[0]
                rend.append(tlvldiff)
        if self.collapse:
            rend = array(rend)
            rend = rend[where(rend[:,1]>0)]
            rend = [sqrt( sum(rend[:,0]*rend[:,1])/sum(rend[:,1]) )]
        return rend
    def totlvoltdf_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('V',i)]
        if len(self.cond['V']) != len(oo.cond['V']):
                logging.error("Numbers of V-conditions are different\n\n")
                raise RuntimeError("Numbers of V-conditions are different")
        rend = []
        for vl1,vl2 in zip(self.cond['V'],oo.cond['V']):
            m = min(vl1.shape[0],vl2.shape[0])
            rend.append(sum((vl1[:m]-vl2[:m])**2)/m)
        if self.collapse:
            return mean( array(rend) )
        return rend
    def spikecount_error(self,oo)->list:
        if self.cond is None or oo.cond is None:
            return [ None for i in range(self.Nrec) if self._cmam('V',i)]
        if self.cond['O'].shape[0] != oo.cond['O'].shape[0]:
                logging.error("Numbers of O-conditions are different\n\n")
                raise RuntimeError("Numbers of O-conditions are different")
        if self.collapse:
            return sum(abs(self.cond['O']-oo.cond['O']))
        return abs(self.cond['O']-oo.cond['O']).tolist()
    def diff(self, oo, marks:bool=False)->list:
        if oo.mode != self.mode:
            logging.error(f"Different modes {oo.mode} vs {self.mode}")
            RuntimeError( f"Different modes {oo.mode} vs {self.mode}")
        if "U" in self.mode or 'S' in self.mode:
            if 2.*(oo.expdt - self.expdt)/(oo.expdt + self.expdt) > 0.01:
                logging.error(f"Different dt {oo.expdt} vs {self.expdt}")
                RuntimeError( f"Different dt {oo.expdt} vs {self.expdt}")
        if "W" in self.mode or 'S' in self.mode:
            if oo.spikecount != self.spikecount:
                logging.error(f"Different number of spike counts {oo.spikecount} vs {self.spikecount}")
                RuntimeError( f"Different number of spike counts {oo.spikecount} vs {self.spikecount}")

        u = self.subthreshold_error(oo) if "U" in self.mode else []
        t = self.spiketimes_error(oo)   if "T" in self.mode else []
        s = self.spikeshape_error(oo)   if "S" in self.mode else []
        w = self.spikewidth_error(oo)   if "W" in self.mode else []
        r = self.restingpot_error(oo)   if "R" in self.mode else []
        l = self.poststtail_error(oo)   if "L" in self.mode else []
        m = self.voltststat_error(oo)   if "M" in self.mode else []
        a = self.avspkshape_error(oo)   if "A" in self.mode else []
        n = self.spikenumbr_error(oo)   if "N" in self.mode else []
        c = self.stimvoltdf_error(oo)   if "C" in self.mode else []
        d = self.tailvoltdf_error(oo)   if "D" in self.mode else []        
        v = self.totlvoltdf_error(oo)   if "V" in self.mode else []
        o = self.spikecount_error(oo)   if "O" in self.mode else []

        rend = []
        for _ in self.mode:
            if _ == "U": rend += [ ['U',x] for x in u ] if marks else u
            if _ == "T": rend += [ ['T',x] for x in t ] if marks else t
            if _ == "S": rend += [ ['S',x] for x in s ] if marks else s
            if _ == "W": rend += [ ['W',x] for x in w ] if marks else w
            if _ == "R": rend += [ ['R',x] for x in r ] if marks else r
            if _ == "L": rend += [ ['L',x] for x in l ] if marks else l
            if _ == "M": rend += [ ['M',x] for x in m ] if marks else m
            if _ == "A": rend += [ ['A',x] for x in a ] if marks else a
            if _ == "N": rend += [ ['N',x] for x in n ] if marks else n
            if _ == "C": rend += [ ['C',x] for x in c ] if marks else c
            if _ == "D": rend += [ ['D',x] for x in d ] if marks else d
            if _ == "V": rend += [ ['V',x] for x in v ] if marks else v
            if _ == "O": rend += [ ['O',x] for x in o ] if marks else o
        #DB>>
        # print(f"DB>> u={len(u)}, t={len(t)}, s={len(s)}, w={len(w)}")
        # print(f"DB>> r={len(r)}, l={len(l)}, m={len(m)}, a={len(a)}, n={len(n)}")
        # print(f"DB>> rend={len(rend)}")
        #<<DB
        return rend

    def __sub__(self, oo) -> list:
        return self.diff(oo)
    
    def getmap(self) -> str:
        ret = []
        for _ in self.mode:
            if 'U'  == _: ret += [ 'U' for i in range(self.Nrec) if self._cmam('U',i) ]
            if 'T'  == _: ret += [ 'T' for i in range(self.Nrec) if self._cmam('T',i) for j in range(self.spikecount if self.spikecount > 0 else 1) ]
            if 'S'  == _: ret += [ 'S' for i in range(self.Nrec) if self._cmam('S',i) for j in range(self.spikecount if self.spikecount > 0 else 1) ]
            if 'W'  == _: ret += [ 'W' for i in range(self.Nrec) if self._cmam('W',i) for j in range(self.spikecount if self.spikecount > 0 else 1) ] 
            if 'R'  == _: ret += [ 'R','R' ]
            if 'L'  == _: ret += [ 'L' for i in range(self.Nrec) if self._cmam('L',i) for j in range(4)]
            if 'M'  == _: ret += [ 'M' for i in range(self.Nrec) if self._cmam('M',i) for j in range(4)]
            if 'A'  == _: ret += [ 'A' ]
            if 'N'  == _: ret += [ 'N' for i in range(self.Nrec) if self._cmam('N',i) ]
            if 'C'  == _: ret += [ 'C' for i in range(self.Nrec) if self._cmam('C',i) ]
            if 'D'  == _: ret += [ 'D' for i in range(self.Nrec) if self._cmam('D',i) ]
            if 'V'  == _: ret += [ 'V' for i in range(self.Nrec) if self._cmam('V',i) ]
            if 'O'  == _: ret += [ 'O' for i in range(self.Nrec) if self._cmam('O',i) ]
        return "".join(ret)
