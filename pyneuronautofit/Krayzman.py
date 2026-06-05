import time
import logging
import numpy as np
from inspyred.ec import Individual
from inspyred.ec.replacers import generational_replacement


class Krayzman(Individual):
    """
    This class serves as the foundation for Krayzman’s algorithm for 
    multi-objective optimization, which is based on adaptive weights. 
    Any instance of this class shares weights with all other instances, 
    allowing for updates to weights in one operation and dynamically 
    reevaluating the scaled fitness.

    Krayzman’s algorithm in a nutshell.
    - Weights are first set up as 1/standard_deviation for all fitness 
      values of the initial population.
    - GA uses sum of fitness*weight
    - At each evolution point, the weighted sum of fitness is computed 
      for all vectors in the current population. Then, correlations 
      between individual fitness functions and the final weighted 
      fitness are computed. 
      - The algorithm iteratively increases the weight for the highest 
        correlated fitness and increases the weight for the lowest 
        correlated fitness until the ratio between the minimal and the 
        maximal correlations is below a threshold.
    - If the ratio does not decrease, the weights are reset to 
      1/standard_deviation.
    Some fitness functions can be automatically masked if 
    standard_deviation is zero.
    
    The Krayzman’s algorithm effectively produces a gradient ascent, 
    flattening representations of individual fitness functions in 
    the weighted single-value fitness.
    """
    # This elegant way doesn't work, when this class is a object in another object
    #  I guess it is namespace problem. So switching to old-school explicit members
    # # Shared weights and masks
    # adaptive_weights = []
    # unmasked_weights = []
    
    def __set_weights_and_masks__(self,size:int):
        self.logger.debug(f"updating size to {size}") 
        if   len(self.adaptive_weights) < size:
            self.logger.info(f"adaptive weights is smaller {len(self.adaptive_weights)} than requested {size} ... updating") 
            self.adaptive_weights += [ 1 for i in range(len(self.adaptive_weights),size) ]
        elif len(self.adaptive_weights) == size:
            return
        else:
            self.logger.error(f"Cannot downsize existing weight vector of len={len(self.adaptive_weights)} to size {size}.")
            raise RuntimeError(f"Cannot downsize existing weight vector of len={len(self.adaptive_weights)} to size {size}.")
        if   len(self.unmasked_weights) < size:
            self.logger.info(f"maskes is smaller {len(self.unmasked_weights)} than requested {size} ... updating") 
            
            self.unmasked_weights = [ i for i in range(size) if self.adaptive_weights[i] > 0 ]
    def normalize_weights(self, normspace:bool, ):
        self.logger.debug(\
        "  > normalizing by space under the curve" \
        if normspace else \
        "  > normalizing by maximum")
        adaptive_weights = np.array( self.adaptive_weights[:] )
        adaptive_weights[self.unmasked_weights] = \
            adaptive_weights[self.unmasked_weights]/(\
                np.sum(adaptive_weights[self.unmasked_weights])\
                if normspace else \
                np.amax(adaptive_weights[self.unmasked_weights])\
            )
        self.adaptive_weights = adaptive_weights.tolist()[:]

    def init_weights(self,fitness:list, normspace:bool, variance_cutoff:float):
        self.logger.debug(f'KAW init weights:')
        variance         = np.var(np.array(fitness), axis = 0)
        self.logger.debug(f'  > variance         : {variance}')
        finite_ids,      = np.where( variance > variance_cutoff )
        self.logger.debug(f'  > finite_ids       : {finite_ids}')
        self.unmasked_weights = finite_ids.astype(int).tolist()
        self.logger.debug(f'  > unmasked_weights : {self.unmasked_weights}')
        inverce_variance = 1./variance[self.unmasked_weights]
        self.logger.debug(f'  > inverce_variance : {inverce_variance}')
        self.adaptive_weights = [
            float(inverce_variance[self.unmasked_weights.index(i)])\
            if i in self.unmasked_weights else -1. \
            for i in range(variance.shape[0])
        ]
        self.logger.debug(f'  > adaptive_weights : {self.adaptive_weights}')
        self.normalize_weights(normspace = normspace)
        self.logger.debug(f'  --------- NORMALIZATION -------------')
        self.logger.debug(f'  > adaptive_weights : {self.adaptive_weights}')

    def compute_correlations(self, fitness:list, wfitness:list):
        fitness = np.array( fitness )[ : ,self.unmasked_weights].T
        fitness = np.vstack( (fitness, wfitness) )
        corr    = np.corrcoef(fitness)[-1][:-1].tolist()
        minid   = np.argmin(corr)
        maxid   = np.argmax(corr)
        m2m     = float( corr[minid] / corr[maxid] )
        self.logger.debug(f"compute correlations: {m2m}={corr[minid]} /{corr[maxid]} : {corr}")
        return m2m, int(minid), int(maxid)
        
        
        
    def update_weights(self,accumulator:list, threshold:float = 0.0, normspace:bool=False, init=False, variance_cutoff:float=1e-9):
        import time
        fitness = []
        for pi,p in enumerate(accumulator):
            if   p is None:
                continue
            elif not isinstance(p,Krayzman):
                self.logger.error(f"item #{pi} in population is not a Krayzman's class but {type(p)}")
                raise RuntimeError(f"item #{pi} in population is not a Krayzman's class but {type(p)}")
            elif p.values is None:
                continue
            if len(fitness) > 0 and len(p.values) != len(fitness[-1]):
                self.logger.error(f"inhomogeneous optimization")
                raise RuntimeError(f"inhomogeneous optimization")
            if len(self.adaptive_weights) == 0 or len(self.unmasked_weights) == 0:
                self.__set_weights_and_masks__( len(p.values) )

            fitness.append( p.values )
        # self.__set_weights_and_masks__(len(fitness[-1]))
        
        if init:
            self.logger.info("(re)initializing weights") 
            self.init_weights(fitness, normspace = normspace, variance_cutoff=variance_cutoff)
            return
        
        wfitness = [ p.fitness for p in accumulator if p.fitness is not None ]
        m2m, minid, maxid = self.compute_correlations(fitness,wfitness)
        old_m2m  = m2m*2
        cnt,st   = 0, time.time()
        while m2m < threshold:
            self.logger.debug(f"KAW: m2m lower than threshold {m2m} < {threshold}: updating weights in #({minid},{maxid})")
            old_m2m  = m2m
            
            self.logger.debug("KAW: Updating and Normalizing wight")
            self.logger.debug("   > weights before update: {self.adaptive_weights}")
            self.logger.debug("   > updating unmasked_weights[{minid}]]={unmasked_weights[minid]]}, unmasked_weights[{maxid}]]={unmasked_weights[maxid]]}")
            self.adaptive_weights[self.unmasked_weights[minid]] *= 1+1.1/float(len(self.unmasked_weights))
            self.adaptive_weights[self.unmasked_weights[maxid]] *= 1-1.0/float(len(self.unmasked_weights))
            self.logger.debug("   > weights after  update: {self.adaptive_weights}")
            self.normalize_weights(normspace = normspace)
            self.logger.debug("   > weights after normalization: {self.adaptive_weights}")
            
            self.logger.debug("KAW: Computing weighted fitness and correlations")
            wfitness = [ p.fitness for p in accumulator if p.fitness is not None ]
            m2m, minid, maxid = self.compute_correlations(fitness,wfitness)
            
            if cnt >= 299 or time.time()-st > 120:
                self.logger.info(f"KAW: does not converge: {cnt} >=299 or {time.time()}-{st} > 120")
                self.logger.info(f"KAW: reset it and recompute weighted fitness") 
                self.init_weights(fitness, normspace = normspace, variance_cutoff=variance_cutoff)
                return
                            
            if old_m2m >= m2m: cnt += 1
            elif cnt   >  0  : cnt -= 1
            self.logger.debug(f"KAW: interation is going #{cnt}")
        self.logger.debug(f"KAW converged")
        self.logger.debug(f'  > adaptive_weights : {self.adaptive_weights}')
        return

    def __init__(self,fitness, weights=[], unmasked=[], maximize=False):
        self.logger           = logging.getLogger(self.__class__.__name__)
        self.fitness          = fitness
        self.maximize         = maximize
        self.adaptive_weights = weights
        self.unmasked_weights = unmasked
    @property
    def fitness(self):
        return self.fitness
    
    @fitness.setter
    def fitness(self, new_fitness:(list,None)):
        if   new_fitness is None:
            self.values = new_fitness
        elif all([ f is not None and bool(np.isfinite(f)) for f in new_fitness ]):
            self.values = new_fitness
        else:
            self.values = None

    @fitness.getter
    def fitness(self):
        if self.values is None:
            return None
        if len(self.adaptive_weights) == 0 or len(self.unmasked_weights) == 0:
            self.__set_weights_and_masks__( len(self.values) )
        try:
            # return float(\
                # np.array(self.fitness)[self.unmasked_weights].dot(np.array(nself.adaptive_weights)[self.unmasked_weights]\
            # )
            return sum([\
                self.adaptive_weights[i]*self.values[i]
                for i in self.unmasked_weights
                ])
        except BaseEventLoop as e:
            self.logger.error(f"Critical Exception: {e}")
            self.logger.error(f" > Values  = {self.values}")
            self.logger.error(f" > Weights = {self.adaptive_weights}")
            self.logger.error(f" > Masks   = {self.unmasked_weights}")
            raise RuntimeError(f"Krayzman - {e} : Values  = {self.values}, Weights = {self.adaptive_weights}, Masks   = {self.unmasked_weights}")
    
        
    def __lt__(self,other):
        if not isinstance(other, Krayzman):
            other = Krayzman(other,weights=self.adaptive_weights,unmasked=self.unmasked_weights)
        if self.maximize:
            if   self.values   is None: return True
            elif other.fitness is None: return False
            return self.fitness >  other.fitness
        else:
            if   self.values   is None: return False
            elif other.fitness is None: return True
            return self.fitness <  other.fitness
    def __le__(self,other):
        if not isinstance(other, Krayzman):
            other = Krayzman(other,weights=self.adaptive_weights,unmasked=self.unmasked_weights)
        if self.maximize:
            if   self.values   is None: return True
            elif other.fitness is None: return False
            return self.fitness >= other.fitness
        else:
            if    self.values  is None: return False
            elif other.fitness is None: return True
            return self.fitness <= other.fitness
    def __gt__(self,other):
        if not isinstance(other, Krayzman):
            other = Krayzman(other,weights=self.adaptive_weights,unmasked=self.unmasked_weights)
        if self.maximize:
            if   self.values   is None: return False
            elif other.fitness is None: return True
            return self.fitness <  other.fitness
        else:
            if   self.values   is None: return True
            elif other.fitness is None: return False
            return self.fitness >  other.fitness
    def __ge__(self,other):
        if not isinstance(other, Krayzman):
            other = Krayzman(other,weights=self.adaptive_weights,unmasked=self.unmasked_weights)
        if self.maximize:
            if   self.values   is None: return False
            elif other.fitness is None: return True
            return self.fitness <= other.fitness
        else:
            if   self.values   is None: return True
            elif other.fitness is None: return False
            return self.fitness >= other.fitness
    def __eq__(self,other):
        if not isinstance(other, Krayzman):
            other = Krayzman(other)
        return self.fitness == other.fitness
    def __ne__(self,other):
        if not isinstance(other, Krayzman):
            other = Krayzman(other)
        return self.fitness != other.fitness
    def __call__(self):
        return self.fitness
    def __str__(self):
        if self.values is None:
            return f"{self.fitness}"
        unmasked_values           = [ self.values[i]       for i in self.unmasked_weights]
        unmasked_adaptive_weights = [ self.adaptive_weights[i] for i in self.unmasked_weights]        
        return f"{self.fitness} = {str(unmasked_values)} x {str(unmasked_adaptive_weights)}"
    def __repr__(self):
        cls                       = self.__class__.__name__
        if self.values is None:
            return f'{cls}({self.fitness})'
        unmasked_values           = [ self.values[i]       for i in self.unmasked_weights]
        unmasked_adaptive_weights = [ self.adaptive_weights[i] for i in self.unmasked_weights]
        return f'{cls}({self.fitness} = {str(unmasked_values)} x {str(unmasked_adaptive_weights)})'
    def dump(self):
        return {
            'values'  : self.values,
            'maximaze': self.maximize,
            'weights' : self.adaptive_weights,
            'unmasked': self.unmasked_weights
        }
    def load(self, dump:dict):
        if not 'values' in dump:
            raise RuntimeError(f'There are not values in the dump:{dump}')
        self.values   = dump['values']
        if not 'weights' in dump:
            raise RuntimeError(f'There are not weights in the dump:{dump}')
        self.maximize = dump['maximaze'] 
        if 'weights' in dump:
            self.adaptive_weights.clear()
            self.adaptive_weights += dump['weights']
        if 'unmasked' in dump:
            self.unmasked_weights.clear()
            self.unmasked_weights += dump['unmasked']

# def KAW_updater(random, population, parents, offspring, args):
def KAW_updater(population, num_generations, num_evaluations, args):
    adap_size = args.get("adaptation_size", None )
    threshold = args.get("threshold"      , 0.05 )
    reinit    = args.get("reinit_after"   , 40   )
    varlimit  = args.get("variance_cutoff", 1e-9 )
    # replacer  = args.get("KAW_replacer"   , generational_replacement)

    if not hasattr(KAW_updater,"reinit_counter"):
        KAW_updater.reinit_counter = reinit
        KAW_updater.weight         = []
        KAW_updater.unmasked       = []

    accumulator =[]
    for p in args["_ec"].population:
        if p is not None and p.fitness is not None:
            if not isinstance(p.fitness,Krayzman):
                p.fitness = Krayzman(p.fitness, weights=KAW_updater.weight[:], unmasked=KAW_updater.unmasked[:])
            accumulator.append(p.fitness)
    
    if adap_size is None:
        if KAW_updater.reinit_counter >= reinit:
            logging.info(f"KAW_updater: Reinit weights after {KAW_updater.reinit_counter} updates")
            accumulator[0].update_weights(accumulator, init=True,variance_cutoff=varlimit)
            KAW_updater.reinit_counter  = 0
        else:
            logging.info(f"KAW_updater: Incrimental update #{KAW_updater.reinit_counter}  hreshold = {threshold}")
            accumulator[0].update_weights(accumulator, init=False,threshold = threshold,variance_cutoff=varlimit)
            KAW_updater.reinit_counter += 1
    else:
        if not hasattr(KAW_updater,"accumulator"):
            KAW_updater.accumulator = []
        if len(KAW_updater.accumulator) < adap_size:
            logging.info(f"KAW_updater: Accumulation phase {len(KAW_updater.accumulator)} < {adap_size}")
            KAW_updater.accumulator += accumulator
        else:
            if KAW_updater.reinit_counter >= reinit:
                logging.info(f"KAW_updater: Grand initialization after {KAW_updater.reinit_counter} updates with candidates size {len(x.accumulator)}")
                accumulator[0].update_weights(KAW_updater.accumulator,init=True,variance_cutoff=varlimit)
                KAW_updater.reinit_counter  = 0
            else:
                logging.info(f"KAW_updater: Incrimental update #{KAW_updater.reinit_counter} with candidates size {len(x.accumulator)} and threshold = {threshold}")
                accumulator[0].update_weights(KAW_updater.accumulator,init=False,threshold = threshold,variance_cutoff=varlimit)
                KAW_updater.reinit_counter += 1
    
    KAW_updater.weight   = accumulator[0].adaptive_weights[:]
    KAW_updater.unmasked = accumulator[0].unmasked_weights[:]
    logging.debug(f"KAW_updater: KAW_updater.weight   = {KAW_updater.weight}")
    logging.debug(f"KAW_updater: KAW_updater.unmasked = {KAW_updater.unmasked}")
    for p in args["_ec"].population:
        if p is None or p.fitness is None: continue
        if not isinstance(p.fitness,Krayzman): continue
        p.fitness.adaptive_weights = KAW_updater.weight[:]
        p.fitness.unmasked_weights = KAW_updater.unmasked[:]

    return
    
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(lineno)-6d%(levelname)-8s:%(message)s', level=logging.DEBUG )
    a,b,c = Krayzman([1,2,3,4,5]), Krayzman([1,3,4,5,6]), Krayzman([1,7,6,5,4])
    print("=== a,b ===")
    print(a.unmasked_weights, a.adaptive_weights)
    print(a < b, a <= b, a == b, a != b ,a >=b, a > b)
    print("== init ==")
    a.update_weights([a,b],init=True)
    print(a.unmasked_weights, a.adaptive_weights)
    print(a.fitness,b.fitness)
    print()
    
    print("== a,b,c ==")
    a.update_weights([a,b,c],init=True)
    print(a.unmasked_weights)
    print(a.adaptive_weights)
    print(a(),b(),c())
    print(a < b, a <= b, a == b, a != b ,a >=b, a > b)
    print(a < c, a <= c, a == c, a != c ,a >=c, a > c)
    print(b < c, b <= c, b == c, b != c ,b >=c, b > c)
    print()

    print("== sorting ==")
    l = [c,a,b]
    for p in sorted(l):
        print(p)
        
    print("== weight update ==")
    print(a.adaptive_weights)
    a.update_weights([a,b,c],threshold = 0.2)
    print(a.adaptive_weights)
    l = [c,a,b]
    for p in sorted(l):
        print(p)

    d = Krayzman([0,2,3,None,5])
    e = Krayzman([0,2,3,np.inf,5])
    print("d.fitness=",d.fitness)
    print("e.fitness=",e.fitness)
    a.update_weights([a,b,c,d,e],threshold = 0.2)
    print(a.adaptive_weights)
    l = [a,b,c,d,e]
    for p in sorted(l):
        print(p)
