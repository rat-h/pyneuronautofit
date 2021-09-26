from numpy import *
import json, gzip, os
from optparse import OptionParser

oprs = OptionParser("USAGE: %prog [flags] [flags] input.json[.gzip] ouput.json")
# oprs.add_option("-0", "--off-header",       dest="header",default=True,            action="store_false",
    # help="Off header skipping (for backward compatibility with old arXives")
oprs.add_option("-W", "--weight",         dest="weight",default=False,            action="store_true",
    help="Scale by STD")
oprs.add_option("-s", "--scale-by-target", dest="wbt",   default=False,            action="store_true",
    help="Scale scores by target values. Works only with header.")
oprs.add_option("-A", "--sum-abs-scores",  dest="sas",   default=False,            action="store_true",
    help="sum unweighted scores.")
oprs.add_option("-F", "--fitness-scores",  dest="fas",   default=False,            action="store_true",
    help="fitness is the score.")
oprs.add_option("-M", "--sum-max-norm",    dest="smn",   default=False,            action="store_true",
    help="sum scores scaled by maximal value.")
oprs.add_option("-x", "--scores-first",    dest="sf",    default=False,            action="store_true",
    help="score is the first list in each entrance.")

oprs.add_option("-S", "--off-sorting",       dest="sort", default=True,            action="store_false",
    help="skip sorting archive")
oprs.add_option("-r", "--road-map",         dest="rmap",  default=None,            type='str',
    help="save csv roadmap to this file" )

opt, args = oprs.parse_args()

if len(args) != 2:
    print(f"USAGE {sys.argv[0]} [flags] input.json[.gzip] ouput.json")
    print(f"Run {sys.argv[0]} -h for more options")
    exit(1)

def read_arxive(fd):
    arx = json.load(fd)
    if type(arx) is list:
        if not type(arx[0][-1]) is list and not type(arx[1][-1]) is list:
            markers = arx[0]
            targets = arx[1]
            arXive  = arx[2:]
        else:
            arXive  = arx
            markers = None
            targets = None
        if len(arXive[0]) == 3:
            arXive = [ [s,v] for s,_,v in arXive ] if opt.sf else [ [s,v] for _,s,v in arXive ]
        elif len(arXive[0]) != 2:
            sys.stderr.write(f"Unsupported size of rec  {len(arXive[0])}")
            exit(1)
        parameters = None
        version    = None
        cmd        = None
    elif type(arx) is dict:
        markers    = arx['markers']    if 'markers'    in arx else None
        targets    = arx['bvalues']    if 'bvalues'    in arx else None
        parameters = arx['parameters'] if 'parameters' in arx else None
        version    = arx['version']    if 'version'    in arx else None
        cmd        = arx['cmd']        if 'cmd'        in arx else None

        arXive  = [
                [ p['fitness'],p['parameters'] ]
                for r in 'final records unique'.split() if r in arx
                for p in arx[r] if not p is None
        ]
    else:
        sys.stderr.write(f"Wrong format of arXive {type(arx)}\n")
        exit(1)
    return arXive,markers,targets,parameters,version,cmd
    
"""

"""
    
fname, fext = os.path.splitext(args[0])

if   fext == ".gz":
    print( "=========================")
    print(f"Reading GZIP {args[0]}")
    with gzip.open(args[0],'r') as fd:
        arXive,markers,targets,\
        parameters,version,cmd = read_arxive(fd)
    print( "==================== DONE")
elif fext == ".json":
    print( "=========================")
    print(f"Reading JSON {args[0]}")
    with open(args[0],'r') as fd:
        arXive,markers,targets,parameters,version,cmd = read_arxive(fd)
    print( "==================== DONE")
else:
    sys.stderr.write(f"Unknown input file extension {fext}")
    exit(1)


recv=2
vectors = array([ v for s,v in arXive ])

u,idx  = unique(vectors,axis=0,return_index=True)
UniqueArXive = [arXive[i] for i in idx.astype(int) ]

if not (opt.weight or opt.sas or opt.smn or opt.fas): opt.weight = True
    
if opt.weight:
    print("Computing var/score weights scales")
    scores  = array([ s for s,v in UniqueArXive ]) 
    if opt.wbt:
        if targets is None:
            sys.stderr.write(f"Need am arXive with header to get target values (-0 -A does NOT work)")
            exit(1) 
        weights = array([ x if x != 0 else 1. for x in targets])
    else:
        weights = var(scores,axis=0)
        weights[where(weights<1e-9)] = mean(weights)
    weights = 1./weights.T
    wscores = dot(scores,weights)
    UniqueArXive = [ (w,s,v) for w,(s,v) in zip(wscores,UniqueArXive) ]
    recv = 3
elif opt.fas:
    print("No scaler just fitness")
    UniqueArXive =  [ (s,s,v) for s,v in UniqueArXive ]
    recv = 3
elif opt.sas:
    print("Computing sum sores scales")
    UniqueArXive =  [ (sum(s),s,v) for s,v in UniqueArXive ]
    recv = 3
elif opt.smn:
    print("Computing max scales")
    scores  =  array([ s for s,v in UniqueArXive ]) 
    maxup = amax(scores,axis=0)
    maxup[where(maxup<1e-12)] = 1
    print("Max scales")
    print(f"  > {maxup.tolist()}")
    UniqueArXive =  [ (sum(s/maxup),s,v) for s,v in UniqueArXive ]
    print( "==================== DONE")

if opt.sort: UniqueArXive = sorted(UniqueArXive)

print(len(arXive),"=>",len(UniqueArXive))

fname, fext = os.path.splitext(args[1])

def saveJson(fd):
    fd.write('{\n')
    fd.write("\t\"markers\"   :" + json.dumps(markers)   +",\n")
    fd.write("\t\"bvalues\"   :" + json.dumps(targets)   +",\n")
    fd.write("\t\"parameters\":" + json.dumps(parameters)+",\n")
    fd.write("\t\"version\"   :" + json.dumps(version)   +",\n")
    fd.write("\t\"cmd\"       :" + json.dumps(cmd)       +",\n")
    fd.write("\t\"unique\"    : [\n")
    for w,s,p in UniqueArXive:
        fd.write("\t\t"+json.dumps({'fitness':s,'parameters':p,'weighted-fitness':w})+",\n")
    fd.write("\t\tnull\n\t]\n}\n")
            

if   fext == ".json":
    with open(args[1],"w") as fd:
        saveJson(fd)
elif fext == ".gz":
    with gzip.open(args[1],"wt") as fd:
        saveJson(fd)
elif fext == ".npz":
    UniqueArXive = [ (w,array(s),array(v)) for w,s,v in UniqueArXive ] if recv == 3 else [ (array(s),array(v)) for s,v in UniqueArXive ]
    savez(args[1],
        markers = markers,
        bvalues = targets,
        parameters = json.dumps(parameters),
        version = version,
        cmd     = cmd,
        arXive  = UniqueArXive
    )

if not opt.rmap is None:
    with open(opt.rmap,"w") as fd:
        if not markers is None: fd.write("Markers,Weights,"+",".join(markers)+"\n")
        if not targets is None: fd.write("Targets,       ,"+",".join([f"{o:0.6g}" for o in targets])+"\n")
        for i,(w,s,v) in enumerate(UniqueArXive):
            fd.write(f"{i},{w:0.2f},"+",".join([f"{o:0.6g}" for o in s])+"\n")
