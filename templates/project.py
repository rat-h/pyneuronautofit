"""
project directory structure
/{project}{prefix}{gversion}{postfix}         - a directory which is zipped to submit for NSG or move to HPC
   |- @../pyneuronautofit                     - links to files in parent directory
   |- @../project.py
   |- @../xxx.py
   |- @../xx.abf
   ....
/versions                                     - directory with project versions
   |-/v{prefix}{gversion}{postfix}            - **WILL BE CREATED BY THIS SCRIPT** directory for results obtained with given version
   |-{project}{prefix}{gversion}{postfix}.zip - **WILL BE CREATED BY THIS SCRIPT** zipped archive with the version for NSG 
/@pyneuronautofit                          - a link to autofit module (if module wasn't installed)
project.py                                 - file from this template
xxx.py                                     - neuron model
xxx.abf                                    - data files to fit
"""

from XXX import L5neuron as fitmeneuron   # importing neuron for fitting
from XXX import param_nslh                # parameters list each entry is (name,scale,lowest,highest)

project   = 'dLGN-TC-fit-v'
gversion  = '0.00'
prefix    = ''
postfix   = ''
simulator = 'neuron'

if __name__ == "__main__":
    import sys,os
    cmd = ""
    oldgversion = gversion
    with open(sys.argv[0],"r") as fd:
        for il,l in enumerate(fd.readlines()):
            if l[:len("gversion")] == "gversion": continue
            cmd += l
    gversion = [ int(m) for m in  gversion.split(".") ]
    gversion[-1] += 1
    gversion =".".join(["{{:0{}d}}".format(i+1).format(m) for i,m in enumerate(gversion) ])
    with open(sys.argv[0],"w") as fd:
        fd.write(f"gversion = \'{gversion}\'\n\n"+cmd)
    os.system(f"git commit version.py -m \'New version {prefix}{gversion}{postfix}\'")
    os.system(f"git tag v{prefix}{gversion}{postfix}")
    os.system(f"mv {project}* {project}{prefix}{gversion}{postfix}")
    os.system(f"mkdir versions/v{prefix}{gversion}{postfix}")
    os.system(f"zip -r versions/{project}{prefix}{gversion}{postfix}.zip dLGN-TC-fit-v{prefix}{gversion}{postfix}")
    
def getversion():
    return f"{prefix}{gversion}{postfix}"
