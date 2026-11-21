import json
import logging

class arXiver:
    def __init__(self,
            filename:str,
            algorithm:str,
            markers,
            bvalues,
            parameters,
            version,
            target,
            evaluation,
            cmd,
            logscale:bool):
        self.filename   = filename
        self.algorithm  = algorithm
        self.mm         = self.algorithm[0] != "N" or self.algorithm[0] != "K"
        self.parameters = parameters
        with open(self.filename,'w') as fd:
            fd.write('{\n')
            fd.write("\t\"markers\"   :" + json.dumps(markers)    +",\n")
            fd.write("\t\"bvalues\"   :" + json.dumps(bvalues)    +",\n")
            fd.write("\t\"parameters\":" + json.dumps(parameters) +",\n")
            fd.write("\t\"log scale\" :" + json.dumps(logscale)   +",\n")
            fd.write("\t\"version\"   :" + json.dumps(version)    +",\n")
            fd.write("\t\"target\"    :" + json.dumps(target)     +",\n")
            fd.write("\t\"evaluation\":" + json.dumps(evaluation) +",\n")
            fd.write("\t\"cmd\"       :" + json.dumps(cmd)+",\n")
            fd.write(f"\t\"records\":[\n")

    def __call__(self,wfitness,fitness,candidates,args):
        with open(self.filename,"a") as fd:
            fd.write("\t\t[\n")
            for w,f,p in zip(wfitness,fitness,candidates):
                fd.write("\t\t\t"+json.dumps({"fitness":f, "parameters":p, "weighted-fitness":w})+ ",\n")
            fd.write(f"\t\t\t{json.dumps(None)}\n\t\t],\n")


    def close(self):
        with open(arXive,'a') as fd:
            fd.write("\t\tnull\n\t]\n}\n")

class charly:
    def __init__(self,
            filename:str,
            fname:str, 
            parameters:list,
            algorithm:str,
            riter:int,
            logscale:bool,
            evalate:dict,
            target:str,
            checkpoint_print:int=-1,
            checkpoint_count:int=-1):
        self.filename         = filename
        self.fname            = fname
        self.parameters       = parameters
        self.algorithm        = algorithm
        self.riter            = riter
        self.logscale         = logscale
        self.evalate          = evalate
        self.target           = target
        self.checkpoint_print = checkpoint_print
        self.checkpoint_count = checkpoint_count
        self.chcount          = -1
        self.mm               = self.algorithm[0] != "N" or self.algorithm[0] != "K"
        self.logger           = logging.getLogger("Checkpoint")
        
    def __call__(self,wfitness,fitness,candidates,args):
        self.chcount += 1
        xpop = sorted([ 
            [w,f,c] for w,f,c in zip(wfitness,fitness,candidates)
        ])
        mpop = sorted([ 
            [ c.fitness if self.mm else c.fitness.values,c.candidate]
            for c in args["_ec"].archive 
        ])
        ppop = sorted([ 
            [ c.fitness if self.mm else c.fitness.values,c.candidate]
            for c in args["_ec"].population 
        ])
        self.logger.debug(f"--- Check Point ---")
        self.logger.debug(f" > counter : {self.chcount}")
        if self.checkpoint_count > 0:
            self.logger.debug(f" > rev : {self.chcount}%{self.checkpoint_count} = {self.chcount%self.checkpoint_count}")
            if self.chcount%self.checkpoint_count == 0   :
                with open(self.filename,"w") as fd:
                    fd.write("{\n\t\"checkpoint id\":"+f"{self.chcount},\n")
                    fd.write("\t\"parameters\":"+f"{json.dumps(self.parameters)},\n")
                    fd.write("\t\"log scale\" :"+f"{json.dumps(self.logscale)},\n")
                    fd.write("\t\"evaluate\"  :"+f"{json.dumps(self.evalate)},\n")
                    fd.write("\t\"target\"    :"+f"{json.dumps(self.target)},\n")
                    fd.write("\t\"checkpoint\":[\n")
                    for il,(w,f,p) in enumerate(xpop):
                        fd.write("\t\t"+json.dumps({"fitness":f, "parameters":p, "weighted-fitness":w })+(",\n" if il < len(xpop)-1 else "\n") )
                    fd.write("\t],\n\t\"population\":[\n")
                    for il,(f,p) in enumerate(mpop):
                        fd.write("\t\t"+json.dumps({"fitness":f, "parameters":p})+(",\n" if il < len(mpop)-1 else "\n") )
                    fd.write("\t],\n\t\"archive\":[\n")
                    for il,(f,p) in enumerate(mpop):
                        fd.write("\t\t"+json.dumps({"fitness":f, "parameters":p})+(",\n" if il < len(mpop)-1 else "\n") )
                    fd.write("\t]\n}\n")
                with open(f'{self.fname}.link','w') as fd:
                    json.dump({
                        "check point"     : self.filename,
                        "restart counter" : self.riter
                    },fd, indent=4)
        if self.checkpoint_print > 0:
            if self.chcount%self.checkpoint_print == 0  :
                print(json.dumps(xpop, indent=4)+"\n")
