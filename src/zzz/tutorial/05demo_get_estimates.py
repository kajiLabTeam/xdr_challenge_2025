#! /usr/bin/env -S python3 -O
#! /usr/bin/env -S python3

import os
import sys
import lzma
import requests
import time
from parse import parse
import yaml
import pandas as pd

from evaalapi import statefmt, estfmt

server = "http://127.0.0.1:5000/evaalapi/"
trialname = "onlinedemo"


def do_req (req, n=2):
    r = requests.get(server+trialname+req)
    print("\n==>  GET " + req + " --> " + str(r.status_code))
    if False and r.headers['content-type'].startswith("application/x-xz"):
        l = lzma.decompress(r.content).decode('ascii').splitlines()
    else:
        l = r.text.splitlines()
    if len(l) <= 2*n+1:
        print(r.text + '\n')
    else:
        print('\n'.join(l[:n]
                        + ["   ... ___%d lines omitted___ ...   " % len(l)]
                        + l[-n:] + [""]))
    
    return(r)


def demo (maxw, output_csv):
    ## Get estimates
    r = do_req("/estimates", 3)
    result = []
    for l in r.text.splitlines()[2:]: # ignore first sample (given origin)
        print(l)
        s = parse(estfmt, l); 
        x, y, yaw = s.named["pos"].split(",")
        result.append({"timestamp" : s.named["pts"], "x": x, "y": y, "yaw": yaw})
    df = pd.DataFrame(result)
    print(df)

    df.to_csv(output_csv, index=False)

    ## Get log
    time.sleep(maxw)
    r = do_req("/log", 12)

    ## We finish here
    print("Demo stops here")

################################################################

if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print("""A demo for the EvAAL API.  Usage is
%s [trial] [server]

if omitted, TRIAL defaults to '%s' and SERVER to %s""" %
              (sys.argv[0], trialname, server))
    else:
        trialname = sys.argv[1]
        server = sys.argv[2]
        output_csv = sys.argv[3]

    print("# Running %s demo test suite\n")
    print(f"trial: {trialname}, server: {server}")
    maxw = 0.5
    demo(maxw, output_csv)
    exit(0)

