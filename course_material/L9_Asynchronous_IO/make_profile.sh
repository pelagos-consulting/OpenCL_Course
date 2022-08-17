#!/bin/bash

export taudir=./tau

export TAU_TRACE=1
export TRACEDIR=$taudir
export PROFILEDIR=$taudir

rm -rf $taudir
mkdir -p $taudir

tau_exec -T serial -opencl ./wave2d_async.exe
cd $taudir; echo 'y' | tau_treemerge.pl
tau_trace2json ./tau.trc ./tau.edf -chrome -ignoreatomic -o trace.json
