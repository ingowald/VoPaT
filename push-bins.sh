#!/bin/bash
#cp ~/Projects/brix/bin/brixViewer ~/cluster-bin/
#cp ./brixViewer ./brixOffline  ~/cluster-bin/
#for f in mog snert shadow hank; do
for f in wally shady hasky hanky; do
#for f in hasky moggy snorty shady hanky wally; do
#     walter
#    rsync -avz ~/cluster-bin/brixViewer ~/cluster-bin/brixOffline $f:cluster-bin/
    rsync -avz /home/wald/opt $f:
done
