#!/bin/bash

datadir=$1

srcfile=$datadir/utt2num_spk

[ -f $srcfile ] || exit 1;

awk -F'[_ ]' '{for(i=1; i<NF-1; ++i){printf("%s_", $i)}; printf("%s ", $(NF-1)); for(i=1; i<$NF; ++i){ind=i*9-8; printf("%s ", $ind)}; ind=$NF*9-8; printf("%s\n",$ind)}' $srcfile > $datadir/utt2spk
