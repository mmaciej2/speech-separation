#!/bin/bash

if [ $# -le 1 ]; then
  echo "Usage:"
  echo "$0 <scp-file> <dir-out>"
  echo "optional arguments:"
  echo " --bwlimit"
  echo " --find-sources <false>"
  exit 1;
fi

scp=$1
dir_out=$2
shift 2
find_sources=false

# Parse optional arguments
while true; do
  [ -z "${1:-}" ] && break;
  case "$1" in
    --*) name=$(echo "$1" | sed 's/--//g' | sed 's/-/_/g')
      printf -v $name "$2"
      shift 2
      ;;
    *) echo "ERROR: malformed arguemnts"
      exit 1
      ;;
  esac
done

opts=

[ -z "$bwlimit" ] || opts="$opts --bwlimit=$bwlimit"

mkdir -p $dir_out
if [ "$find_sources" == true ]; then
  rsync $opts --files-from=<(cut -d' ' -f2 $scp | sed -e 's/\/mix\//\/*\//g' | while read line; do find $line; done) / $dir_out/
else
  rsync $opts --files-from=<(cut -d' ' -f2 $scp) / $dir_out/
fi
