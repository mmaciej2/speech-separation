#!/bin/bash

if [ $# -le 1 ]; then
  echo "Usage:"
  echo "$0 <scp-file> <dir-out>"
  echo "optional arguments:"
  echo " --bwlimit"
  exit 1;
fi

scp=$1
dir_out=$2
shift 2

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
rsync $opts --files-from=<(cut -d' ' -f2 $scp) / $dir_out/
