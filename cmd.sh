case $(hostname -d) in
  clsp.jhu.edu)
    export cpu_cmd="-l ram_free=8G,mem_free=8G"
    export train_cmd="-l hostname=c0*&!c06*&!c09*,ram_free=8G,mem_free=8G"
    export eval_cmd="-l hostname=c0*&!c06*&!c09*,ram_free=8G,mem_free=8G,h_rt=48:00:00"
    ;;
  *)
    export cpu_cmd="-l ram_free=8G,mem_free=8G"
    export train_cmd="-l ram_free=8G,mem_free=8G"
    export eval_cmd="-l ram_free=8G,mem_free=8G,h_rt=48:00:00"
    ;;
esac
