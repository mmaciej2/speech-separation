case $(hostname -d) in
  clsp.jhu.edu)
    export cpu_cmd="-l ram_free=8G,mem_free=8G"
    export train_cmd="-l hostname=c0*&!c06*&!c09*,ram_free=8G,mem_free=8G"
    export eval_cmd="-l hostname=c0*&!c06*&!c09*,ram_free=8G,mem_free=8G,h_rt=48:00:00"
    ;;
  cm.gemini)
    export run_cmd="qsub -cwd -sync y -q all.q -l mem_free=4G,num_proc=1,h_rt=48:00:00"
    export run_cmd_nosync="qsub -cwd -q all.q -l mem_free=2G,num_proc=1,h_rt=48:00:00"
    export cpu_cmd="-cwd -q all.q -l mem_free=2G,num_proc=1,h_rt=48:00:00"
    export train_cmd="-q gpu.q@@1080 -l gpu=1,mem_free=10G,h_rt=72:00:00,num_proc=1"
    export eval_cmd="-q gpu.q@@1080 -l gpu=1,mem_free=10G,h_rt=4:00:00,num_proc=1"
    export eval_all_cmd="-q gpu.q -l gpu=1,mem_free=10G,h_rt=4:00:00,num_proc=1"
    ;;
  *)
    export cpu_cmd="-l ram_free=8G,mem_free=8G"
    export train_cmd="-l ram_free=8G,mem_free=8G"
    export eval_cmd="-l ram_free=8G,mem_free=8G,h_rt=48:00:00"
    ;;
esac
