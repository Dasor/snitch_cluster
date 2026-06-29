#!/bin/bash
# run a gemm executable and extract timing information from its verilator logs
# this script takes as arguments
# buildDir
# extractKernelTime // path to python script
# expName
# logs
# M
# N
# K
# m
# n
# k

buildDir=$1
extractKernelTime=$2
expName=$3
logs=$4
M=$5
N=$6
K=$7
m=$8
n=$9
k=${10}
here=${11}
elf=$(basename $gemmDir)



_kill_tree(){
    local pid=$1
    local child grandchild
    child=$(pgrep -P $pid)
    if [ -n "$child" ]; then
        grandchild=$(pgrep -P $child)
        [ -n "$grandchild" ] && kill -9 $grandchild 2>/dev/null
        kill -9 $child 2>/dev/null
    fi
    kill -9 $pid 2>/dev/null
}

timeout(){
    # Start verify.py with cycle-limit disabled (--myrtleTimeout=0);
    # wall-clock enforcement is handled by the elapsed counter below.
    $gemmDir/scripts/verify.py snitch_cluster.vlt $elf.elf "--myrtleTimeout=0" > verify-output.txt &
    PID_A=$!
    echo "Started Process A (PID: $PID_A)"
    elapsed=0

    while true; do
        # Process finished on its own
        if ! kill -0 $PID_A 2>/dev/null; then
            wait $PID_A
            EXIT_STATUS=$?
            break
        fi

        # C++ cycle-limit timeout (legacy path, only fires if --myrtleTimeout > 0)
        if grep -q "Myrtle Experiment Timeout" output.txt 2>/dev/null; then
            echo "Cycle-limit timeout detected! Cleaning up process tree..."
            _kill_tree $PID_A
            EXIT_STATUS=1
            echo "simulation failed on timeout." > verify-output.txt
            rm -rf "logs" "dma_trace_00008_00000.log"
            break
        fi

        # Wall-clock timeout
        if [[ ${TIMEOUT:-0} -gt 0 ]] && [[ $elapsed -ge ${TIMEOUT} ]]; then
            echo "Wall-clock timeout after ${TIMEOUT}s! Cleaning up process tree..."
            _kill_tree $PID_A
            EXIT_STATUS=1
            echo "simulation failed on timeout." > verify-output.txt
            # Write the sentinel that combineTilingSchemeDataIntoSingleCSV.py looks for
            echo "what():  Myrtle Experiment Timeout" >&2
            rm -rf "logs" "dma_trace_00008_00000.log"
            break
        fi

        sleep 1
        elapsed=$((elapsed + 1))
    done

    if [ $EXIT_STATUS -eq 0 ]; then
        echo "ran simulation correctly"
    else
        echo "error or timeout while running simulation"
    fi
}

genTrace(){
    gen_trace="$here/util/trace/gen_trace.py"
    llvm_mc="/tools/riscv-llvm/bin/llvm-mc"
    logsDir="$1" # absolute path of the logs directory
    logWOFE="$2" # log file without the .dasm file extension
    dma="$3"
    echo "logsDir is $logsDir and log file is $logWOFE and dma is $dma"
    if [[ "$dma" != "dma" ]]; 
    then
        $gen_trace "$logsDir/$logWOFE.dasm" --mc-exec $llvm_mc --mc-flags "-disassemble -mcpu=snitch" --dma-trace "$logsDir/$logWOFE.log" --dump-hart-perf "$logsDir/hart-$logWOFE-perf.json" --dump-dma-perf "$logsDir/dma-$logWOFE-perf.json" -o "$logsDir/$logWOFE.txt"
    else
        $gen_trace "$logsDir/$logWOFE.dasm" --mc-exec $llvm_mc --mc-flags "-disassemble -mcpu=snitch" --dump-hart-perf "$logsDir/hart-$logWOFE-perf.json" -o "$logsDir/$logWOFE.txt"
    fi
    return 0
}

main(){

    echo -e "\trun_and_extract_time.sh: Arguments passed in are $buildDir $extractKernelTime $expName $logs $M $N $K $m $n $k"

    cd $buildDir
    #correct=$($gemmDir/scripts/verify.py snitch_cluster.vlt $elf.elf --myrtleTimeout=5 > verify-output.txt; echo $?)
    timeout
    correct=$(echo $?)
    if [[ "$correct" != "0" ]]; 
    then
        echo -e "\trun_and_extract_time.sh: Simulation failed, so skipping export step. $correct"
        rm -rf "dma_trace_00008_00000.log"
        # delete huge log files
        cd $logs
        ls -l -h *.dasm
        rm -rf *.dasm
        rm -rf *.txt
        return 1
    fi

    # extract timing info: only process DMA hart, skip compute cores
    gen_trace="$here/util/trace/gen_trace.py"
    llvm_mc="/tools/riscv-llvm/bin/llvm-mc"
    e2e_cycles=$(
        $gen_trace "logs/trace_hart_00008.dasm" \
            --mc-exec $llvm_mc --mc-flags "-disassemble -mcpu=snitch" \
            -o /dev/null
    )
    rm -f logs/trace_hart_00008.dasm
    rm -f logs/trace_hart_0000[0-7].dasm
    python $extractKernelTime $expName $logs $M $N $K $m $n $k "$e2e_cycles"
    correct=$(echo $?)
    if [[ "$correct" == "0" ]];
    then
        echo -e "\trun_and_extract_time.sh: Successfully exported timing information. Deleting logs..."
        # .dasm files already deleted above; clean up any other leftovers
        cd $logs
        rm -f *.dasm *.txt
        cd ..
        rm -rf "dma_trace_00008_00000.log"
    else
        echo -e "\trun_and_extract_time.sh: Error exporting timing info!"
    fi
}

main

