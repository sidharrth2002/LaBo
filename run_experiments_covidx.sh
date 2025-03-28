#!/bin/bash

# Kill all subprocesses on Ctrl+C
trap "echo 'Interrupted. Killing all jobs...'; kill 0; exit 1" SIGINT

shots=("1" "2" "4" "8" "16")
max_jobs=2
job_count=0

for shot in "${shots[@]}"; do
    sh labo_train.sh "$shot" COVIDX_moe &
    ((job_count++))

    if [[ $job_count -ge $max_jobs ]]; then
        wait -n
        ((job_count--))
    fi
done

# Wait for all remaining jobs to finish
wait