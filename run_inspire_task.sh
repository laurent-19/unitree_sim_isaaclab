#!/bin/bash

# Usage: ./run_inspire_task.sh [task_name]
# Available tasks: cylinder, redblock, stack
# Example: ./run_inspire_task.sh redblock

TASK=${1:-cylinder}

case $TASK in
    cylinder)
        TASK_NAME="Isaac-PickPlace-Cylinder-G129-Inspire-Joint"
        echo "Running: Pick & Place Cylinder (Inspire Hand)"
        ;;
    redblock)
        TASK_NAME="Isaac-PickPlace-RedBlock-G129-Inspire-Joint"
        echo "Running: Pick & Place Red Block"
        ;;
    stack)
        TASK_NAME="Isaac-Stack-RgyBlock-G129-Inspire-Joint"
        echo "Running: Stack RGY Block"
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available tasks: cylinder, redblock, stack"
        exit 1
        ;;
esac

python sim_main.py --device cuda:0 --enable_cameras --task $TASK_NAME --robot_type g129 --enable_inspire_dds


