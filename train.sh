#!/bin/bash

# while true; do
    python3 dreamer.py --configs dmc_vision --task dmc_humanoid_run --logdir ./logdir/dmc_humanoid_run
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Script finished successfully. Exiting."
        break
    else
        echo "Script crashed with exit code $EXIT_CODE. Restarting..."
        sleep 1  # Short pause before restarting (optional)
    fi
# done
