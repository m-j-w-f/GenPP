#!/bin/bash

IDS=(
    # LNGM (Running)
    # Engression (Running)
    #"7ww3tj56"
    # "gpdka5qd"
    # "kf7uayf6"
    # "5jjbvdex"
    # "3eevjkfj"
    # "euak9uee"
    # "2ajwxmir"
    # "3j5g7ils"
    # FM UViT (Running)
    #"oddm8ydj"
    #"2t98jag4"
    # FM UNET
    "fmz08y1j"
    "f5yyzzxf"

)

python src/genpp/eval/permutation_importance.py --run-path feik/genpp/f5yyzzxf -v --device 1 --n-repeats 1 --batch-size 16 --channels 4

for ID in "${IDS[@]}"; do
    SCREEN_NAME="permutation_${ID}"
    CMD="pixi run -e gpu python src/genpp/eval/permutation_importance.py --run-path feik/genpp/${ID} -v --device 1 --n-repeats 1 --batch-size 16 --channels 1 0 62 58 59 7 6 28 5 2"

    echo "Launching screen '${SCREEN_NAME}' for ID: ${ID}"
    screen -dmS "${SCREEN_NAME}" bash -c "${CMD}; exec bash"
done

echo "Done! launched ${#IDS[@]} screen sessions."
echo "Use 'screen -ls' to list sessions, 'screen -r <name>' to attach."
