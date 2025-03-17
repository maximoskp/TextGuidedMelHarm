#!/bin/bash

scripts=(
    "train_text_bart_base.py -t ChordSymbolTokenizer -m specific_chord -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t ChordSymbolTokenizer -m chord_root -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t ChordSymbolTokenizer -m pitch_class -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 2 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t RootTypeTokenizer -m specific_chord -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t RootTypeTokenizer -m chord_root -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t RootTypeTokenizer -m pitch_class -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 2 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t PitchClassTokenizer -m specific_chord -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t PitchClassTokenizer -m chord_root -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t PitchClassTokenizer -m pitch_class -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 2 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t RootPCTokenizer -m specific_chord -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t RootPCTokenizer -m chord_root -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 100 -l 5e-5 -b 4"
    "train_text_bart_base.py -t RootPCTokenizer -m pitch_class -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 2 -e 100 -l 5e-5 -b 4"
)

# Name of the conda environment
conda_env="torch"

# Loop through the scripts and create a screen for each
for script in "${scripts[@]}"; do
    # Extract the base name of the script (first word) to use as the screen name
    screen_name=$(basename "$(echo $script | awk '{print $1}')" .py)
    
    # Start a new detached screen and execute commands
    screen -dmS "$screen_name" bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh;  # Update this path if your conda is located elsewhere
        conda activate $conda_env;
        python $script;
        exec bash
    "
    echo "Started screen '$screen_name' for script '$script'."
done
