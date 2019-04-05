#!/usr/bin/env bash

printf "Running Data Collection. These will take some time, so they'll run in the background."
printf "Use 'top' to identify process IDs and kill <processID> to stop a process.\n"

printf "Some data may differ slightly from the data in my pdf, since I ran it with more detail over a longer amount of time.\n"

printf "\nComputing adaBoost. Results will spit into Results/adaBoost.txt. \nThis will take time.\n"
python3 EnsembleMain.py adaBoost > ../Results/adaBoost.txt &

printf "\nComputing Bagged Trees Data. Results will spit into Results/bag.txt. \nThis will take... time.\n"
python3 EnsembleMain.py bag > ../Results/bag.txt &

printf "\nComputing Random Forest Data. Results will spit into Results/forest.txt. \nThis will take time. Like, four hours or more.\n"
python3 EnsembleMain.py forest > ../Results/forest.txt &

printf "\nComputing that huge bias / variance calculation. Results will spit into Results/biases.txt. \nThis will take four hours or more.\n"
python3 EnsembleMain.py somuchstuff > ../Results/biases.txt


printf "\nRunning Linear Regression Data. Output in Results/regression.txt\n"
python3 Regression.py > ../Results/regression.txt &
