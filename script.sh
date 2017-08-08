#!/bin/bash


# This script executes all the possible preprocessor options in the Python3 file.



# Function excecutes all preprocessor options of the Python3 script
Execute()
{
	echo "############################################################"
	echo "Executing Python3 script"
	echo -e "############################################################\n"

	# iterate all preprocessor options
	for i in $(seq 0 5); do	
		python3 project.py weather.zip katkam-secret-location.zip ${i} 
	done
	echo -e "Script Terminates\n"
}


# run script
Execute
