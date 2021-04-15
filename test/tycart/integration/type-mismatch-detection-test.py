#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:47:12 2020

@author: mority
"""

import subprocess
import os
from os import remove
from shutil import copyfile
import time

type_set = [	"char",\
			"short int",\
			"int",\
			"long int",\
			"float", "double"]

uuts = ["mini-cpr/gol/", "veloc/gol/"]

def main():
	t_start = time.time()
	num_tests = 0
	num_build = 0
	num_clean = 0
	num_pass = 0
	num_false_positive = 0
	failures = []
	false_positive = False

	for uut in uuts:
		print("\nBeginning type-mismatch tests of TY_protect macro with " + uut)
		for actual_type in type_set:
			for expected_type in type_set:
				print("\nActual Type: " + actual_type + ", Expected Type: " + expected_type)

				# set false positve mode
				false_positive = False
				if(actual_type == expected_type):
					false_positive = True
					print("Actual and expected type are equal, checking for false positive")

				# copy original make file
				copyfile(uut + "Makefile", uut + "MakefileTesting")

				# prepend preprocessor macros to make file copy
				with open(uut + "MakefileTesting", "r+") as file:
					content = file.read()
					file.seek(0, 0)
					file.write('ACTUAL_TYPE = -DACTUAL_TYPE="' + actual_type + '"\n')
					file.write('EXPECTED_TYPE = -DEXPECTED_TYPE="' + expected_type + '"\n')

					if(uut == "fti/gol/"):
						actual_type_fti = actual_type.replace(" ", ", ") # adds comma before whitespace to make it work with the fti verison of the macro
						expected_type_fti = expected_type.replace(" ", ", ")
						file.write('ACTUAL_TYPE_FTI = -DACTUAL_TYPE_FTI="' + actual_type_fti + '"\n')
						file.write('EXPECTED_TYPE_FTI = -DEXPECTED_TYPE_FTI="' + expected_type_fti + '"\n')

					file.write(content)
					file.close()


				# execute make file copy
				print("Building...")
				exitcode, output = subprocess.getstatusoutput("make -C " + uut + " -f MakefileTesting")
				if exitcode:
					msg = "Build failed"
					print(output)
					failures.append([uut, actual_type, expected_type, msg, exitcode, output])
					print(msg)
				else:
					num_build += 1
					print("Build successful")

					# change cwd to where bin is
					os.chdir(uut)

					# run the binary and assign output to file-like object
					cmd = 'gol-templ'
					if uut == 'veloc/gol/':
						cmd = 'mpirun -np 1 ' + cmd
					elif uut == 'fti/gol/':
						cmd = 'mpirun -np 4 ' + cmd
					elif uut == 'mini-cpr/gol/':
						cmd = './' + cmd

					print("Executing binary...")
					exitcode, output = subprocess.getstatusoutput(cmd)

					# check if expected output occured
					if exitcode and not false_positive:
						if "Assert failed: Expected type" in output:
							num_pass += 1
							print("Type-mismatch detected, Test passed")
						else:
							msg = "Expected failed assertion but execution terminated  abnormally for some other reason"
							failures.append([uut, actual_type, expected_type, msg, exitcode, output])
							print(msg)
					elif exitcode and false_positive:
						if "Assert failed: Expected type" in output:
							num_false_positive += 1
							msg = "False positive occured"
							failures.append([uut, actual_type, expected_type, msg, exitcode, output])
						else:
							msg = "Execution terminated abnormally for some other reason"
							failures.append([uut, actual_type, expected_type, msg, exitcode, output])
							print(msg)
					elif not exitcode and not false_positive:
						msg = "type-mismatch not detected"
						failures.append([uut, actual_type, expected_type, msg, exitcode, output])
						print(msg)
					elif not exitcode and false_positive:
						num_pass += 1
						print("Test passed")

					# reset cwd
					os.chdir("../..")

				# run make clean
				exitcode, output = subprocess.getstatusoutput("make -C " + uut + " -f MakefileTesting clean")
				if exitcode:
					print("Clean up failed")
				else:
					num_clean += 1
					print("Clean up complete")

				# delete make file copy
				remove(uut + "MakefileTesting")
				num_tests += 1

	t_end = time.time()
	delta_t = t_end - t_start
	print("Finished after running", num_tests, "tests in", delta_t, "seconds")
	print("Build:", num_build,"| Clean:", num_clean)
	print("Pass:", num_pass, "| Failed:", len(failures))
	with open("type-mismatch-test-results.txt", "a") as file:
		file.write("Finished after running " + str(num_tests) + " tests in " + str(delta_t) +" seconds\n")
		file.write("Build: " + str(num_build) + " | Clean: " + str(num_clean) + "\n")
		file.write("Pass: " + str(num_pass) + " | Failed: " + str(len(failures)) + "\n")
		for failure in failures:
			file.write("\n")
			file.write("uut: " + failure[0] + ", actual: " + failure[1] + ", expected: " + failure[2] + "\n")
			file.write(failure[3] + " with exitcode: " + str(failure[4]) + ", output:\n")
			file.write(failure[5] + "\n")
		file.close()
		print("Results written to file")


if __name__ == "__main__":
	main()
