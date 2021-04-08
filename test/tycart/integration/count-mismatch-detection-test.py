#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
from os import remove
from shutil import copyfile
import time
import random

uuts = ["veloc/gol/"]


def main():
	t_start = time.time()
	num_tests = 0
	num_build = 0
	num_clean = 0
	num_pass = 0
	num_false_positive = 0
	failures = []
	false_positive = False
	both_types = "int"

	for uut in uuts:
		print("\nBeginning count-mismatch tests of TY_protect macro with " + uut)
		actual_size = 2500
		num_count_tests = 3
		print("Running", num_count_tests, "count-mismatch tests with actual size =", actual_size, ", type =", both_types)
		for i in range(num_count_tests):
			false_positive = False
			if i < num_count_tests / 2:
				if i == 0:
					expected_size = 0
				else:
					expected_size = random.randrange(1, actual_size)
			else:
				if i == num_count_tests - 1:
					expected_size = actual_size
					false_positive = True
				else:
					expected_size = random.randrange(actual_size + 1, actual_size * 2)

			print("\nCount-mismatch test", i+1, "-- expected_size =", expected_size)
			if false_positive:
				print("expected_size == actual_size, checking for false positive")

			# copy original make file
			copyfile(uut + "Makefile", uut + "MakefileTesting")

			# prepend preprocessor macros to make file copy
			with open(uut + "MakefileTesting", "r+") as file:
				content = file.read()
				file.seek(0, 0)
				file.write('EXPECTED_SIZE = -DEXPECTED_SIZE=' + str(expected_size) + '\n')
				file.write('ACTUAL_TYPE = -DACTUAL_TYPE="' + both_types + '"\n')
				file.write('EXPECTED_TYPE = -DEXPECTED_TYPE="' + both_types + '"\n')

				if(uut == "fti/gol/"):
					actual_type_fti = both_types.replace(" ", ", ") # adds comma before whitespace to make it work with the fti verison of the macro
					expected_type_fti = both_types.replace(" ", ", ")
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
				failures.append([uut, actual_size, expected_size, msg, exitcode, output])
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
					if "Assert failed: Expected number" in output:
						num_pass += 1
						print("Count-mismatch detected, Test passed")
					else:
						msg = "Expected failed assertion but execution terminated  abnormally for some other reason"
						failures.append([uut, actual_size, expected_size, msg, exitcode, output])
						print(msg)
				elif exitcode and false_positive:
					if "Assert failed: Expected number" in output:
						num_false_positive += 1
						msg = "False positive occured"
						failures.append([uut, actual_size, expected_size, msg, exitcode, output])
					else:
						msg = "Execution terminated  abnormally for some other reason"
						failures.append([uut, actual_size, expected_size, msg, exitcode, output])
						print(msg)
				elif not exitcode and not false_positive:
					msg = "count-mismatch not detected"
					failures.append([uut, actual_size, expected_size, msg, exitcode, output])
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

			print()
			# delete make file copy
			remove(uut + "MakefileTesting")
			num_tests += 1

	t_end = time.time()
	delta_t = t_end - t_start
	print("Finished after running", num_tests, "tests in", delta_t, "seconds")
	print("Build:", num_build, "| Clean:", num_clean)
	print("Pass:", num_pass, "| Failed:", len(failures))
	with open("count-mismatch-" + both_types + "-test-results.txt", "a") as file:
		file.write("Finished after running " + str(num_tests) + " tests in " + str(delta_t) +" seconds\n")
		file.write("actual and expected type were set to: " + both_types + "\n")
		file.write("Build: " + str(num_build) + " | Clean: " + str(num_clean) + "\n")
		file.write("Pass: " + str(num_pass) + " | Failed: " + str(len(failures)) + "\n")
		for failure in failures:
			file.write("\n")
			file.write("uut: " + failure[0] + ", actual: " + str(failure[1]) + ", expected: " + str(failure[2]) + "\n")
			file.write(failure[3] + " with exitcode: " + str(failure[4]) + ", output:\n")
			file.write(failure[5] + "\n")
		file.close()
		print("Results written to file")


if __name__ == "__main__":
	main()
