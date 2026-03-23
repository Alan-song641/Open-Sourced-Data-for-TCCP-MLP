
# # Preprocessing to read weighted FPS and generate fpParas.dat


import os
from itertools import product

def weightedFPs_to_FPs(filename):
	print("Generating fpParas.dat from weighted_fpParas.dat")

	with open(filename+'/weighted_fpParas.dat', "r") as file:
		filelst = file.readlines()

	elements = filelst[3].split()
	nG1 = int(filelst[6].split()[0])
	nG2 = int(filelst[6].split()[1])

	G1 = filelst[8: 8+nG1]
	G2 = filelst[9+nG1: 9+nG1+nG2]

	# Generate two-element combinations
	two_combinations = [p for p in product(elements, repeat=2)]
	# print("Two-element combinations:", two_combinations)

	# Generate three-element combinations
	three_combinations = [p for p in product(elements, repeat=3)]
	three_combinations = [comb for comb in three_combinations if not ((comb[1] != comb[0]) and (comb[0] == comb[2]))]
	# print("Three-element combinations:", three_combinations)

	total_G2 = list()
	for comb in three_combinations:
		total_G2 += [line.replace('X X X', f'{comb[0]} {comb[1]} {comb[2]}') for line in G2]

	total_G1 = list()
	for comb in two_combinations:
		total_G1 += [line.replace('X   X', f'{comb[0]} {comb[1]}') for line in G1]

	output_filename = filename + '/fpParas.dat'
	
	with open(output_filename, 'w') as outfile:
		outfile.write('# FP type\n')
		outfile.write('BP\n')
		outfile.write('#Elements\n')
		outfile.write(' '.join(elements) + '\n')
		outfile.write(' '.join(['0.0'] * len(elements)) + '\n')
		outfile.write('#nG1s nG2s\n')
		outfile.write(f'{len(total_G1)} {len(total_G2)}\n')

		outfile.write('#type  center neighbor   eta       Rs   Rcut\n')
		for g1 in total_G1:
			if '\n' not in g1:
				g1 += '\n'
			outfile.write(g1)

		outfile.write('#type  center neighbor1 neighbor2  eta    zeta   lambda   thetas  rcut\n')
		for g2 in total_G2:
			if '\n' not in g2:
				g2 += '\n'
			outfile.write(g2)

	with open(output_filename, 'r') as file:
		lines = file.readlines()

	if lines[-1].endswith('\n'):
		lines[-1] = lines[-1][:-1]

	with open(output_filename, 'w') as file:
		file.writelines(lines)


