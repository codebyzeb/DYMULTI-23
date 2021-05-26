import sys

filename = sys.argv[1]

with open(filename, 'r') as evalfile:
	lines = evalfile.readlines()
	vals = [str(round(float(line.strip().split('\t')[1])*100,1)) for line in lines]
	out_vals = vals[9:12] + vals[0:6] + [vals[14], vals[13]]
	print(' & '.join(out_vals))
