import os, sys

info = {}

with open(sys.argv[1], "r") as f:
	for line in f:
		ls = line[:-1].split("\t")
		if len(ls) == 6:		
			info[ls[0]] = [float(x)/float(ls[1]) for x in ls[1:]]

print(info)

data = list(info.values())
avg = [float(sum(col))/len(col) for col in zip(*data)]
print(avg)