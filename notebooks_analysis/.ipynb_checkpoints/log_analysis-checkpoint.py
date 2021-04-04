import os, sys

info = {}

with open(sys.argv[1], "r") as f:
    for line in f:
        ls = line[:-1].split("\t")
        if ls[-1] == "success":
            ls = ls[:-1]
            info[ls[0]] = [float(x)/float(ls[1]) for x in ls[1:]]


    
data = list(info.values())
avg = [float(sum(col))/len(col) for col in zip(*data)]

track_idx = 2

for k, v in sorted(info.items(), key=lambda item: item[1][track_idx]):
    if v[track_idx] > avg[track_idx]:
        print(k, v)

print(avg)