import os, sys
from statistics import median
from collections import defaultdict, ChainMap

info = defaultdict(dict)

files = [
    os.path.join("notebooks", "titanic", "log.txt"), 
    os.path.join("notebooks", "fifa19", "log.txt"),
    os.path.join("notebooks", "google", "log.txt"), 
    os.path.join("notebooks", "airbnb", "log.txt")
]

for file_name in files:
    with open(file_name, "r") as f:
        for line in f:
            ls = line[:-1].split("\t")
            if ls[-1] == "success":
                ls = ls[:-1]
                info[file_name][ls[0]] = [float(x)/float(ls[1]) for x in ls[1:]] + [0] + [float(ls[i]) for i in range(1, len(ls))]

def print_data(cur_dict):
    data = list(cur_dict.values())
    avg = [float(sum(col))/len(col) for col in zip(*data)]
    med = [median(col) for col in zip(*data)]

    track_idx = 2

    # for k, v in sorted(cur_dict.items(), key=lambda item: item[1][track_idx]):
    #     if v[track_idx] > avg[track_idx]:
    #         print(k, v)

    print(avg, avg[-1]/avg[-2])
    # print(med)

for file_name in files:
    cur_dict = info[file_name]
    print(file_name)
    print_data(cur_dict)

print_data(dict(ChainMap(*info.values())))
