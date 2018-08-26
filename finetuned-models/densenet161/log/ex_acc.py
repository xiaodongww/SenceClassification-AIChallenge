import re
accs = {}
with open('log18.txt') as f:
    for line in f:
        strs = re.split(':| |\n|,', line)
        try:
            idx = int(strs[2])
            acc = float(strs[-6])
            acc_t3 = float(strs[-2])
        except:
            continue
        if idx not in accs.keys():
            accs[idx] = acc_t3

