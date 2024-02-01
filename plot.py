import pickle as pkl

pkl_list = [
    "./log/H=20_const/20240201_001340_20_0.010_const_0.050_0.20.pkl",
    "./log/H=20_const/20240201_001345_20_0.010_const_0.100_0.20.pkl",
    "./log/H=20_const/20240201_001346_20_0.010_const_0.200_0.20.pkl",
    "./log/H=20_const/20240201_001807_20_0.010_const_0.400_0.20.pkl",
    "./log/H=20_const/20240201_003024_20_0.010_const_0.800_0.20.pkl",
    "./log/H=20_const/20240201_003939_20_0.010_const_1.200_0.20.pkl",
]

reward_list = []
for pk in pkl_list:
    with open(pk, "rb") as f:
        db = pkl.load(f)
        reward_list.append(db["reward"])

print(reward_list)