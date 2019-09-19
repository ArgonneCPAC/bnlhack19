import subprocess
import tqdm
import numpy as np
import h5py

rockstar_file = "hlist_1.00000.list"

n_lines_total = int(
    subprocess.check_output("wc -l %s" % rockstar_file, shell=True)
    .split()[0]
    .strip()
)
n_comments = int(
    subprocess.check_output(
        "grep '#' %s | wc -l" % rockstar_file, shell=True
    ).strip()
)

n_gals = n_lines_total - n_comments

print("n_gals:", n_gals)

data = np.zeros(
    n_gals,
    dtype=[
        ("id", "i8"),
        ("mpeak", "f4"),
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("vx", "f4"),
        ("vy", "f4"),
        ("vz", "f4"),
    ],
)

# these are the columns
# scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6)
# desc_pid(7) phantom(8) sam_mvir(9) mvir(10) rvir(11) rs(12) vrms(13)
# mmp?(14) scale_of_last_MM(15) vmax(16) x(17) y(18) z(19) vx(20) vy(21)
# vz(22) Jx(23) Jy(24) Jz(25) Spin(26) Breadth_first_ID(27) Depth_first_ID(28)
# Tree_root_ID(29) Orig_halo_ID(30) Snap_num(31)
# Next_coprogenitor_depthfirst_ID(32) Last_progenitor_depthfirst_ID(33)
# Rs_Klypin(34) Mmvir_all(35) M200b(36) M200c(37) M500c(38) M2500c(39)
# Xoff(40) Voff(41) Spin_Bullock(42) b_to_a(43) c_to_a(44) A[x](45) A[y](46)
# A[z](47) b_to_a(500c)(48) c_to_a(500c)(49) A[x](500c)(50) A[y](500c)(51)
# A[z](500c)(52) T/|U|(53) Macc(54) Mpeak(55) Vacc(56) Vpeak(57)
# Halfmass_Scale(58) Acc_Rate_Inst(59) Acc_Rate_100Myr(60) Acc_Rate_1*Tdyn(61)
# Acc_Rate_2*Tdyn(62) Acc_Rate_Mpeak(63) Mpeak_Scale(64) Acc_Scale(65)
# M4%_Scale(66)

loc = 0
with open(rockstar_file, "r") as f:
    for line in tqdm.tqdm(f.readlines(), total=n_gals):
        if line[0] == "#":
            continue
        line = line.strip().split()
        data["id"][loc] = int(line[1])
        data["mpeak"][loc] = float(line[55])
        data["x"][loc] = float(line[17])
        data["y"][loc] = float(line[18])
        data["z"][loc] = float(line[19])
        data["vx"][loc] = float(line[20])
        data["vy"][loc] = float(line[21])
        data["vz"][loc] = float(line[22])

        loc += 1

print("writing HDF5 data", flush=True)
with h5py.File("halos_Lb125_1024.h5", "w") as hf:
    hf.create_dataset("halos", data=data)
