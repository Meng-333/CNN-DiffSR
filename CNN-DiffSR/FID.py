import argparse
import os
from litsr.metrics import calc_fid

rslt_path = "/data/mfe/mfe_DiffSR/FID/AID_Test_e483"
hr_path = "/data/mfe/mfe_diffsr/dataset/Test/AID_Test"

# rslt_path = "/data/mfe/mfe_DiffSR/FID/UCM_Test_e483"
# hr_path = "/data/mfe/mfe_diffsr/dataset/Test/UCM_Test"

#rslt_path = "/data/mfe/mfe_DiffSR/FID/NWPU_Test_e483/airplane"
#hr_path =  "/data/mfe/mfe_diffsr/dataset/Test/NWPU_Test/airplane"

paths = [rslt_path, hr_path]
fid_score = calc_fid(paths)
print("- SR_FID : {:.5f}".format(fid_score))
print16 = "- SR_FID : {:.5f}".format(fid_score)


