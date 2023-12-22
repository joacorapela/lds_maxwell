import sys
import os
import random
import argparse
import configparser
import json
import math
import pickle
import numpy as np
import pandas as pd

sys.path.append("../../../../lds_python/src")
import lds.tracking.utils
import lds.inference

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtering_params_filename", type=str,
                        default="../../metadata/00000010_smoothing.ini",
                        # default=None,
                        help="filtering parameters filename")
    parser.add_argument("--estRes_filename", type=str,
                        default=None,
                        # default="../../results/14017016_estimation.pickle",
                        help="estimation results data filename")
    parser.add_argument("--first_sample", type=int, default=122000,
                        help="start position to smooth")
    parser.add_argument("--number_samples", type=int, default=200000,
                        help="number of samples to smooth")
    parser.add_argument("--sample_rate", type=float, default=40,
                        help="sample rate")
    parser.add_argument("--min_like", type=float, default=0.2,
                        help=("x and y of samples with likelihood lower than "
                              "min_like is set to nan"))
    parser.add_argument("--filtering_params_section", type=str,
                        default="params",
                        help="section of ini file containing the filtering params")
    parser.add_argument("--body_part", type=str, default="bodycenter",
                        help="body part to track")
    parser.add_argument("--data_filename", type=str,
                        default="../../data/mouse_1120297_corrected_DLC_data.h5",
                        help="data filename")
    parser.add_argument("--smoothing_results_metadata_filename_pattern", type=str,
                        default="../../results/{:08d}_smoothing_metada.ini",
                        help="smoothed results metadata filename pattern")
    parser.add_argument("--smoothing_results_filename_pattern", type=str,
                        default="../../results/{:08d}_smoothing.csv",
                        help="smoothing results filename pattern")
    args = parser.parse_args()

    filtering_params_filename = args.filtering_params_filename
    estRes_filename = args.estRes_filename 
    first_sample = args.first_sample
    number_samples = args.number_samples
    sample_rate = args.sample_rate
    min_like = args.min_like
    filtering_params_section = args.filtering_params_section
    body_part = args.body_part
    data_filename = args.data_filename
    smoothing_results_metadata_filename_pattern = args.smoothing_results_metadata_filename_pattern 
    smoothing_results_filename_pattern = args.smoothing_results_filename_pattern

    if filtering_params_filename is not None and \
       estRes_filename is not None:
        raise ValueError("filtering_params_filename and estRes_filename "
                         "cannot both be non None")

    df = pd.read_hdf(data_filename)
    scorer=df.columns.get_level_values(0)[0]
    body_part_df = df[scorer][body_part]
    body_part_df.loc[body_part_df["likelihood"]<min_like, ("x", "y")] = np.nan
    pos = np.transpose(body_part_df[["x", "y"]].to_numpy())

    # the first sample should not be nan
    while np.isnan(pos[0, first_sample]) or np.isnan(pos[1, first_sample]):
        first_sample += 1

    if number_samples is None:
        number_samples = pos.shape[1] - first_sample

    pos = pos[:, first_sample:(first_sample+number_samples)]

    dt = 1.0/sample_rate

    if filtering_params_filename is not None:
        smoothing_params = configparser.ConfigParser()
        smoothing_params.read(filtering_params_filename)
        pos_x0 = float(smoothing_params[filtering_params_section]["pos_x0"])
        pos_y0 = float(smoothing_params[filtering_params_section]["pos_y0"])
        vel_x0 = float(smoothing_params[filtering_params_section]["vel_x0"])
        vel_y0 = float(smoothing_params[filtering_params_section]["vel_x0"])
        acc_x0 = float(smoothing_params[filtering_params_section]["acc_x0"])
        acc_y0 = float(smoothing_params[filtering_params_section]["acc_x0"])
        sigma_a = float(smoothing_params[filtering_params_section]["sigma_a"])
        sigma_x = float(smoothing_params[filtering_params_section]["sigma_x"])
        sigma_y = float(smoothing_params[filtering_params_section]["sigma_y"])
        sqrt_diag_V0_value = float(smoothing_params[filtering_params_section]["sqrt_diag_V0_value"])
        if math.isnan(pos_x0):
            pos_x0 = pos[0, 0]
        if math.isnan(pos_y0):
            pos_y0 = pos[0, 1]

        m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0],
                      dtype=np.double)
        V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
        R = np.diag([sigma_x**2, sigma_y**2])

        B, Q, Z, R, _ = lds.tracking.utils.getLDSmatricesForTracking(
            dt=dt, sigma_a=sigma_a, sigma_x=sigma_x, sigma_y=sigma_y)
    elif estRes_filename is not None:
        with open(estRes_filename, "rb") as f:
            em_res = pickle.load(f)
        sigma_a = em_res["estimates"]["sigma_a"]
        m0 = em_res["estimates"]["m0"]
        V0 = em_res["estimates"]["V0"]
        R = em_res["estimates"]["R"]
        sigma_x = R[0, 0]
        sigma_y = R[1, 1]
        B, Q, Z, _, _ = lds.tracking.utils.getLDSmatricesForTracking(
            dt=dt, sigma_a=sigma_a, sigma_x=sigma_x, sigma_y=sigma_y)
    else:
        raise ValueError("one of filtering_params_filename and estRes_filename "
                         "should be non None")

    filterRes = lds.inference.filterLDS_SS_withMissingValues_np(y=pos, B=B, Q=Q,
                                                                m0=m0, V0=V0,
                                                                Z=Z, R=R)
    smoothRes = lds.inference.smoothLDS_SS(B=B, xnn=filterRes["xnn"],
                                       Vnn=filterRes["Vnn"],
                                       xnn1=filterRes["xnn1"],
                                       Vnn1=filterRes["Vnn1"],
                                       m0=m0, V0=V0)
    time = np.arange(first_sample*dt, (first_sample+number_samples)*dt, dt)
    data={"time": time, "pos1": pos[0,:], "pos2": pos[1,:],
          "fpos1": filterRes["xnn"][0,0,:], "fpos2": filterRes["xnn"][3,0,:],
          "fvel1": filterRes["xnn"][1,0,:], "fvel2": filterRes["xnn"][4,0,:],
          "facc1": filterRes["xnn"][2,0,:], "facc2": filterRes["xnn"][5,0,:],
          "spos1": smoothRes["xnN"][0,0,:], "spos2": smoothRes["xnN"][3,0,:],
          "svel1": smoothRes["xnN"][1,0,:], "svel2": smoothRes["xnN"][4,0,:],
          "sacc1": smoothRes["xnN"][2,0,:], "sacc2": smoothRes["xnN"][5,0,:]}
    df = pd.DataFrame(data=data)

    # save results
    smoothing_results_prefix_used = True
    while smoothing_results_prefix_used:
        smoothing_results_number = random.randint(0, 10**8)
        smoothing_results_metadata_filename = \
            smoothing_results_metadata_filename_pattern.format(smoothing_results_number)
        if not os.path.exists(smoothing_results_metadata_filename):
            smoothing_results_prefix_used = False
    smoothing_results_filename = smoothing_results_filename_pattern.format(smoothing_results_number)

    df.to_csv(smoothing_results_filename)
    print(f"Saved results to {smoothing_results_filename}")

    # save metadata
    smoothingResConfig = configparser.ConfigParser()
    if filtering_params_filename is not None:
        smoothingResConfig["params"] = {"filtering_params_filename":
                                        filtering_params_filename,
                                        "first_sample": first_sample,
                                        "number_samples": number_samples,
                                        "min_like": min_like,
                                        "body_part": body_part,
                                       }
    elif estRes_filename is not None:
        smoothingResConfig["params"] = {"estRes_filename": estRes_filename,
                                        "first_sample": first_sample,
                                        "number_samples": number_samples,
                                        "min_like": min_like,
                                        "body_part": body_part,
                                       }
    else:
        raise ValueError("one of filtering_params_filename and estRes_filename "
                         "should be non None")

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
