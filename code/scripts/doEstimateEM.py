import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import numpy as np
import scipy.interpolate
import scipy.stats
import pandas as pd

import lds.learning


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--estMeta_number", type=int, default=24,
                        help="estimation metadata number")
    parser.add_argument("--first_sample", type=int, default=122000,
                        help="start position to smooth")
    parser.add_argument("--number_samples", type=int, default=200000,
                        help="number of samples to smooth")
    parser.add_argument("--sample_rate", type=float, default=40,
                        help="sample rate")
    parser.add_argument("--body_part", type=str, default="bodycenter",
                        help="body part to track")
    parser.add_argument("--min_like", type=float, default=0.2,
                        help=("x and y of samples with likelihood lower than "
                              "min_like is set to nan"))
    parser.add_argument("--skip_estimation_sigma_a",
                        help=("use this option to skip the estimation of the "
                              "sqrt noise inensity"), action="store_true")
    parser.add_argument("--skip_estimation_R",
                        help="use this option to skip the estimation of R",
                        action="store_true")
    parser.add_argument("--skip_estimation_m0",
                        help="use this option to skip the estimation of m0",
                        action="store_true")
    parser.add_argument("--skip_estimation_V0",
                        help="use this option to skip the estimation of V0",
                        action="store_true")
    parser.add_argument("--estInit_metadata_filename_pattern", type=str,
                        default="../../metadata/{:08d}_estimation.ini",
                        help="estimation initialization metadata filename pattern")
    parser.add_argument("--data_filename", type=str,
                        default="../../data/mouse_1120297_corrected_DLC_data.h5",
                        help="inputs positions filename")
    parser.add_argument("--estRes_metadata_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.ini",
                        help="estimation results metadata filename pattern")
    parser.add_argument("--estRes_data_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.pickle",
                        help="estimation results data filename pattern")
    args = parser.parse_args()

    estMeta_number = args.estMeta_number
    first_sample = args.first_sample
    number_samples = args.number_samples
    sample_rate = args.sample_rate
    body_part = args.body_part
    min_like = args.min_like
    skip_estimation_sigma_a = \
        args.skip_estimation_sigma_a
    skip_estimation_R = args.skip_estimation_R
    skip_estimation_m0 = args.skip_estimation_m0
    skip_estimation_V0 = args.skip_estimation_V0
    estInit_metadata_filename_pattern = args.estInit_metadata_filename_pattern
    data_filename = args.data_filename
    estRes_metadata_filename_pattern = args.estRes_metadata_filename_pattern
    estRes_data_filename_pattern = args.estRes_data_filename_pattern

    estInit_metadata_filename = \
        estInit_metadata_filename_pattern.format(estMeta_number)

    estMeta = configparser.ConfigParser()
    estMeta.read(estInit_metadata_filename)
    # start_position = int(estMeta["data_params"]["start_position"])
    # number_positions = int(estMeta["data_params"]["number_positions"])
    pos_x0 = float(estMeta["initial_params"]["pos_x0"])
    pos_y0 = float(estMeta["initial_params"]["pos_y0"])
    vel_x0 = float(estMeta["initial_params"]["vel_x0"])
    vel_y0 = float(estMeta["initial_params"]["vel_y0"])
    ace_x0 = float(estMeta["initial_params"]["ace_x0"])
    ace_y0 = float(estMeta["initial_params"]["ace_y0"])
    sigma_a0 = float(estMeta["initial_params"]["sigma_a"])
    sigma_x0 = float(estMeta["initial_params"]["sigma_x"])
    sigma_y0 = float(estMeta["initial_params"]["sigma_y"])
    sqrt_diag_V0_value = float(estMeta["initial_params"]["sqrt_diag_v0_value"])
    em_max_iter = int(estMeta["optim_params"]["em_max_iter"])
    Qe_reg_param = float(estMeta["optim_params"]["em_Qe_reg_param"])

    df = pd.read_hdf(data_filename)
    scorer=df.columns.get_level_values(0)[0]
    body_part_df = df[scorer][body_part]
    body_part_df.loc[body_part_df["likelihood"]<min_like, ("x", "y")] = np.nan
    y = np.transpose(body_part_df[["x", "y"]].to_numpy())

    # the first sample should not be nan
    while np.isnan(y[0, first_sample]) or np.isnan(y[1, first_sample]):
        first_sample += 1

    y = y[:, first_sample:(first_sample+number_samples)]

    dt = 1.0/sample_rate

#     data = pd.read_csv(filepath_or_buffer=data_filename)
#     data = data.iloc[start_position:start_position+number_positions,:]
#     y = np.transpose(data[["x", "y"]].to_numpy())
#     date_times = pd.to_datetime(data["time"])
#     dt = (date_times.iloc[1]-date_times.iloc[0]).total_seconds()

    times = np.arange(0, y.shape[1]*dt, dt)
    not_nan_indices_y0 = set(np.where(np.logical_not(np.isnan(y[0, :])))[0])
    not_nan_indices_y1 = set(np.where(np.logical_not(np.isnan(y[1, :])))[0])
    not_nan_indices = np.array(sorted(not_nan_indices_y0.union(not_nan_indices_y1)))
    y_no_nan = y[:, not_nan_indices]
    t_no_nan = times[not_nan_indices]
    y_interpolated = np.empty_like(y)
    tck, u = scipy.interpolate.splprep([y_no_nan[0, :], y_no_nan[1, :]], s=0, u=t_no_nan)
    y_interpolated[0, :], y_interpolated[1, :] = scipy.interpolate.splev(times, tck)
    y = y_interpolated

    if math.isnan(pos_x0):
        pos_x0 = y[0, 0]
    if math.isnan(pos_y0):
        pos_y0 = y[1, 0]

    B, Q, Z, R_0, Qe = \
        lds.tracking.utils.getLDSmatricesForTracking(dt=dt,
                                                     sigma_a=sigma_a0,
                                                     sigma_x=sigma_x0,
                                                     sigma_y=sigma_y0)

    Qe_regularized = Qe + Qe_reg_param*np.eye(Qe.shape[0])
    m0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
                  dtype=np.double)
    m0_0 = m0
    sqrt_diag_V0 = np.array([sqrt_diag_V0_value for i in range(len(m0))])
    V0_0 = np.diag(sqrt_diag_V0)

    vars_to_estimate = {}
    if skip_estimation_sigma_a:
        vars_to_estimate["sigma_a"] = False
    else:
        vars_to_estimate["sigma_a"] = True

    if skip_estimation_R:
        vars_to_estimate["R"] = False
    else:
        vars_to_estimate["R"] = True

    if skip_estimation_m0:
        vars_to_estimate["m0"] = False
    else:
        vars_to_estimate["m0"] = True

    if skip_estimation_V0:
        vars_to_estimate["V0"] = False
    else:
        vars_to_estimate["V0"] = True

    optim_res  = lds.learning.em_SS_tracking(
        y=y, B=B, sigma_a0=sigma_a0,
        Qe=Qe_regularized, Z=Z, R_0=R_0, m0_0=m0_0, V0_0=V0_0,
        vars_to_estimate=vars_to_estimate,
        max_iter=em_max_iter)

    print(optim_res["termination_info"])

    # save results
    est_prefix_used = True
    while est_prefix_used:
        estRes_number = random.randint(0, 10**8)
        estRes_metadata_filename = \
            estRes_metadata_filename_pattern.format(estRes_number)
        if not os.path.exists(estRes_metadata_filename):
            est_prefix_used = False
    estRes_data_filename = estRes_data_filename_pattern.format(estRes_number)

    estimRes_metadata = configparser.ConfigParser()
    estimRes_metadata["data_params"] = {"data_filename": data_filename}
    estimRes_metadata["estimation_params"] = {"estInitNumber": estMeta_number}
    with open(estRes_metadata_filename, "w") as f:
        estimRes_metadata.write(f)

    with open(estRes_data_filename, "wb") as f:
        pickle.dump(optim_res, f)
    print("Saved results to {:s}".format(estRes_data_filename))

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
