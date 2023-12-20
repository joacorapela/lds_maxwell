import sys
import argparse
import configparser
import json
import math
import numpy as np
import pandas as pd

sys.path.append("../../../../lds_python/src/lds")
import tracking.utils
import inference

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtering_params_filename", type=str,
                        default="../../metadata/00000010_smoothing.ini",
                        help="filtering parameters filename")
    parser.add_argument("--first_sample", type=int, default=0,
                        help="start position to smooth")
    parser.add_argument("--number_samples", type=int, default=None,
                        help="number of samples to smooth")
    parser.add_argument("--sample_rate", type=float, default=1,
                        help="sample rate")
    parser.add_argument("--filtering_params_section", type=str,
                        default="params",
                        help="section of ini file containing the filtering params")
    parser.add_argument("--body_part", type=str, default="bodycenter",
                        help="body part to track")
    parser.add_argument("--data_filename", type=str,
                        default="../../data/mouse_1120297_corrected_DLC_data.h5",
                        help="data filename")
    parser.add_argument("--smoothed_data_filename_pattern", type=str,
                        default="../../results/smoothed_results_firstSample{:d}_numberOfSamples{:d}.csv",
                        help="smoothed data filename pattern")
    args = parser.parse_args()

    filtering_params_filename = args.filtering_params_filename
    first_sample = args.first_sample
    number_samples = args.number_samples
    sample_rate = args.sample_rate
    filtering_params_section = args.filtering_params_section
    body_part = args.body_part
    data_filename = args.data_filename
    smoothed_data_filename_pattern = args.smoothed_data_filename_pattern

    df = pd.read_hdf(data_filename)
    scorer=df.columns.get_level_values(0)[0]
    pos = np.transpose(df[scorer][body_part][["x", "y"]].to_numpy())

    if number_samples is None:
        number_samples = pos.shape[1]

    smoothed_data_filename = args.smoothed_data_filename_pattern.format(
        first_sample, number_samples)

    pos = pos[:, first_sample:(first_sample+number_samples)]

    dt = 1.0/sample_rate

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

    # Taken from the book
    # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
    # section 6.3.3
    # Eq. 6.3.3-2
    B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
                   [0, 1, dt, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, dt, .5*dt**2],
                   [0, 0, 0, 0, 1, dt],
                   [0, 0, 0, 0, 0, 1]],
                  dtype=np.double)
    Z = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0]],
                  dtype=np.double)
    # Eq. 6.3.3-4
    Qt = np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                  dtype=np.double)
    R = np.diag([sigma_x**2, sigma_y**2])
    m0 = np.array([pos[0, 0], 0, 0, pos[1, 0], 0, 0], dtype=np.double)
    # m0.shape = (len(m0), 1)
    V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)
    Q = tracking.utils.buildQfromQt_np(Qt=Qt, sigma_ax=sigma_a, sigma_ay=sigma_a)

    filterRes = inference.filterLDS_SS_withMissingValues_np(y=pos, B=B, Q=Q,
                                                            m0=m0, V0=V0, Z=Z,
                                                            R=R)
    smoothRes = inference.smoothLDS_SS(B=B, xnn=filterRes["xnn"],
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
    df.to_csv(smoothed_data_filename)
    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
