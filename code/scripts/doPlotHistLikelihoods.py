import sys
import pdb
import numpy as np
import pandas as pd
import argparse
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str,
                        default="../../data/mouse_1120297_corrected_DLC_data.h5",
                        help="data filename")
    parser.add_argument("--body_part", type=str, default="bodycenter",
                        help="body part to track")
    parser.add_argument("--first_sample", type=int, default=0,
                        help="first sample")
    parser.add_argument("--number_samples", type=int, default=325906,
                        help="number of samples")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/hist_like_from{:d}_numberSamples{:d}.{:s}")

    args = parser.parse_args()

    data_filename = args.data_filename
    body_part = args.body_part
    first_sample = args.first_sample
    number_samples = args.number_samples
    fig_filename_pattern = args.fig_filename_pattern

    df = pd.read_hdf(data_filename)
    scorer=df.columns.get_level_values(0)[0]
    body_part_df = df[scorer][body_part]
    likelihoods = body_part_df["likelihood"]

    fig = go.Figure()
    trace = go.Histogram(x=likelihoods)
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="Likelihood",
                      yaxis_title="Count",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    fig.write_image(fig_filename_pattern.format(first_sample, number_samples, "png"))
    fig.write_html(fig_filename_pattern.format(first_sample, number_samples, "html"))
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
