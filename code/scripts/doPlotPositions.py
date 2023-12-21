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
    parser.add_argument("--first_sample", type=int, default=122000,
                        help="first sample")
    parser.add_argument("--number_samples", type=int, default=200000,
                        help="number of samples to plot")
    parser.add_argument("--sample_rate", type=float, default=40,
                        help="sample rate")
    parser.add_argument("--min_like", type=float, default=0.0,
                        help=("x and y of samples with likelihood lower than "
                              "min_like is set to nan"))
    parser.add_argument("--colorscale", type=str, default="Rainbow",
                        help="colorscale name")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/positions_from{:d}_numberSamples{:d}_minLike{:.02f}.{:s}")

    args = parser.parse_args()

    data_filename = args.data_filename
    body_part = args.body_part
    first_sample = args.first_sample
    number_samples = args.number_samples
    sample_rate = args.sample_rate
    min_like = args.min_like
    colorscale = args.colorscale
    fig_filename_pattern = args.fig_filename_pattern

    df = pd.read_hdf(data_filename)
    scorer=df.columns.get_level_values(0)[0]
    body_part_df = df[scorer][body_part]
    body_part_df.loc[body_part_df["likelihood"]<min_like, ("x", "y")] = np.nan
    pos = np.transpose(body_part_df[["x", "y"]].to_numpy())

    if number_samples is None:
        number_samples = pos.shape[1]

    pos = pos[:, first_sample:(first_sample+number_samples)]
    body_part_df = body_part_df.iloc[first_sample:(first_sample+number_samples), :]
    N = pos.shape[1]
    dt = 1.0/sample_rate
    time = np.arange(first_sample*dt, (first_sample+number_samples)*dt, dt)

    customdata = np.stack((time, body_part_df["likelihood"].to_numpy()), axis=-1)
    fig = go.Figure()
    trace_mes = go.Scatter(x=pos[0, :], y=pos[1, :],
                           mode="lines+markers",
                           # mode="markers",
                           marker={"color": time,
                                   "colorscale": colorscale,
                                   "colorbar": {"title": "Time (sec)"},
                                  },
                           customdata=customdata,
                           hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata[0]}<br><b>like</b>:%{customdata[1]:.4f}",
                           name="measured",
                           showlegend=False,
                          )
    fig.add_trace(trace_mes)
    fig.update_layout(xaxis_title="x (pixels)",
                      yaxis_title="y (pixels)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    fig.write_image(fig_filename_pattern.format(first_sample, number_samples,
                                                min_like, "png"))
    fig.write_html(fig_filename_pattern.format(first_sample, number_samples,
                                               min_like, "html"))
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
