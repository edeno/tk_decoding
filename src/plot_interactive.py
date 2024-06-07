import holoviews as hv
import numpy as np
import panel as pn


def create_2D_decoding_plot(
    position_info,
    posterior,
    sampling_frequency=250,
    width=800,
    height=600,
    head_dir_radius=4,
    position_names=("x", "y"),
    head_dir_name="head_direction",
    cmap="viridis",
    animal_color="magenta",
    decode_opacity=1.0,
):
    n_time = len(posterior.time)

    def get_animal_position_dot(time_ind):
        head_point = hv.Scatter(
            position_info.iloc[[time_ind]],
            kdims=position_names[0],
            vdims=position_names[1],
        ).opts(size=10, color=animal_color, width=width, height=height)

        # Calculate the end point of the head direction line
        x_start = position_info.iloc[time_ind][position_names[0]]
        y_start = position_info.iloc[time_ind][position_names[1]]

        direction = position_info.iloc[time_ind][head_dir_name]

        x_end = x_start + head_dir_radius * np.cos(direction)
        y_end = y_start + head_dir_radius * np.sin(direction)

        head_direction_line = hv.Segments(
            [{"x0": x_start, "y0": y_start, "x1": x_end, "y1": y_end}]
        ).opts(color=animal_color, line_width=3)

        return head_point * head_direction_line

    def get_posterior_quadmesh(time_ind):
        posterior_slice = posterior.isel(time=time_ind)
        return hv.QuadMesh(
            posterior_slice,
            kdims=["x_position", "y_position"],
            vdims=["probability"],
        ).opts(cmap=cmap, alpha=decode_opacity, width=width, height=height)

    # Create player controls
    player = create_player(n_time, sampling_frequency=sampling_frequency)
    playback_speed_dropdown = create_playback_speed_dropdown(
        player, sampling_frequency=sampling_frequency
    )

    posterior_image = hv.DynamicMap(pn.bind(get_posterior_quadmesh, player))
    animal_position = hv.DynamicMap(pn.bind(get_animal_position_dot, player))

    # Overlay the animal position on the posterior
    decode_plot = pn.Column(
        posterior_image * animal_position, pn.Row(player, playback_speed_dropdown)
    )

    return decode_plot, player


def create_scrolling_line_plot(
    data,
    ylabel,
    ylim,
    player,
    window_lim=(-0.4, 0.4),
    height_factor=3,
    sampling_frequency=250,
    width=800,
    height=600,
):
    n_time = len(data)
    n_samples = int((window_lim[1] - window_lim[0]) * sampling_frequency)
    half_samples = n_samples // 2

    def plot_func(n):
        start_idx = max(0, n - half_samples)
        end_idx = min(n_time, n + half_samples)
        window = np.asarray(data, dtype=float)[start_idx:end_idx].squeeze()
        line = hv.Curve(
            (np.arange(start_idx - n, end_idx - n) / sampling_frequency, window),
            kdims="time relative to current (s)",
            vdims=ylabel,
        ).opts(
            color="black",
            width=width,
            height=int(height / height_factor),
            ylim=ylim,
            xlim=window_lim,
        )
        return line

    return hv.DynamicMap(pn.bind(plot_func, player))


def create_player(n_time, sampling_frequency=250):
    base_interval = int(1000 / sampling_frequency)  # ms/frame
    player = pn.widgets.Player(
        start=0, end=n_time - 1, value=0, interval=base_interval, name="Time"
    )
    return player


def create_playback_speed_dropdown(player, sampling_frequency=250):
    # Dropdown to control the speed of the player (interval)
    speed_options = {
        "1/8x": 8,
        "1/4x": 4,
        "1/2x": 2,
        "1x": 1,
        "2x": 0.5,
        "4x": 0.25,
        "8x": 0.125,
    }
    speed_dropdown = pn.widgets.Select(
        name="Playback Speed", options=speed_options, value=1
    )
    base_interval = 1000 / sampling_frequency  # ms/frame

    # Update the player's interval property based on the dropdown's value
    def update_interval(event):
        player.interval = int(event.new * base_interval)

    speed_dropdown.param.watch(update_interval, "value")

    return speed_dropdown


def create_visualization(
    position_info,
    posterior,
    multiunit_firing_rate,
    sampling_frequency=250,
    width=800,
    height=600,
    window_lim=(-0.4, 0.4),
):
    decode_plot, player = create_2D_decoding_plot(
        position_info,
        posterior,
        sampling_frequency=sampling_frequency,
        width=width,
        height=height,
    )

    multiunit_plot = create_scrolling_line_plot(
        multiunit_firing_rate,
        "Rate (spikes/s)",
        (0, multiunit_firing_rate.to_numpy().max()),
        player,
        window_lim=window_lim,
        height_factor=3,
        sampling_frequency=sampling_frequency,
        width=width,
        height=height,
    )
    speed_plot = create_scrolling_line_plot(
        position_info["speed"],
        "Speed (cm/s)",
        (0, 100),
        player,
        window_lim=window_lim,
        height_factor=3,
        sampling_frequency=sampling_frequency,
        width=width,
        height=height,
    )
    head_direction_plot = create_scrolling_line_plot(
        position_info["head_direction"],
        "Head dir. (rad)",
        (-np.pi, np.pi),
        player,
        window_lim=window_lim,
        height_factor=3,
        sampling_frequency=sampling_frequency,
        width=width,
        height=height,
    )

    # Layout the components using Panel
    layout = pn.Row(
        decode_plot,
        pn.Column(multiunit_plot, speed_plot, head_direction_plot),
    )

    return layout
