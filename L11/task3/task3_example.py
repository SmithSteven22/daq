import nidaqmx
from nidaqmx import stream_readers
from nidaqmx.constants import TerminalConfiguration

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import DE_alg_RRC

from nidaqmx.constants import TriggerType, Edge, AcquisitionType, TaskMode, LineGrouping


DEV = "Dev3"
TRIG_SRC = "PFI0"
AI_CH = ["ai0", "ai1"]
AO_CH = ["ao0"]
T_TO_CHRG = 0.05
SAMPLING_RATE = 100000
PRETRIGGER_SAMPLES_RATIO = 0.01
NO_OF_SAMPLES = int(T_TO_CHRG * SAMPLING_RATE)
AO_VS = 5

samples_buf = None  # Initialize a buffer to hold the acquired samples in.

# Initializes the DAQ device.
# Parameters:
# dev: DAQ device name (String)
# ai_channels: The analouge channels to use in the analouge in task (1D list of strings)
# ao_channels: The analouge channels to use in the analouge out task (1D list of strings)
# trig_src: The trigger source pin which triggers the ai_task. (String)
# no_of_samples: The total number of samples to acquire (Integer)
# sampling_rate: The sampling rate to use for the task. (Float)
###
# Returns:
# ai_task: DAQ device task that measures ai_channels
# ao_task: DAQ device task that writes to ao_channels
###


def initialize_device(dev, ai_channels, ao_channels, trig_src, no_of_samples, sampling_rate):
    global samples_buf

    # Initialize buffer for samples. This creates a 2D numpy array: a row for each of the input channels, each row holds no_of_samples data points.
    samples_buf = np.zeros((len(ai_channels), no_of_samples))

    ai_task = nidaqmx.Task("AI TASK")  # Create a task, named "AI TASK"

    # Iterate over all input channels and add them to the ai_task
    for i in range(len(ai_channels)):
        # Voltage channel must be in the form <Device name>/<Analouge in channel>. More info: https://nidaqmx-python.readthedocs.io/en/latest/_modules/nidaqmx/_task_modules/ai_channel_collection.html#AIChannelCollection.add_ai_voltage_chan
        ai_task.ai_channels.add_ai_voltage_chan(
            dev + "/" + ai_channels[i], ai_channels[i], terminal_config=TerminalConfiguration.RSE)

    # Set timing for task.
    ai_task.timing.cfg_samp_clk_timing(
        sampling_rate, samps_per_chan=no_of_samples)

    ai_task.in_stream.auto_start = False  # Forces to explicitly start task.

    ######################################################################################
    ai_task.triggers.start_trigger.cfg_dig_edge_start_trig("PFI0")
    # ai_task.triggers.start_trigger.trig_type = TriggerType.DIGITAL_EDGE
    # ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
    #     trigger_source="Dev4/port0/PFI0", trigger_edge=Edge.RISING)

    reader = stream_readers.AnalogMultiChannelReader(
        ai_task.in_stream)  # Creates a reader instance

    ao_task = nidaqmx.Task("AO TASK")
    for ao_channel in ao_channels:
        ao_task.ao_channels.add_ao_voltage_chan(
            dev + "/" + ao_channel, ao_channel)

    return ai_task, ao_task, reader

# Starts the tasks: Generates a pulse using ao_task and collects the samples into samples_buf using ai_task.
# Parameters:
# ai_task: The ai_task to use for sample collection (nidaqmx.task)
# ao_task: The ao_task to use for pulse generation (nidaqmx.task)
# reader: Reader instance to use for sample reading (nidaqmx.stream_readers.AnalogMultiChannelReader)
# ao_Vs: The voltage (in volts) to write using ao_task (float)
# t_to_dischrg: The time to wait fater reseting output to 0V for the system to discharge (float)


def start_tasks(ai_task, ao_task, reader, ao_Vs, t_to_dischrg=1):
    global samples_buf

    ai_task.start()

    ao_task.start()
    ao_task.write(ao_Vs)

    reader.read_many_sample(samples_buf, nidaqmx.constants.READ_ALL_AVAILABLE)
    ai_task.stop()
    print("discharge ", t_to_dischrg)
    time.sleep(t_to_dischrg)
    ao_task.write(0)
    time.sleep(t_to_dischrg)
    ao_task.stop()

# Generates a matplotlib graph on a subplot and places it on the figure.
# Paramteters:
# ydata: The data to plot. Each row plots a single line. (2D numpy array of floats)
# xdata: X-axis values (1D numpy array of integers)
# title: Title to use for the graph (String)
# xlabel: Label for the x-axis (String)
# ylabel: Label for the y-axis (String)
# data_labels: Labels to use for each of the lines given in ydata.
# plot_shape: The shape to use for the fiugre. This is the shape of the whole plottable area, given in number of rows and columns (2-tuple of integers)
# subplot_colspan: Specifies how many columns the given subplot spans in plot_shape units (Integer)
# subplot_rowspan: Specifies how many rows the given subplot spans in plot_shape units (Integer)
# Returns:
# ax: The axis of the current subplot. (numpy.axes.SubplotBase)


def generate_graph(ydata, xdata=None, title="Graph Title", xlabel="x", ylabel="y", data_labels=None, plot_shape=(1, 1), subplot_location=(0, 0), subplot_colspan=1, subplot_rowspan=1):

    # Generate default values for X-axis if not given. This creates a numpy array that contains numbers from 0 to the lingth of ydata.
    if xdata is None:
        xdata = np.linspace(0, ydata.shape[1], ydata.shape[1])

    # Generate default values for data_labels if not given.
    if data_labels is None:
        data_labels = []
        for i in range(len(ydata)):
            data_labels.append("line " + str(i))

    # Generate a pandas dataframe for the input data for seaborn lineplot function to use.
    df = pd.DataFrame()
    for i in range(len(ydata)):
        x = xdata
        y = ydata[i]
        legend = np.array([data_labels[i] for j in range(ydata.shape[1])])
        df = pd.concat(
            [pd.DataFrame({xlabel: x, ylabel: y, "Legend": legend}), df])

    # Create a subplot into which the current data will be plotted
    ax = plt.subplot2grid(plot_shape, subplot_location,
                          colspan=subplot_colspan, rowspan=subplot_rowspan)

    # Create a line plot into the previosly created subplot
    sns.lineplot(x=xlabel, y=ylabel, hue="Legend", data=df, ax=ax)

    # Set plot parameters
    ax.set_title(title)
    ax.xlabel = xlabel
    ax.ylabel = ylabel
    ax.legend()
    ax.minorticks_on()

    # Set colors of the plot
    ax.grid(visible=True, which='major', linestyle='-', color="#005600")
    ax.grid(visible=True, which='minor',
            linestyle='--', alpha=0.8, color="#005600")
    ax.set_facecolor("#222222")

    return ax


ai_task, ao_task, reader = initialize_device(
    DEV, AI_CH, AO_CH, TRIG_SRC, NO_OF_SAMPLES, sampling_rate=SAMPLING_RATE)  # Initialize the DAQ device


try:
    start_tasks(ai_task, ao_task, reader, AO_VS, T_TO_CHRG)  # Start the tasks
    # Generate x-axis values: a numpy array with whole numbers in the range of [0, <length of samples buffer>)
    # divided by sampling rate gives us the point in time each sample was measured relative to the time the first sample was measured.
    xdata = (np.linspace(
        0, samples_buf.shape[1], samples_buf.shape[1]))/SAMPLING_RATE

    # Assume that the first measurment in the buffer is the measurment of voltage on the capacitor.
    capacitor_chraging_curve = samples_buf[0]

    # Differential evolution parameters
    popsize = 15
    its = 150
    mut = 0.3
    crossp = 0.5
    bounds = [(22E3, 22E3), (0, 100000), (0.47E-6, 0.47E-6)]

    ########################################################################################
    while True:
        de_result = DE_alg_RRC.main(xdata, capacitor_chraging_curve, [
                                    popsize, its, mut, crossp, bounds, AO_VS])  # Run differential evolution

        # Extract the results of differential evolution
        params = de_result[0]
        # Each row of params contains values for R1, R2 and C. To plot the evolution of each of the values,
        # we transpose the array containing these results.
        param_iterations = params.T
        R1 = np.array([param_iterations[0]])
        R2 = np.array([param_iterations[1]])
        C = np.array([param_iterations[2]])
        err = np.array([de_result[1]])
        opt_curves = de_result[2]
        # Select the last iteration of optimized curve to plot.
        opt_curve = np.array([opt_curves[-1]])

        # Print or plot only the best parameter set (last value) for each measurement.
        print("R1: ", R1[-1])
        print("R2: ", R2[-1])
        print("C: ", C[-1])

        # Initialize a numpy array for plotting the optimized curve generated with DE and the actually measured data on the same plot.
        curve_plot_y = np.zeros(
            (samples_buf.shape[0] + 1, samples_buf.shape[1]))

        # Assign the elements from the measurement buffer to buffer created in the previous step.
        for i in range(samples_buf.shape[0]):
            curve_plot_y[i] = samples_buf[i]
        # Append the optimized curve to the buffer to be plotted
        curve_plot_y[-1] = opt_curve

        plot_shape = (14, 8)

        # Plot the measured data and the optimized curve.
        graph_title = r"Charging of 1uF Capacitor in parallel with a potentiometer." + \
            '\n' + r"Fs=" + str(SAMPLING_RATE/1000) + "kHz"
        data_labels = ["Voltage on AO0",
                       "Measured voltage on capactior", "Fitted curve"]
        xlabel = "Sample number"
        ylabel = "Voltage (V)"
        ax1 = generate_graph(curve_plot_y, xdata=xdata, title=graph_title, xlabel=xlabel, ylabel=ylabel,
                             data_labels=data_labels, plot_shape=plot_shape, subplot_colspan=4, subplot_rowspan=4)

        # Plot convergence of DE algorithm.
        graph_title = "Convergence"
        data_labels = ["RMSE"]
        xlabel = "Iteration"
        ylabel = "RMS Error (V)"
        generate_graph(err, xdata=np.arange(err.shape[1]), title=graph_title, xlabel=xlabel, ylabel=ylabel,
                       data_labels=data_labels, plot_shape=plot_shape, subplot_location=(9, 0), subplot_colspan=4, subplot_rowspan=4)

        # Plot the values used for R1 through th curve optimization
        graph_title = "R1"
        data_labels = ["R1"]
        xlabel = "Measurment cycle"
        ylabel = "Resistance (Ohm)"
        generate_graph(R1, xdata=np.arange(R1.shape[1]), title=graph_title, xlabel=xlabel, ylabel=ylabel,
                       data_labels=data_labels, plot_shape=plot_shape, subplot_location=(0, 5), subplot_colspan=4, subplot_rowspan=3)

        # Plot the values used for R2 through th curve optimization
        graph_title = "R2"
        data_labels = ["R2"]
        xlabel = "Measurment cycle"
        ylabel = "Resistance (Ohm)"
        generate_graph(R2, xdata=np.arange(R2.shape[1]), title=graph_title, xlabel=xlabel, ylabel=ylabel,
                       data_labels=data_labels, plot_shape=plot_shape, subplot_location=(5, 5), subplot_colspan=4, subplot_rowspan=3)

        # Plot the values used for C through th curve optimization
        graph_title = "C"
        data_labels = ["C"]
        xlabel = "Measurment cycle"
        ylabel = "Capacitance (F)"
        generate_graph(C, xdata=np.arange(C.shape[1]), title=graph_title, xlabel=xlabel, ylabel=ylabel,
                       data_labels=data_labels, plot_shape=plot_shape, subplot_location=(10, 5), subplot_colspan=4, subplot_rowspan=3)

        # Show the plot.
        plt.tight_layout()
        ax1.get_figure().set_facecolor('#BFBFBF')
        plt.show()


except Exception as e:
    raise e
finally:  # Close the created tasks if the proigram exits
    ai_task.stop()
    ao_task.write(0)
    ao_task.stop()
    ai_task.close()
    ao_task.close()

# it only plots the potentiom........R2
