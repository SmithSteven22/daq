import nidaqmx
from nidaqmx.constants import TerminalConfiguration


ai_task = nidaqmx.Task("AI TASK")
ai_task.ai_channels.add_ai_voltage_chan(
    "Dev4/ai0", terminal_config=TerminalConfiguration.RSE)

ao_task = nidaqmx.Task("AO TASK")
ao_task.ao_channels.add_ao_voltage_chan(
    "Dev4/ao0")

datapoint_out = ao_task.write([1.2])

datapoint = ai_task.read()

print("Ai0: ", datapoint)
print("Ao0: ", datapoint_out)

ai_task.close()
ao_task.close()
