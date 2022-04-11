import sys
import nest
nest.Install("cerebmodule")
nest.Install("util_neurons_module")
data_folder = './Data/'
figure_folder = './Figures/'
import os
os.makedirs(data_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)
nest.set_verbosity("M_WARNING")
nest.SetKernelStatus({"overwrite_files": True})
res = nest.GetKernelStatus("resolution")

import numpy as np
import matplotlib.pyplot as plt
import statistics
from population_view_cristiano import PopView
import random
import time

n_trial = 1
trial_len = 500
FBK_REMOVAL = False
n_neurons = 100
time_vect = np.arange(0, trial_len, res)

def minimumJerk(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =   6*(x_des-x_init)/np.power(T_max,5)
    b = -15*(x_des-x_init)/np.power(T_max,4)
    c =  10*(x_des-x_init)/np.power(T_max,3)
    d =  np.zeros(x_init.shape)
    e =  np.zeros(x_init.shape)
    g =  x_init

    pol = np.array([a,b,c,d,e,g])
    pp  = a*np.power(tmspn,5) + b*np.power(tmspn,4) + c*np.power(tmspn,3) + g

    return pp, pol

def minimumJerk_ddt(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =  120*(x_des-x_init)/np.power(T_max,5)
    b = -180*(x_des-x_init)/np.power(T_max,4)
    c =  60*(x_des-x_init)/np.power(T_max,3)
    d =  np.zeros(x_init.shape)

    pol = np.array([a,b,c,d])
    pp  = a*np.power(tmspn,3) + b*np.power(tmspn,2) + c*np.power(tmspn,1) + d

    return pp, pol

def minimumJerk_dt(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =  30*(x_des-x_init)/np.power(T_max,5)
    b = -60*(x_des-x_init)/np.power(T_max,4)
    c =  30*(x_des-x_init)/np.power(T_max,3)
    d =  np.zeros(x_init.shape)
    e =  np.zeros(x_init.shape)

    pol = np.array([a,b,c,d])
    pp  = a*np.power(tmspn,4) + b*np.power(tmspn,3) + c*np.power(tmspn,2) + d*np.power(tmspn,1) + e

    return pp, pol

# Sigmoid = 100; parabola = 5e4

# Cerebellum
# cerebellum_target 100 e variability 0;
# cerebellum_target 0 e variability 1
cerebellum_target = 100 # 100
file_pattern = data_folder+'pattern_cerebellum.dat'
a_file = open(file_pattern, "w")
# Sinusoid
#pattern, pol = minimumJerk_ddt(np.array([0.0]), np.array([5e6]), time_vect)
# Sigmoid
pattern, pol = minimumJerk(np.array([0.0]), np.array([cerebellum_target]), time_vect)
# Parabola
#pattern, pol = minimumJerk_dt(np.array([0.0]), np.array([cerebellum_target]), time_vect)
np.savetxt(a_file, pattern)
a_file.close()

# Sensory feedback
sensory_target = -100 # 100
file_pattern = data_folder+'pattern_sensory.dat'
a_file = open(file_pattern, "w")
# Sinusoid
#pattern, pol = minimumJerk_ddt(np.array([0.0]), np.array([5e6]), time_vect)
# Sigmoid
fbk_delay = 100
silence = np.zeros(int(fbk_delay/res))
pattern, pol = minimumJerk(np.array([0.0]), np.array([sensory_target]), time_vect)
pattern = np.concatenate((silence, pattern.reshape(-1)))[:int(trial_len/res)]
plt.figure()
plt.plot(pattern)
noise = 30*np.sin(2*np.pi*(5/trial_len)*time_vect)
#pattern = [j + noise[i] for i,j in enumerate(pattern)]
plt.plot(pattern)
plt.show()
# Parabola
#pattern, pol = minimumJerk_dt(np.array([0.0]), np.array([sensory_target]), time_vect)
np.savetxt(a_file, pattern)
a_file.close()

params = {
    "base_rate": 30.0,
    "kp": 1.0,
    }

class Input_signal:
    def __init__(self, n, pathData, filename, time_vect, **kwargs):

        self.time_vect = time_vect

        # Path where to save the data file
        self.pathData = pathData

        # General parameters of neurons
        params = {
            "base_rate": 0.0,
            "kp": 1.0,
            "repeatable":False,
            }
        params.update(kwargs)

        # Initialize population arrays
        self.pops_p = []
        self.pops_n = []

        # Create populations
        file_pattern = self.pathData+filename

        # Negative population (joint i)
        tmp_pop_n = nest.Create("tracking_neuron",n=n, params=params)
        nest.SetStatus(tmp_pop_n, {"pos": False, "pattern_file": file_pattern})
        self.pops_n = PopView(tmp_pop_n,self.time_vect)

        # Positive population (joint i)
        tmp_pop_p = nest.Create("tracking_neuron", n=n, params=params)
        nest.SetStatus(tmp_pop_p, {"pos": True, "pattern_file": file_pattern})
        self.pops_p = PopView(tmp_pop_p,self.time_vect)

sensory_input = Input_signal(n=n_neurons, pathData=data_folder, filename='pattern_sensory.dat', time_vect=time_vect, **params)
cerebellum_input = Input_signal(n=n_neurons, pathData=data_folder, filename='pattern_cerebellum.dat',time_vect=time_vect, **params)

# Create Fbk population to modulate variability
variability = 0 # range: 0-1 
w = 0.01*(24*variability + 1)
conv = (4 - n_neurons)*variability + n_neurons
feedback_neurons_p = nest.Create("diff_neuron",n = n_neurons)
nest.SetStatus(feedback_neurons_p, {"kp": 1.0, "pos": True, "buffer_size": 25.0, "base_rate": 0.0})
feedback_neurons_n = nest.Create("diff_neuron",n = n_neurons)
nest.SetStatus(feedback_neurons_n, {"kp": 1.0, "pos": False, "buffer_size": 25.0, "base_rate": 0.0})
syn_exc = {"weight": w, "delay": 1.0}
syn_inh = {"weight": -w, "delay": 1.0}
#nest.Connect(sensory_input.pops_p.pop, feedback_neurons_p, {'rule': 'all_to_all'}, syn_spec=syn_exc)
#nest.Connect(sensory_input.pops_n.pop, feedback_neurons_n, {'rule': 'all_to_all'}, syn_spec=syn_inh)
nest.Connect(sensory_input.pops_p.pop, feedback_neurons_p, {'rule': 'fixed_indegree','indegree': conv}, syn_spec=syn_exc)
nest.Connect(sensory_input.pops_n.pop, feedback_neurons_n, {'rule': 'fixed_indegree','indegree': conv}, syn_spec=syn_inh)
print(w)
print(conv)

# Create Cerebellum population to modulate variability
# w = 0.5, conv = 2 # TODO aggiungere questo caso a "w" e "conv"
# w = 0.2, conv = 5
# w = 0.01, conv = n_neurons
if cerebellum_target == 100:
    variability = 0
elif cerebellum_target == 0:
    variability = 1
elif cerebellum_target == 50:
    variability = 1
else:
    print('Errore cerebellum target')
    sys.exit() 
w = 0.01*(99*variability + 1) # 19
# variability = 0 => w = 0.01
# variability = 1 => w = 0.2
conv = int((1 - n_neurons)*variability + n_neurons)
# variability = 0 => conv = n_neurons
# variability = 1 => conv = 5
cerebellar_neurons_p = nest.Create("diff_neuron",n = n_neurons)
nest.SetStatus(cerebellar_neurons_p, {"kp": 1.0, "pos": True, "buffer_size": 25.0, "base_rate": 0.0})
cerebellar_neurons_n = nest.Create("diff_neuron",n = n_neurons)
nest.SetStatus(cerebellar_neurons_n, {"kp": 1.0, "pos": False, "buffer_size": 25.0, "base_rate": 0.0})
syn_exc = {"weight": w, "delay": res}
syn_inh = {"weight": -w, "delay": res}
nest.Connect(cerebellum_input.pops_p.pop, cerebellar_neurons_p, {'rule': 'fixed_indegree','indegree': conv}, syn_spec=syn_exc)
nest.Connect(cerebellum_input.pops_n.pop, cerebellar_neurons_n, {'rule': 'fixed_indegree','indegree': conv}, syn_spec=syn_inh)
print(w)
print(conv)

# Create State Estimator neurons
state_neurons_p = nest.Create("state_neuron", n_neurons)
nest.SetStatus(state_neurons_p, {"kp": 1.0, "pos": True, "buffer_size": 20.0, "base_rate": 0.0})
state_neurons_n = nest.Create("state_neuron", n_neurons)
nest.SetStatus(state_neurons_n, {"kp": 1.0, "pos": False, "buffer_size": 20.0, "base_rate": 0.0})
# Connect inputs
syn_1 = {"weight": 1.0, "receptor_type": 1}
syn_2 = {"weight": 1.0, "receptor_type": 2}
# Positive neurons
nest.Connect(cerebellar_neurons_p, state_neurons_p, "all_to_all", syn_spec=syn_1)
nest.Connect(feedback_neurons_p, state_neurons_p, "all_to_all", syn_spec=syn_2)
nest.SetStatus(state_neurons_p, {"num_first": float(n_neurons), "num_second": float(n_neurons)})
# Negative neurons
nest.Connect(cerebellar_neurons_n, state_neurons_n, "all_to_all", syn_spec=syn_1)
nest.Connect(feedback_neurons_n, state_neurons_n, "all_to_all", syn_spec=syn_2)
nest.SetStatus(state_neurons_n, {"num_first": float(n_neurons), "num_second": float(n_neurons)})


# Create and Connect spikedetectors
# Cerebellum
spikedetector_cer_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True,"to_file":True, "label": "Cerebellum Pos"})
nest.Connect(cerebellar_neurons_p, spikedetector_cer_pos)
spikedetector_cer_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True,"to_file":True, "label": "Cerebellum Neg"})
nest.Connect(cerebellar_neurons_n, spikedetector_cer_neg)
# Feedback
spikedetector_feedback_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True,"to_file":True, "label": "Feedback Pos"})
nest.Connect(feedback_neurons_p, spikedetector_feedback_pos)
spikedetector_feedback_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True,"to_file":True, "label": "Feedback Neg"})
nest.Connect(feedback_neurons_n, spikedetector_feedback_neg)
# State Estimator
spikedetector_state_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True,"to_file":True, "label": "State Pos"})
nest.Connect(state_neurons_p, spikedetector_state_pos)
spikedetector_state_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True,"to_file":True, "label": "State Neg"})
nest.Connect(state_neurons_n, spikedetector_state_neg)

nest.SetKernelStatus({"overwrite_files": True,"data_path": data_folder})
if FBK_REMOVAL:
    conns_pos = nest.GetConnections(sensory_input.pops_p.pop, feedback_neurons_p)
    conns_neg = nest.GetConnections(sensory_input.pops_n.pop, feedback_neurons_n)
    time_pre = time.time()
    nest.Simulate(int(trial_len/2))
    #Remove fbk connections
    nest.SetStatus(conns_pos, {'weight': 0.0})
    nest.SetStatus(conns_neg, {'weight': 0.0})
    nest.Simulate(int(trial_len/2))
    print('Simulation lasted {} seconds'.format(time.time() - time_pre))
else:
    for trial in range(n_trial):
        print('Simulating trial:',trial+1)
        time_pre = time.time()
        nest.Simulate(trial_len)
        print('Simulation lasted {} seconds'.format(time.time() - time_pre))

dSD_cer_pos = nest.GetStatus(spikedetector_cer_pos, keys="events")[0]
evs_cer_pos = dSD_cer_pos["senders"]
ts_cer_pos = dSD_cer_pos["times"]
dSD_cer_neg = nest.GetStatus(spikedetector_cer_neg, keys="events")[0]
evs_cer_neg = dSD_cer_neg["senders"]
ts_cer_neg = dSD_cer_neg["times"]

dSD_feedback_pos = nest.GetStatus(spikedetector_feedback_pos, keys="events")[0]
evs_feedback_pos = dSD_feedback_pos["senders"]
ts_feedback_pos = dSD_feedback_pos["times"]
dSD_feedback_neg = nest.GetStatus(spikedetector_feedback_neg, keys="events")[0]
evs_feedback_neg = dSD_feedback_neg["senders"]
ts_feedback_neg = dSD_feedback_neg["times"]

dSD_state_pos = nest.GetStatus(spikedetector_state_pos, keys="events")[0]
evs_state_pos = dSD_state_pos["senders"]
ts_state_pos = dSD_state_pos["times"]
dSD_state_neg = nest.GetStatus(spikedetector_state_neg, keys="events")[0]
evs_state_neg = dSD_state_neg["senders"]
ts_state_neg = dSD_state_neg["times"]

y_min = np.min(evs_cer_neg)
y = [i-y_min for i in evs_cer_neg]
plt.figure(figsize=(10,8))
plt.scatter(ts_cer_neg, y, marker='.', color='blue')
y = [i-y_min for i in evs_cer_pos]
plt.scatter(ts_cer_pos, y, marker='.', color='red')
plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 5),fontsize=25)
plt.yticks(fontsize=25)
plt.title('Scatter plot Cerebellum', size =25)
plt.xlabel('Time [ms]', size = 25)
plt.ylabel('Neuron ID', size = 25)
plt.savefig(figure_folder+'Scatter plot cerebellum.svg')

plt.figure(figsize=(10,8))
x = []
delta_t = 20
for i in range(0,trial_len*n_trial,delta_t):
    n_spikes = len([k for k in ts_cer_pos if k<i+delta_t and k>=i])
    freq = n_spikes/(delta_t/1000*(n_neurons/2))
    x.append(freq)
cerebellar_prediction_pos = x
plt.plot(range(0,trial_len*n_trial,delta_t), x, color='red')
x = []
delta_t = 20
for i in range(0,trial_len*n_trial,delta_t):
    n_spikes = len([k for k in ts_cer_neg if k<i+delta_t and k>=i])
    freq = n_spikes/(delta_t/1000*(n_neurons/2))
    x.append(freq)
cerebellar_prediction_neg = x
plt.plot(range(0,trial_len*n_trial,delta_t), x, color='blue')
plt.title('Spike frequency Cerebellum', size =25)
plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 5),fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
ax = plt.gca()
lims = ax.get_ylim()
ax.set_ylim([0,lims[1]])
plt.savefig(figure_folder+'Spike frequency cerebellum.svg')

y_min = np.min(evs_feedback_neg)
y = [i-y_min for i in evs_feedback_neg]
plt.figure(figsize=(10,8))
plt.scatter(ts_feedback_neg, y, marker='.', color='blue')
y = [i-y_min for i in evs_feedback_pos]
plt.scatter(ts_feedback_pos, y, marker='.', color='red')
plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 5),fontsize=25)
plt.yticks(fontsize=25)
plt.title('Scatter plot sensory feedback', size =25)
plt.xlabel('Time [ms]', size = 25)
plt.ylabel('Neuron ID', size = 25)
plt.savefig(figure_folder+'Scatter plot sensory feedback.svg')

x_pos = []
x_neg = []
feedback_signal = []
feedback_signal_pos = []
feedback_signal_neg = []
delta_t = 20
for i in range(0,trial_len*n_trial,delta_t):
    n_spikes_pos = len([k for k in ts_feedback_pos if k<i+delta_t and k>=i])
    n_spikes_neg = len([k for k in ts_feedback_neg if k<i+delta_t and k>=i])
    freq_pos = n_spikes_pos/(delta_t/1000*int(n_neurons/2))
    freq_neg = n_spikes_neg/(delta_t/1000*int(n_neurons/2))
    x_pos.append(freq_pos)
    x_neg.append(freq_neg)
    feedback_signal_pos.append(x_pos[-1])
    feedback_signal_neg.append(x_neg[-1])
    feedback_signal.append(x_pos[-1]-x_neg[-1])
plt.figure(figsize=(10,8))
plt.plot(range(0,trial_len*n_trial,delta_t), x_pos, color='red')
plt.plot(range(0,trial_len*n_trial,delta_t), x_neg, color='blue')
plt.title('Spike frequency sensory feedback', size =25)
plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 5),fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
plt.savefig(figure_folder+'Spike frequency sensory feedback.svg')

x_pos = []
x_neg = []
state_signal = []
state_signal_pos = []
state_signal_neg = []
delta_t = 20
for i in range(0,trial_len*n_trial,delta_t):
    n_spikes_pos = len([k for k in ts_state_pos if k<i+delta_t and k>=i])
    n_spikes_neg = len([k for k in ts_state_neg if k<i+delta_t and k>=i])
    freq_pos = n_spikes_pos/(delta_t/1000*int(n_neurons/2))
    freq_neg = n_spikes_neg/(delta_t/1000*int(n_neurons/2))
    x_pos.append(freq_pos)
    x_neg.append(freq_neg)
    state_signal_pos.append(x_pos[-1])
    state_signal_neg.append(x_neg[-1])
    state_signal.append(x_pos[-1]-x_neg[-1])
plt.figure(figsize=(10,8))
plt.plot(range(0,trial_len*n_trial,delta_t), x_pos, color='red')
plt.plot(range(0,trial_len*n_trial,delta_t), x_neg, color='blue')
plt.title('Spike frequency State Estimator', size =25)
plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 5),fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
plt.savefig(figure_folder+'Spike frequency State Estimator.svg')

times_cereb = {}
ids_pos_cereb = cerebellar_neurons_p
for id_value in ids_pos_cereb:
    times_cereb[id_value] = [t for i,t in enumerate(ts_cer_pos) if evs_cer_pos[i] == id_value]
ids_neg_cereb = cerebellar_neurons_n
for id_value in ids_neg_cereb:
    times_cereb[id_value] = [t for i,t in enumerate(ts_cer_neg) if evs_cer_neg[i] == id_value]

reliability_pred_pos = []
reliability_pred_neg = []
delta_t = 20
for i in range(0,trial_len*n_trial,delta_t):
    freq_pos = []
    freq_neg = []
    for id_value in times_cereb.keys():
        spikes = [t for t in times_cereb[id_value] if t<i+delta_t and t>=i]
        if id_value in ids_pos_cereb:
            freq_pos.append(len(spikes)/(delta_t/1000))
        else:
            freq_neg.append(len(spikes)/(delta_t/1000))
    var_pos = statistics.pvariance(freq_pos)
    var_neg = statistics.pvariance(freq_neg)
    reliability_pred_pos.append(var_pos/(np.mean(freq_pos)))
    reliability_pred_neg.append(var_neg/(np.mean(freq_neg)))
print('Variability prediction pos: {} +- {}'.format(np.mean(reliability_pred_pos), np.std(reliability_pred_pos)))
print('Variability prediction neg: {} +- {}'.format(np.mean(reliability_pred_neg), np.std(reliability_pred_neg)))
plt.figure(figsize=(10,8))
plt.plot(np.arange(0,trial_len*n_trial,delta_t),reliability_pred_neg, color=(255/255, 51/255, 0))
plt.plot(np.arange(0,trial_len*n_trial,delta_t),reliability_pred_pos, color=(176/255, 34/255, 0))
plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 5),fontsize=25)
plt.yticks(fontsize=25)
ax_pred = plt.gca()
lims_pred = ax_pred.get_ylim()
plt.ylabel('Variability of the firing rates [Hz]', size =25)
plt.xlabel('Time [ms]', size =25)
plt.grid(True)
plt.title('Variability of the cerebellar prediction', size =25)
plt.savefig(figure_folder+'Variability of the cerebellar prediction.svg')
variability_cer_pos = np.concatenate((np.arange(0,trial_len*n_trial,delta_t), reliability_pred_pos))
np.save(data_folder+"variability_cer_pos", variability_cer_pos)
variability_cer_neg = np.concatenate((np.arange(0,trial_len*n_trial,delta_t), reliability_pred_neg))
np.save(data_folder+"variability_cer_neg", variability_cer_neg)

times_feedback = {}
ids_pos_feedback = feedback_neurons_p
for id_value in ids_pos_feedback:
    times_feedback[id_value] = [t for i,t in enumerate(ts_feedback_pos) if evs_feedback_pos[i] == id_value]
ids_neg_feedback = feedback_neurons_n
for id_value in ids_neg_feedback:
    times_feedback[id_value] = [t for i,t in enumerate(ts_feedback_neg) if evs_feedback_neg[i] == id_value]

reliability_feedback_pos = []
reliability_feedback_neg = []
delta_t = 20
for i in range(0,trial_len*n_trial,delta_t):
    freq_pos = []
    freq_neg = []
    for id_value in times_feedback.keys():
        spikes = [t for t in times_feedback[id_value] if t<i+delta_t and t>=i]
        if id_value in ids_pos_feedback:
            freq_pos.append(len(spikes)/(delta_t/1000))
        else:
            freq_neg.append(len(spikes)/(delta_t/1000))
    var_pos = statistics.pvariance(freq_pos)#/(np.mean(freq_pos)**2)
    var_neg = statistics.pvariance(freq_neg)#/(np.mean(freq_neg)**2)
    # Add fake noise lower than cerebellum noise when prediction = 0
    var_pos = var_pos + 0.3*random.randint(int(var_pos), int(var_pos))
    var_neg = var_neg + 0.3*random.randint(int(var_neg), int(var_neg))
    reliability_feedback_pos.append(var_pos/(np.mean(freq_pos)))
    reliability_feedback_neg.append(var_neg/(np.mean(freq_neg)))
print('Variability feedback pos: {} +- {}'.format(np.mean(reliability_feedback_pos[3:]), np.std(reliability_feedback_pos[3:])))
print('Variability feedback neg: {} +- {}'.format(np.mean(reliability_feedback_neg[3:]), np.std(reliability_feedback_neg[3:])))
plt.figure(figsize=(10,8))
plt.plot(np.arange(0,trial_len*n_trial,delta_t),reliability_feedback_neg, color=(0, 192/255, 0))
plt.plot(np.arange(0,trial_len*n_trial,delta_t),reliability_feedback_pos, color=(0, 126/255, 0))
plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 5),fontsize=25)
plt.yticks(fontsize=25)
ax_fbk = plt.gca()
lims_fbk = ax_fbk.get_ylim()
ax_fbk.set_ylim([0, max(lims_pred[1], lims_fbk[1])])
ax_pred.set_ylim([0, max(lims_pred[1], lims_fbk[1])])
ax_pred.figure.savefig(figure_folder+'Variability of the sensory feedback.svg')
plt.ylabel('Variability of the firing rates [Hz]', size =25)
plt.xlabel('Time [ms]', size =25)
plt.title('Variability of the sensory feedback', size =25)
plt.grid(True)
plt.savefig(figure_folder+'Variability of the sensory feedback.svg')
variability_feedback_pos = np.concatenate((np.arange(0,trial_len*n_trial,delta_t), reliability_feedback_pos))
np.save(data_folder+"variability_fbk_pos", variability_feedback_pos)
variability_feedback_neg = np.concatenate((np.arange(0,trial_len*n_trial,delta_t), reliability_feedback_neg))
np.save(data_folder+"variability_fbk_neg", variability_feedback_neg)

outputs = []
for i in range(len(cerebellar_prediction_pos)):
    feedback_variance = 0.5 #+ abs(feedback_signal[i]*0.003)
    #kalman_gain_pos = reliability_pred_pos[i]/(feedback_variance+reliability_pred_pos[i])
    #kalman_gain_neg = reliability_pred_neg[i]/(feedback_variance+reliability_pred_neg[i])
    kalman_gain_pos = reliability_pred_pos[i]/(reliability_feedback_pos[i]+reliability_pred_pos[i])
    kalman_gain_neg = reliability_pred_neg[i]/(reliability_feedback_neg[i]+reliability_pred_neg[i])
    output_value_pos = kalman_gain_pos*feedback_signal_pos[i] + (1-kalman_gain_pos)*cerebellar_prediction_pos[i]
    output_value_neg = kalman_gain_neg*feedback_signal_neg[i] + (1-kalman_gain_neg)*cerebellar_prediction_neg[i]
    if np.isnan(output_value_pos) or np.isnan(output_value_neg):
        outputs.append(cerebellar_prediction_pos[i] - cerebellar_prediction_neg[i])
    else:
        outputs.append(output_value_pos-output_value_neg)

plt.figure()
plt.plot(np.arange(0,trial_len*n_trial,20),cerebellar_prediction_pos,label = 'cerebellum', color='saddlebrown')
plt.plot(np.arange(0,trial_len*n_trial,20),feedback_signal_pos,label = 'feedback', color='gold')
plt.legend()

plt.figure()
plt.plot(np.arange(0,trial_len*n_trial,20),cerebellar_prediction_neg,label = 'cerebellum', color='saddlebrown')
plt.plot(np.arange(0,trial_len*n_trial,20),feedback_signal_neg,label = 'feedback', color='gold')
plt.legend()

feedback_signal_pos = np.array(feedback_signal_pos)
feedback_signal_neg = np.array(feedback_signal_neg)
cerebellar_prediction_pos = np.array(cerebellar_prediction_pos)
cerebellar_prediction_neg = np.array(cerebellar_prediction_neg)
state_signal_pos = np.array(state_signal_pos)
state_signal_neg = np.array(state_signal_neg)
np.save(data_folder+"feedback_signal_pos", feedback_signal_pos)
np.save(data_folder+"feedback_signal_neg", feedback_signal_neg)
np.save(data_folder+"cerebellar_prediction_pos", cerebellar_prediction_pos)
np.save(data_folder+"cerebellar_prediction_neg", cerebellar_prediction_neg)
np.save(data_folder+"state_signal_pos", state_signal_pos)
np.save(data_folder+"state_signal_neg", state_signal_neg)

plt.figure(figsize=(12,8))
#plt.plot(np.arange(0,trial_len*n_trial,20), outputs, label='State', linewidth=5)
plt.plot(np.arange(0,trial_len*n_trial,20), feedback_signal_pos - feedback_signal_neg, label='Feedback diff', color='gold', linewidth=5)
plt.plot(np.arange(0,trial_len*n_trial,20), cerebellar_prediction_pos - cerebellar_prediction_neg, label='Cerebellum diff', color='saddlebrown', linewidth=5)
plt.plot(np.arange(0,trial_len*n_trial,20), state_signal_pos - state_signal_neg, label='State Estimator', color='b', linewidth=5)
plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 5),fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel('Firing rate [Hz]', size = 25)
plt.xlabel('Time [ms]', size = 25)
plt.title('Bayesian integration of prediction and sensory feedback', size =25)
plt.savefig(figure_folder+'Bayesian integration of prediction and sensory feedback.svg')

'''
plt.figure(figsize=(10,8))
x = []
delta_t = 30
for i in range(0,trial_len,delta_t):
    n_spikes = len([k for k in ts_cer_pos if k<i+delta_t and k>=i])
    freq = n_spikes/(delta_t/1000*(n_neurons/2))
    x.append(freq)
cerebellar_prediction_pos = x
plt.plot(range(0,trial_len,delta_t), x, color='red')
x = []
delta_t = 30
for i in range(0,trial_len,delta_t):
    n_spikes = len([k for k in ts_cer_neg if k<i+delta_t and k>=i])
    freq = n_spikes/(delta_t/1000*(n_neurons/2))
    x.append(freq)
cerebellar_prediction_neg = x
plt.plot(range(0,trial_len,delta_t), x, color='blue')
plt.title('Spike frequency Cerebellum', size =25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
ax = plt.gca()
lims = ax.get_ylim()
ax.set_ylim([0,lims[1]])

x_pos = []
x_neg = []
feedback_signal = []
feedback_signal_pos = []
feedback_signal_neg = []
delta_t = 30
for i in range(0,trial_len,delta_t):
    n_spikes_pos = len([k for k in ts_feedback_pos if k<i+delta_t and k>=i])
    n_spikes_neg = len([k for k in ts_feedback_neg if k<i+delta_t and k>=i])
    freq_pos = n_spikes_pos/(delta_t/1000*int(n_neurons/2))
    freq_neg = n_spikes_neg/(delta_t/1000*int(n_neurons/2))
    x_pos.append(freq_pos)
    x_neg.append(freq_neg)
    feedback_signal_pos.append(x_pos[-1])
    feedback_signal_neg.append(x_neg[-1])
    feedback_signal.append(x_pos[-1]-x_neg[-1])
plt.figure(figsize=(10,8))
plt.plot(range(0,trial_len,delta_t), x_pos, color='red')
plt.plot(range(0,trial_len,delta_t), x_neg, color='blue')
plt.title('Spike frequency sensory feedback', size =25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Time [ms]', size =25)
plt.ylabel('Frequency [Hz]', size =25)
ax = plt.gca()
lims = ax.get_ylim()
ax.set_ylim([0,lims[1]])


y_min = np.min(evs_cer_neg)
y = [i-y_min for i in evs_cer_neg]
plt.figure(figsize=(10,8))
plt.scatter(ts_cer_neg, y, marker='.', color='blue')
y_min = np.min(evs_cer_pos)
y = [i-y_min+max(y) for i in evs_cer_pos]
plt.scatter(ts_cer_pos, y, marker='.', color='red')
ax = plt.gca()
ax.set_xlim([0, 500])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Scatter plot Cerebellum', size =25)
plt.xlabel('Time [ms]', size = 25)
plt.ylabel('Neuron ID', size = 25)

y_min = np.min(evs_feedback_neg)
y = [i-y_min for i in evs_feedback_neg]
plt.figure(figsize=(10,8))
plt.scatter(ts_feedback_neg, y, marker='.', color='blue')
y_min = np.min(evs_feedback_pos)
y = [i-y_min+max(y) for i in evs_feedback_pos]
plt.scatter(ts_feedback_pos, y, marker='.', color='red')
ax = plt.gca()
ax.set_xlim([0, 500])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Scatter plot sensory feedback', size =25)
plt.xlabel('Time [ms]', size = 25)
plt.ylabel('Neuron ID', size = 25)
'''

plt.show()