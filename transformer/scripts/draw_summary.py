import sys
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def smooth_moving_average(x, n):
    fil = np.ones(n)/n
    smoothed = np.convolve(x, fil, mode='valid')
    smoothed = np.concatenate((x[:n-1], smoothed), axis=0)
    
    return smoothed

def moving_stdev(x, n):
    fil = np.ones(n)/n
    avg_sqare = np.convolve(np.power(x, 2), fil, mode='valid')
    squared_avg = np.power(np.convolve(x, fil, mode='valid'), 2)
    var = avg_sqare - squared_avg
    stdev = np.sqrt(var)
    #pad first few values
    stdev = np.concatenate(([0]*(n-1), stdev), axis=0)
    
    return stdev



jlog = defaultdict(list)
jlog['parameters'] = {}

with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        line_dict = json.loads(line[5:])
        if line_dict['type'] == 'LOG':
            if line_dict['step'] == 'PARAMETER':
                jlog['parameters'].update(line_dict['data'])
            else:
                for k, v in line_dict['data'].items():
                    jlog[k].append((line_dict['step'], line_dict['elapsedtime'], v))

fig, ax = plt.subplots(figsize=(20,5))

# Plot smoothed loss curve
steps = [x[0] for x in jlog['loss'] if isinstance(x[0], int)]
loss = [x[2] for x in jlog['loss'] if isinstance(x[0], int)]
smoothed_loss = smooth_moving_average(loss, 150)
stdev = moving_stdev(loss, 150)

ax.plot(steps, smoothed_loss, label='Training loss')
ax.plot(steps, smoothed_loss + stdev, '--', color='orange', linewidth=0.3, label='Stdev')
ax.plot(steps, smoothed_loss - stdev, '--', color='orange', linewidth=0.3)

# Plot validation loss curve
val_steps = [x[0] for x in jlog['val_loss'] if isinstance(x[0], int)]
val_loss = [x[2] for x in jlog['val_loss'] if isinstance(x[0], int)]
ax.plot(val_steps, val_loss, color='blue', label='Validation loss')

min_val_loss_step = val_steps[np.argmin(val_loss)]
ax.axvline(min_val_loss_step, linestyle='dashed', color='blue', linewidth=0.5, label='Validation loss minimum')

# Plot BLEU curves
val_bleu = [x[2] for x in jlog['val_bleu'] if isinstance(x[0], int)]
ax2 = ax.twinx()
ax2.plot(val_steps, val_bleu, color='red', label='Validation BLEU')
if 'test_bleu' in jlog:
    test_bleu = [x[2] for x in jlog['test_bleu']]
    ax2.plot(val_steps, test_bleu, color='pink', label='Test BLEU')
ax2.tick_params(axis='y')

ax.legend(loc='upper left', bbox_to_anchor=(1,1))
ax2.legend(loc='upper left', bbox_to_anchor=(1,0.5))
plt.grid()
plt.savefig(sys.argv[2])
