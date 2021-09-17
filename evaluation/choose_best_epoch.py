import csv
import json
import sys
import torch
import numpy as np

'''
Chooses the best F-score (among 100 epochs) based on a criterion (Reward & Actor_loss).
Takes as input the path to .csv file with all the loss functions 
and a .txt file with the F-Scores (for each split).
Prints a scalar that represents the average best F-score value,
and a scalar that represents the epoch number
'''

def use_logs(logs_file, f_scores):
    losses = {}
    losses_names = []

    with open(logs_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for (i, row) in enumerate(csv_reader):
            if i == 0:
                for col in range(len(row)):
                    losses[row[col]] = []
                    losses_names.append(row[col])
            else:
                for col in range(len(row)):
                    losses[losses_names[col]].append(float(row[col]))

    # criterion: Reward & Actor_loss
    actor = losses['actor_loss_epoch']
    reward = losses['reward_epoch']

    actor_t = torch.tensor(actor)
    reward_t = torch.tensor(reward)

    # Normalize values
    actor_t = abs(actor_t)
    actor_t = actor_t/max(actor_t)
    reward_t = reward_t/max(reward_t)

    product = (1 - actor_t)*reward_t

    epoch = torch.argmax(product).item()

    return np.round(f_scores[epoch], 2), epoch

# with args
# exp_path = sys.argv[1]
# dataset = sys.argv[2]

# without args
exp_path = 'exp1/'
dataset = 'TVSum'
split = 0   # change this number if you use different split

path = exp_path + dataset   # change this path if you use different structure for your directories inside the experiment
results_file = path + '/results/split' + str(split) + '/f_scores.txt'
logs_file = path + '/logs/split' + str(split) + '/scalars.csv'

# read F-Scores
with open(results_file) as f:
    f_scores = json.loads(f.read()) # list of F-Scores

fscore, best_epoch = use_logs(logs_file, f_scores)

print('Best epoch: ', best_epoch)
print('F-score: ', fscore)