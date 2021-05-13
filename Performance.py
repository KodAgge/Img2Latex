import numpy as np
import editdistance as ed
import statistics as stats
from nltk import bleu_score

class Performance:
    
    def __init__(self, predictions, ground_truth, get_stats):
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.get_stats = get_stats

        #CHECK DataTypes
        assert type(self.predictions) == type(self.ground_truth), 'Non-matching data-types'    

    def levenshtein(self):
        labels = []
        distances = []

        for pred, real in zip(self.predictions, self.ground_truth):
            dist =  ed.eval(pred, real)
            labels.append([pred, real])
            distances.append(dist)
        
        if self.get_stats:
            return stats.mean(distances), stats.stdev(distances), labels, distances
        
        return distances

    def bleu(self):
        scores_bleu = []
        labels= []

        for pred, real in zip(self.predictions, self.ground_truth):
            labels.append([pred, real])
            pred, real = [pred.split()], real.split()

            score =  bleu_score.sentence_bleu(pred, real)
            scores_bleu.append(score)

        if self.get_stats:
            return stats.mean(scores_bleu), stats.stdev(scores_bleu), labels, scores_bleu

        return labels, scores_bleu

        

predictions = ['This should match all the time', 'This should not match all the time']
ground_truth = ['This should match all the time','Hey, this should not match all the time']


performance = Performance(predictions, ground_truth, get_stats=True)
lev_mean, lev_stdev, lev_labels, lev_distances = performance.levenshtein()
bleu_mean, bleu_stdev, bleu_labels, bleu_distances = performance.bleu()

assert len(lev_labels) == len(bleu_labels), "Labels not of same length"
print('PERFORMANCE')

for i in range(len(lev_labels)):
    print('Prediction: ', lev_labels[i][0])
    print('Ground Truth: ', lev_labels[i][1])
    print(f'Levenshtein Distance: {lev_distances[i]} | BLEU Score: {bleu_distances[i]}')
    print()


