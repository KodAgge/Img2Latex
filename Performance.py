import numpy as np
import editdistance as ed
import statistics as stats
from nltk import bleu_score

class Performance:
    
    def __init__(self, predictions, ground_truth):
        self.predictions = predictions
        self.ground_truth = ground_truth

        #CHECK DataTypes
        assert type(self.predictions) == type(self.ground_truth), 'Non-matching data-types'    

    def levenshtein(self):
        labels = []
        scores_lev = []

        for pred, real in zip(self.predictions, self.ground_truth):
            dist =  ed.eval(pred, real)
            labels.append([pred, real])
            scores_lev.append(dist)
        
        if self.get_stats:
            return stats.mean(scores_lev), stats.stdev(scores_lev), labels, scores_lev
        
        return labels, scores_lev

    def bleu(self):
        scores_bleu = []
        bleu_labels= []

        for pred, real in zip(self.predictions, self.ground_truth):
            bleu_labels.append([pred, real])
            pred, real = [pred.split()], real.split()

            score =  bleu_score.sentence_bleu(pred, real)
            scores_bleu.append(score)

        if self.get_stats:
            return stats.mean(scores_bleu), stats.stdev(scores_bleu), bleu_labels, scores_bleu

        return bleu_labels, scores_bleu


    def get_statistics(self, lev_scores, bleu_scores):
        lev_average, lev_stdev = stats.mean(lev_scores), stats.stdev(lev_scores)
        bleu_average, bleu_stdev = stats.mean(bleu_scores), stats.stdev(bleu_scores)

        print(20*'=')
        print('STATISTICS')
        print(20*'=')

        print(f'Levenshtein || Average score: {lev_average} | STDEV: {lev_stdev}')
        print(f'Bleu        || Average score: {bleu_average} | STDEV: {bleu_stdev}')


    def get_performance(self, lev_labels=None, lev_scores=None, bleu_labels=None, bleu_scores=None):
        print(20*'=')
        print('PERFORMANCE LEVENSHTEIN')
        print(20*'=')

        if lev_labels and lev_scores is not None:
            print('Levenshtein')
            for i in range(len(lev_labels)):
                print('Prediction: ', lev_labels[i][0])
                print('Ground Truth: ', lev_labels[i][1])
                print(f'Levenshtein Distance: {lev_scores[i]}')
                print()

        print(20*'=')
        print('PERFORMANCE BLEU')
        print(20*'=')

        if bleu_labels and bleu_scores is not None:
            print('Bleu')
            for i in range(len(bleu_labels)):
                print('Prediction: ', bleu_labels[i][0])
                print('Ground Truth: ', bleu_labels[i][1])
                print(f'Bleu Distance: {bleu_scores[i]}')
                print()


'''

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


'''