import numpy as np
import editdistance as ed
import statistics as stats
from nltk import bleu_score
from sympy.parsing.latex import parse_latex
import warnings
warnings.simplefilter(action='ignore')
import torch
from CorpusHelper import unTokenize 
import matplotlib.pyplot as plt 


class Performance:
    
    def __init__(self, predictions, ground_truth):
        self.predictions = predictions
        self.ground_truth = ground_truth

        #CHECK DataTypes
        assert type(self.predictions) == type(self.ground_truth), 'Non-matching input-types'    
  
        print('Removing PAD tokens...\n')
        self.predictions, self.ground_truth = self.drop_padding()

        print('Done removing PAD tokens...\n')

    def drop_padding(self):
        padding1= 144 #Padding token
        padding2= 143 #END token
        predictions, ground_truth = [], []
        
        for pred, truth in zip(self.predictions, self.ground_truth):
            
            #If exists padding, remove it
            try:
                pred = list(filter(lambda a: a != padding1, pred))
                #pred = list(filter(lambda a: a != padding2, pred))
                predictions.append(pred)
            
            except:
                predictions.append(pred)
            
            #If exists or end token padding, remove it
            try:
                truth = list(filter(lambda a: a != padding1, truth))
                #truth = list(filter(lambda a: a != padding2, truth))
                ground_truth.append(truth)   

            except:
                ground_truth.append(truth)         
        
        return predictions, ground_truth

    def convert_to_arrays(self):
        predictions, ground_truth = [], []
        
        for predTensor, truthTensor in zip(self.predictions, self.ground_truth):
            predictions.append(np.array(predTensor))
            ground_truth.append(np.array(truthTensor))

        return predictions, ground_truth

    def levenshtein(self, get_best_prediction = False):
        labels = []
        predictions = []
        scores_lev = []

        for pred, real in zip(self.predictions, self.ground_truth):
            length = self.get_max_token_length(pred, real)
            #pred = [12, 13, 12]
            #real = [12, 13, 12]
            #pred, real = self.castListToString(pred), self.castListToString(real)
    
            dist = ed.distance(pred, real)
            #print(dist)
         
            labels.append([pred, real])
            normalized_lev = (length-dist)/length
            scores_lev.append(normalized_lev)
            '''
            print('Pred: ', pred)
            print('Real: ', real)
            print('MAX length (pred vs real): ', length)
            print('Dist: ', dist)
            print('Score: ', normalized_lev)
            print()
            print()
            '''

        if get_best_prediction:
            lev_dict_pred = {}
            
            for l, s in zip(labels, scores_lev):
                pred, real = l[0], l[1]
                key = 'Prediction: ' + str(pred) + " | " + "Real: " + str(real)
                lev_dict_pred[key] = s
               
            best_prediction = min(lev_dict_pred, key=lev_dict_pred.get)
            best_score = lev_dict_pred[best_prediction]
            
            return labels, scores_lev, best_prediction, best_score

        return labels, scores_lev

    def castListToString(self, array):
        array = [str(i) for i in array]
        array = ' '.join(array)

        return array
    
    def bleu(self, get_best_prediction=False):
        scores_bleu = []
        labels= []

        for pred, real in zip(self.predictions, self.ground_truth):
            pred, real = self.castListToString(pred), self.castListToString(real)
     
           
            labels.append([pred, real])
            pred, real = [pred.split()], real.split() #Required format as input

            score =  bleu_score.sentence_bleu(pred, real)
            scores_bleu.append(score)


        if get_best_prediction:
            bleu_dict_pred = {}
            
            for l, s in zip(labels, scores_bleu):
                pred, real = l[0], l[1]
                key = 'Prediction: ' + str(pred) + " | " + "Real: " + str(real)
                bleu_dict_pred[key] = s
               
            best_prediction = max(bleu_dict_pred, key=bleu_dict_pred.get)
            best_score = bleu_dict_pred[best_prediction]
            
            return labels, scores_bleu, best_prediction, best_score
        return labels, scores_bleu

    def get_max_token_length(self, prediction, ground_truth):
        stop_character = 143
        truth_length = 0
        pred_length = 0
        for i in ground_truth:
            if i == 143:
                break

            truth_length +=1

        for j in prediction:
            if j == 143:
                break

            pred_length +=1

        return max(pred_length, truth_length)


    def grouped_bleu(self, plot = True):
        #scores = []
        #token_length = []
        scores_and_token_length = []

        for pred, real in zip(self.predictions, self.ground_truth):
            length = self.get_max_token_length(pred, real)

            pred, real = self.castListToString(pred), self.castListToString(real)
            pred, real = [pred.split()], real.split() #Required format as input
            score =  bleu_score.sentence_bleu(pred, real)
  
            scores_and_token_length.append([score, length])

        scores_and_token_length.sort(key=lambda pair: pair[1])

        scores = []
        length = []

        for i in scores_and_token_length:
            scores.append(i[0])
            length.append(i[1])

        scores_main = []
        length_main = []
        stepsize = 10
        groups = list(range(0, 120, stepsize))

        for group in groups: 
            scores_temp = []

            for s, l in zip(scores, length):
                if (l >= group and l <= group+stepsize):
                    scores_temp.append(s)

            try:
                length_main.append(group)
                scores_main.append(stats.mean(scores_temp))

            except:
                print('Missing data for group-range')
        
        length_main.remove(0)

        if plot:
            plt.plot(length_main, scores_main)
            plt.show()

    def equal_latex(self, expr1, expr2):
        expr1 = parse_latex(expr1)
        expr2 = parse_latex(expr2)
        return expr1.equals(expr2)

    def get_statistics(self, lev_scores, bleu_scores, jacc_scores, lms_scores, exact_matches, matching_sequences):

        lev_average, lev_stdev, lev_max, lev_min = stats.mean(lev_scores), stats.stdev(lev_scores), max(lev_scores), min(lev_scores)
        bleu_average, bleu_stdev, bleu_max, bleu_min = stats.mean(bleu_scores), stats.stdev(bleu_scores), max(bleu_scores), min(bleu_scores)
        jacc_average, jacc_stdev, jacc_max, jacc_min = stats.mean(jacc_scores), stats.stdev(jacc_scores), max(jacc_scores), min(jacc_scores)
        lms_average, lms_stdev, lms_max, lms_min = stats.mean(lms_scores), stats.stdev(lms_scores), max(lms_scores), min(lms_scores)
        

        print(20*'=')
        print('STATISTICS')
        print(20*'=')
        digits = 2

        print(f'Levenshtein  ==> Average score: {round(lev_average, digits)} \t| STDEV: {round(lev_stdev, digits)} \t| MAX (Best): {round(lev_max, digits)} \t| MIN (Worst): {round(lev_min, digits)}')
        print(f'BLEU         ==> Average score: {round(bleu_average, digits)}\t| STDEV: {round(bleu_stdev, digits)}\t| MAX (Best): {round(bleu_max, digits)} \t| MIN (Worst): {round(bleu_min, digits)}')
        print(f'Jaccard      ==> Average score: {round(jacc_average, digits)}\t| STDEV: {round(jacc_stdev, digits)}\t| MAX (Best): {round(jacc_max, digits)} \t| MIN (Worst): {round(jacc_min, digits)}')
        print(f'LMS          ==> Average score: {round(lms_average, digits)} \t| STDEV: {round(lms_stdev, digits)} \t| MAX (Best): {round(lms_max, digits)}  \t| MIN (Worst): {round(lms_min, digits)}')
        print(f'Exact matches: {round(exact_matches, digits)} %')

        print(20*'=')
        print('EXACT MATCHES')
        print(20*'=')
        
        print_matches = False

        if print_matches:
            for i in matching_sequences:
                print(f'PREDICTION: {i[0]} | GROUND TRUTH: {i[1]}') 

            print(20*'=')

    def get_performance(self, lev_labels=None, lev_scores=None, bleu_labels=None, bleu_scores=None):
        print(20*'=')
        print('PERFORMANCE LEVENSHTEIN')
        print(20*'=')
        printout_index = 1

        if lev_labels and lev_scores is not None:
            print('Levenshtein')
            for i in range(len(lev_labels)):
                if i % printout_index == 0: 
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
                if i % printout_index == 0: 
                    print('Prediction: ', bleu_labels[i][0])
                    print('Ground Truth: ', bleu_labels[i][1])
                    print(f'Bleu Distance: {bleu_scores[i]}')
                    print()

    def listToTensor(self):

        predictions = []
        ground_truth = []
        for pred, truth in zip(self.predictions, self.ground_truth):
            predictions.append(torch.tensor(pred))
            ground_truth.append(torch.tensor(truth))

        return predictions, ground_truth

    def jaccard(self, get_best_prediction=False):

        labels = []
        jacc_score = []

        for pred, real in zip(self.predictions, self.ground_truth):
            pred, real = set(pred), set(real)
            
            i = pred.intersection(real)
            u = pred.union(real)
            jacc_similarity = len(list(i))/len(list(u))
            
            labels.append([pred, real])
            jacc_score.append(jacc_similarity)

        if get_best_prediction:
            jacc_dict_pred = {}
            
            for l, s in zip(labels, jacc_score):
                pred, real = l[0], l[1]
                key = 'Prediction: ' + str(pred) + " | " + "Real: " + str(real)
                jacc_dict_pred[key] = s
               
            best_prediction = max(jacc_dict_pred, key=jacc_dict_pred.get)
            best_score = jacc_dict_pred[best_prediction]
            
            return labels, jacc_score, best_prediction, best_score
        

        return labels, jacc_score

    def LMS(self):
        lms_labels = []
        lms_scores = []
        

        for pred, real in zip(self.predictions, self.ground_truth):

            lms_labels.append([pred, real])
            sequence = []

            for p, r in zip(pred, real):
                if p == r:
                    sequence.append(1)

                else:
                    sequence.append(0)
                
            longest_match = 0
            current_match = 0
            matches = []
            
            
            for i in sequence:
                if i == 0:
                    current_match = 0
                    matches.append(current_match)

                else:
                    current_match +=1
                    matches.append(current_match)

            longest_match = max(matches)

         
            lms_scores.append(longest_match/len(matches))

        assert len(lms_labels) == len(lms_scores), 'Not of same length'

        return lms_labels, lms_scores

    def exact(self):
        num_exact_matches = 0
        matching_sequences = []
        for pred, real in zip(self.predictions, self.ground_truth):
            match = 1

            for p, r in zip(pred, real):
                if p != r:
                    match = 0

            if match == 1:
                matching_sequences.append([pred, real])


            num_exact_matches += match

        
        return 100*(num_exact_matches/len(self.predictions)), matching_sequences


            


    

    