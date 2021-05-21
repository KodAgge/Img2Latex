import numpy as np
import editdistance as ed
import statistics as stats
from nltk import bleu_score
from sympy.parsing.latex import parse_latex
import warnings
warnings.simplefilter(action='ignore')
import torch
from CorpusHelper import unTokenize 


class Performance:
    
    def __init__(self, predictions, ground_truth):
        self.predictions = predictions
        self.ground_truth = ground_truth

        #CHECK DataTypes
        assert type(self.predictions) == type(self.ground_truth), 'Non-matching input-types'    

        '''
        print('Converting to numpy arrays... \n')
        self.predictions, self.ground_truth = self.convert_to_arrays()
        
        conversionSuccessful = True

        for pred, truth in zip(self.predictions, self.ground_truth):
            if type(pred) != type(truth):
                conversionSuccessfull = False

        #CHECK DataTypes
        print(f'Successfully converted tensors into arrays ==> {conversionSuccessful} \n') 
        assert len(self.predictions) == len(self.ground_truth), 'Numpy arrays have different sizes'
        '''
  
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
            pred, real = self.castListToString(pred), self.castListToString(real)
            dist = ed.distance(pred, real)
            labels.append([pred, real])

            scores_lev.append(dist)
        

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
    
    def bleu(self):
        scores_bleu = []
        labels= []

        for pred, real in zip(self.predictions, self.ground_truth):
            pred, real = self.castListToString(pred), self.castListToString(real)
           
            labels.append([pred, real])
            pred, real = [pred.split()], real.split() #Required format as input

            score =  bleu_score.sentence_bleu(pred, real)
            scores_bleu.append(score)

        return labels, scores_bleu

    def equal_latex(self, expr1, expr2):
        expr1 = parse_latex(expr1)
        expr2 = parse_latex(expr2)
        return expr1.equals(expr2)

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