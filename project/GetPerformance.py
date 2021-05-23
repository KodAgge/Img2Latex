'''
Test file for Performance class
'''

import Performance as P
import torch
import numpy as np
import torch.tensor as tensor
from CorpusHelper import unTokenize 
from latex2img import latex2img


predictions = []
ground_truths = []
skip_line = True

with open('results/TEST_BEAM_TR2000NE7.txt') as file:
    for line in file:
        if skip_line:
            skip_line = False
            continue

        ground_truth, prediction = line.split(';')
        ground_truth, prediction = eval(ground_truth), eval(prediction) #Cast string to list-objects

        predictions.append(prediction)
        ground_truths.append(ground_truth)
        

performance = P.Performance(predictions, ground_truths)


lev_labels, lev_scores, best_lev_prediction, best_lev_score = performance.levenshtein(get_best_prediction=True)
bleu_labels, bleu_scores, best_bleu_prediction, best_bleu_score = performance.bleu(get_best_prediction=True)
jacc_labels, jacc_scores, best_jacc_prediction, best_jacc_score = performance.jaccard(get_best_prediction=True)
lms_labels, lms_scores = performance.LMS()


performance.get_performance(lev_labels=lev_labels, lev_scores=lev_scores, bleu_labels=bleu_labels, bleu_scores=bleu_scores)
performance.get_statistics(lev_scores=lev_scores, bleu_scores=bleu_scores, jacc_scores=jacc_scores, lms_scores=lms_scores)
predictions, ground_truth = performance.listToTensor()


tokenized_ground_truth = []
tokenized_predictions = []

for pred, truth in zip(predictions, ground_truth):
    try:
        tokenized_ground_truth.append(unTokenize(truth))
        tokenized_predictions.append(unTokenize(pred))

    except:
        KeyError

print('\n')
print(20*'=')
print('BEST SEQUENCE')
print(20*'=')
print(f'Levenshtein ==> {best_lev_prediction}  | Distance: {best_lev_score}')
print(f'BLEU        ==> {best_bleu_prediction} | Score: {best_bleu_score}')
print(f'Jaccard     ==> {best_jacc_prediction} | Similarity: {best_jacc_score}')


#print('Distance', best_score)
run_parser = False
equal_latex = []

if run_parser:
    correct = 0
    total = 0
    index = 0

    for pred, truth in zip(tokenized_predictions, tokenized_ground_truth):
        print(index)
        index +=1
        #latex2img(truth)
        
        try:
            #pred = r"\frac{e+1}{4}"
            #truth = r"\frac{e+1}{4}"
            #print(truth)
            equal = performance.equal_latex(pred, truth)
            
            if equal:
                equal_latex.append([pred, truth])
                correct +=1

            total +=1

        except:
            Exception

    for i in equal_latex:
        print(i)

    print(f'Accuracy of LATEX parser: {(correct/total)*100} %')

    #test = unTokenize(ground_truth[3])
    #print(test)



'''
#Test latex parser
expr1 = r"\frac{e+1}{4}"
expr2 = r"\frac{1+e}{4}"
equal = performance.equal_latex(expr1=expr1, expr2=expr2)
print(f'LATEX-expressions equal: {equal}')

'''


        
        
