'''
Test file for Performance class
'''

import Performance as P

predictions = ['This should match all the time', 'This should not match all the time']
ground_truth = ['This should match all the time','Hey, this should not match all the time']

performance = P.Performance(predictions, ground_truth)
lev_labels, lev_scores = performance.levenshtein()
bleu_labels, bleu_scores = performance.bleu()

performance.get_performance(lev_labels=lev_labels, lev_scores=lev_scores, bleu_labels=bleu_labels, bleu_scores=bleu_scores)
performance.get_statistics(lev_scores=lev_scores, bleu_scores=bleu_scores)

#Test latex parser
expr1 = r"\frac{e+1}{4}"
expr2 = r"\frac{1+e}{4}"
equal = performance.equal_latex(expr1=expr1, expr2=expr2)
print(f'LATEX-expressions equal: {equal}')


        
        
