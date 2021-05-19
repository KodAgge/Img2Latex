import torch


##  README
# FUNCTIONS:
# CorpusHelper.corpus() Returns a python dictionary: [key,value] = [token,symbol]
# CorpusHelper.unTokenize(tokenizedTorchTensor,True,110) # Returns a string of normalized latex label. 

# How to access dictionary and get a un-tokenized label vector:
"""
import CorpusHelper

corpusDict = CorpusHelper.corpus()                                                              
un-tokenizedVector = CorpusHelper.unTokenize(tokenizedTorchTensor,specialToken = True, sequenceLength = 110) 



set specialToken = False in order to get a latex label without special tokens [START, END, UNK, PAD]
"""

def corpus():

    filePath = "./project"
    filename = "Corpus.txt"
    corpusFile = open(filePath +"/"+ filename, "r")
    corpusDict = {}

    for line in corpusFile: #For each line in data file
        token = line.split(',')[1].split(" ")[0]
        latexSymbol = line.split(",")[0]
        corpusDict[token] = latexSymbol

    corpusFile.close()
    return corpusDict # [key,value] = [token,symbol]


def unTokenize(tokenized_tensor, specialToken = True, sequenceLength = 110):
    corpusDict = corpus()
    returnString = ""
    specialChars = ['UNK','START','END','PAD']
    if specialToken == True:
        
        for i in range(sequenceLength):
            key = str(tokenized_tensor[i]).split("(")[1].split(")")[0]
            labelString = corpusDict[key]
            returnString = returnString +" "+ labelString
        
    if specialToken == False:
        for i in range(sequenceLength):
            key = str(tokenized_tensor[i]).split("(")[1].split(")")[0]
            labelString = corpusDict[key]
            if labelString in specialChars:
                returnString = returnString

            else:   
                returnString = returnString +" "+ labelString
        
    
    return returnString

def main():
    corpusDict = corpus()
    testTensor = torch.tensor([142,2,3,4,144])
    untokenizedVector = unTokenize(testTensor,specialToken = True)
    
main()