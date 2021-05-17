from bs4 import BeautifulSoup
import numpy as np


def create_label_csv(entry_path,imageName):
    "Creates a CSV file with 'imageName, img_name, latex-label (not normalized), entry.path' on each row"
    entry_path = entry_path[2:]

    img_path = entry_path.split("\\")[0]
    img_name = entry_path.split("\\")[1]

    
    f = open(img_path +"/"+ img_name,"r")
    file = f.read()
    soup = BeautifulSoup(file, "xml")
    latex_label = soup.find(type="truth").string
    returnString = imageName +"§"+img_name + "§" + latex_label + "§" + img_path + " \n"

    f.close()
    return returnString

def generateCorpus():
    #Extra tokens 'START' 'END'
    modes = ['TRAIN','TEST','VAL']
    specialChars = ['UNK','START','END','PAD']
    vocab = []
    corpus = {}
    mathExpsLenghts = [[],[]]
    for mode in modes: #Run three times to extract symbols from all three datasets
        inputFileName = mode + '_Label_Normalized.txt'
        inputFile = open(inputFileName, 'r')
        
        for line in inputFile: #For each line in data file
            mathExpression = line.split('§')[2]
            mathExpsLenghts[0].append(mathExpression)
            mathExpsLenghts[1].append(len(mathExpression.split(' ')))
            for symbol in mathExpression.split(' '): #For each symbol in each mathematical expression
                if symbol not in vocab:
                    vocab.append(symbol)
        inputFile.close()

    for chars in specialChars:
        vocab.append(chars)
    print('Length of vocab ' + str(len(vocab)))

    #Create corpus dictionary
    iter = 1
    for item in vocab:
        corpus[item] = iter
        iter = iter + 1
    
    i=0
    for items in mathExpsLenghts[1]:
        
        if mathExpsLenghts[1][i] > 100:
            print(mathExpsLenghts[1][i])
            print(mathExpsLenghts[0][i])
        i=i+1
    print(len(mathExpsLenghts[0]))
    return corpus

def tokenize(corpus):
    modes = ['TRAIN','TEST','VAL']
    specialChars = ['UNK','START','END','PAD']
    outputLength = 110 #Length of the tokenized label including "PAD"-padding
    for mode in modes:
        print("Tokenizing the " + mode + " data set")
        inputFileName = mode + '_Label_Normalized.txt'
        inputFile = open(inputFileName, 'r')
        outputFileName = mode + '_Tokenized_Normalized.txt' 
        outputFile = open(outputFileName, 'w')

        for line in inputFile: #For each line in data file
            imageName = line.split('§')[0]
            img_name = line.split('§')[1]
            latex_label = line.split('§')[2]
            img_path = line.split('§')[3]
            num_symbols = len(line.split(" "))
            tokenized_label = []

            #Tokenization
            tokenized_label.append(corpus['START'])
            for symbol in latex_label.split(" "):
                tokenized_label.append(corpus[symbol])
            tokenized_label.append(corpus['END'])

            for k in range(outputLength-num_symbols):
                tokenized_label.append(corpus['PAD'])
            #print(len(tokenized_label))
            outputString = imageName +"§"+img_name + "§" + str(tokenized_label) + "§" + img_path + " \n"
            outputFile.write(outputString)
        inputFile.close()
        outputFile.close()
    return

def printCorpusDict(corpus):
    outputFileName = 'Corpus.txt'
    print("Writing corpus to file " + outputFileName)
    outputFile = open(outputFileName, "w")
    for key in corpus:
        outputString = str(key) + "," + str(corpus[key]) + " \n"
        outputFile.write(outputString)

def main():
    corpus = generateCorpus() # Corpus is a dictionary where they keys are the symbols in math equations e.g. "{" or "\frac"
    tokenize(corpus)
    printCorpusDict(corpus)
# main()


#create_label_csv(filepath,img_name)