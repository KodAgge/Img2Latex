# Script to manually check differences between textfiles

file1 = "./data/sample/formulas.normalized_and_tokenized.lst"  
file2 = "./data/sample/formulas.booth.lst"

with open(file1, 'r+') as f1, open(file2, 'r+') as f2:
    counter = 1
    for line1, line2 in zip(f1, f2):
        if line1!=line2:
            print(line1)
            print(line2)
            print('Non-matching lines at line: ', counter)
            input('---')
        counter +=1
