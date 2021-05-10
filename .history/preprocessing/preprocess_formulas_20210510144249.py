
##### EK)

### Which of the two modes should be ran first: normalization or tokenization?
# It seems that normalization first and then tokenization gives the best results contrary to what's implied in the first "parser.add_argument(...)". For instance we get...
#  \left ( D ^ { * } D ^ { * } + m ^ { 2 } \right ) { \cal H } = 0
# ...instead of...
#  \left( D ^ { * } D ^ { * } + m ^ { 2 } \right) { \cal H } = 0
# ...which seems preferable.

### How to run this script:
# a) normalization
# python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file <input_file_path> --output-file <output_file_path>
# b) tokenization
# python scripts/preprocessing/preprocess_formulas.py --mode tokenize --input-file <input_file_path> --output-file <output_file_path>



#####


#!/usr/bin/env python
# tokenize latex formulas
import sys, os, argparse, logging, subprocess, shutil

def is_ascii(str):
    try:
        str.encode('ascii')
        return True
    except UnicodeError:
        return False

def process_args(args):
    parser = argparse.ArgumentParser(description='Preprocess (tokenize or normalize) latex formulas')

    parser.add_argument('--mode', dest='mode',
                        choices=['tokenize', 'normalize'], required=True,
                        help=('Tokenize (split to tokens seperated by space) or normalize (further translate to an equivalent standard form).'
                        ))
    parser.add_argument('--input-file', dest='input_file',
                        type=str, required=True,
                        help=('Input file containing latex formulas. One formula per line.'
                        ))
    parser.add_argument('--output-file', dest='output_file',
                        type=str, required=True,
                        help=('Output file.'
                        ))
    parser.add_argument('--num-threads', dest='num_threads',
                        type=int, default=4,
                        help=('Number of threads, default=4.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
    parameters = parser.parse_args(args)
    #input(parameters) # EK) Here we can see the object with all command-line arguments
    return parameters

def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s'%__file__)

    input_file = parameters.input_file
    output_file = parameters.output_file

    print(input_file)
    print(output_file)


    input('Block 1')
    # EK) I haven't worked in Pearl before but this block seems to just create a new file (output_file) and copy the contents of input_file to it

    assert input_file
    assert os.path.exists(input_file), input_file
    cmd = 'perl -pe "s|hskip(.*?)(cm\\|in\\|pt\\|mm\\|em)|hspace{\\1\\2}|g" %s > %s'%(input_file, output_file)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        logging.error('FAILED: %s'%cmd)


    input('Block 2')
    # EK) This block just copies the output_file to a temp_file. The string processing doesn't appear to do anything (for normalization)

    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as fout:  
        with open(output_file, 'r') as fin: # EK) added "r" here. Was no "r" before and didn't work
            for line in fin:  
                # EK) If statement below doesn't appear to ever be true so the .replace(...).strip()... doesn't really do anything 
                #if line != line.replace('\r', ' ').strip() + '\n':
                #    print(line)
                #    print(line.replace('\r', ' ').strip() + '\n')
                #    input('---')

                fout.write(line.replace('\r', ' ').strip() + '\n')  # delete \r   


    input('Block 3')
    # EK) This block seems to be doing all the preprocessing. On a high level I think this reads the temp_file, does pre
    # processing (either normalization or tokenization)and then stores the result in output_file???

    cmd = 'cat %s | node scripts/preprocessing/preprocess_latex.js %s > %s '%(temp_file, parameters.mode, output_file)
    ret = subprocess.call(cmd, shell=True)
    os.remove(temp_file)
    print(ret)


    input('Block 4')
    # EK) Since all charachters are ASCII, this code doesn't really do anything except removing some whitespaces (for the normalization case).

    if ret != 0:
        logging.error('FAILED: %s'%cmd)
    temp_file = output_file + '.tmp'
    shutil.move(output_file, temp_file)  # EK) this just creates a new temp_file and moves the content of output_file to it.
    with open(temp_file) as fin: # EK) added r here
        with open(output_file, 'w+') as fout: # EK) added "+" here
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    if is_ascii(token):
                        tokens_out.append(token)
                    #else:
                        #print(line)
                        #input('Attention, the string above is not ASCII')
                #if line != ' '.join(tokens_out)+'\n':
                    #print(line)
                    #print(' '.join(tokens_out)+'\n')
                    #input('Attention, the strings above are different')

                fout.write(' '.join(tokens_out)+'\n')
    os.remove(temp_file)


if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
