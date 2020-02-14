# prep data from philippe's output .txt files

# import stuff
import sys
from find_files import find_files as fifi
import pandas as pd

def merge_txt_files():
    '''
    merges text files in a given dir
    :return:
    '''
    dir = 'datasets/pooled'
    infiles = fifi(dir, '.txt')
    infiles = [item for item in infiles if '1OB1' in item]
    print(infiles)
    for infile in infiles:
        df = pd.read_csv(infile, sep='\t')
        headers1 = [str(item) for item in range(df.shape[1])]
        headers2 = ['slided_seq', 'binding'] + headers1[2:]
        headersdict = dict([(i,j) for i,j in zip(df.columns, headers2)])
        print(headersdict)
        print(df.head())
        df = df.rename(columns=headersdict).sort_values(by='binding')
        print(df.head())
        percent = 0.1
        percent_idx = int(df.shape[0]*percent)
        top10pcdf  = df.iloc[:percent_idx,]
        top10str = ' '.join(top10pcdf.slided_seq)
        allstr = ' '.join(df.slided_seq)
        outfilename = infile.split('/')[-1].split('_')[0]
        top10outfilename = 'outfiles/' + outfilename + '_top10pc_%s.txt' % top10pcdf.shape[0]
        alloutfilename = 'outfiles/' + outfilename + '_all_%s.txt' % df.shape[0]
        print(top10outfilename, alloutfilename)
        outfile1 = open(top10outfilename, 'w')
        outfile1.write(top10str)
        outfile2 = open(alloutfilename, 'w')
        outfile2.write(allstr)



# run stuff
merge_txt_files()


