"""

Things to do:

1) try TF-IDF to identify most "important" words (rather than most common words)
   https://towardsdatascience.com/applying-machine-learning-to-classify-an-unsupervised-text-document-e7bb6265f52
   https://towardsdatascience.com/extracting-taxonomic-data-from-a-journal-articles-using-natural-language-processing-ab794d048da9
    https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05
    
2) Add in bi-grams to list of most-used words

3) outlier detection

4) Remove PI name from list of most-used words

5) Use WordNet to find synonyms (from nltk.corpus import wordnet)


"""

# ============== Import Packages ================

### STANDARD MODULES
import os, sys, pdb, glob
import numpy as np
import pandas as pd
import pickle
import logging
import re

### OCD TERMINAL OUTPUT THINGS
import textwrap
from termcolor import colored

### FITTING & STATS
from scipy.stats import norm
from astropy.stats import mad_std
from scipy.optimize import curve_fit

### FOR NLP
import fitz
fitz.TOOLS.mupdf_display_errors(False)
import nltk
import gensim
from gensim import corpora
from nltk.collocations import *
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter 

### FOR PLOTTING
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim


# ============== Define Functions ===============

def get_text(d, pn):

    """
    PURPOSE:   extract text from a given page of a PDF document
    INPUTS:    d  = fitz Document object
               pn = page number to read (int)
    OUTPUTS:   t  = text of page (str)

    """
                
    ### LOAD PAGE
    p = d.loadPage(int(pn))

    ### GET RAW TEXT
    t = p.getText("text")

    ### FIX ENCODING
    t = t.encode('utf-8', 'replace').decode()
     
    return t


def get_pages(d, pl=15):

    """
    PURPOSE:   find start and end pages of proposal within NSPIRES-formatted PDF
               [assumes proposal starts after budget, and references at end of proposal]
    INPUTS:    d  = fitz Document object
               pl = page limit of proposal (int; default = 15)
    OUTPUTS:   pn = number of pages of proposal (int)
               ps = start page number (int)
               pe = end page number (int)

    """

    ### GET NUMBER OF PAGES
    pn = d.pageCount

    check_words = ["contents", "c o n t e n t s", "budget", "cost", "costs",
                   "submitted to", "purposely left blank", "restrictive notice"]

    ### LOOP THROUGH PDF PAGES
    ps = 0
    for i, val in enumerate(np.arange(pn)):
            
        ### READ IN TEXT FROM THIS PAGE AND NEXT PAGE
        t1 = get_text(d, val)
        t2 = get_text(d, val + 1)
        
        ### FIND PROPOSAL START USING END OF SECTION X
        if ('SECTION X - Budget' in t1) & ('SECTION X - Budget' not in t2):

            ### SET START PAGE
            ps = val + 1

            ### ATTEMPT TO CORRECT FOR (ASSUMED-TO-BE SHORT) COVER PAGES
            if len(t2) < 500:
                ps += 1
                t2 = get_text(d, val + 2)

            ### ACCOUNT FOR TOC OR SUMMARIES
            if any([x in t2.lower() for x in check_words]):
                ps += 1

            ### ASSUMES AUTHORS USED FULL PAGE LIMIT
            pe  = ps + (pl - 1) 
                        
        ### EXIT LOOP IF START PAGE FOUND
        if ps != 0:
            break 

    ### ATTEMPT TO CORRECT FOR TOC > 1 PAGE OR SUMMARIES
    if any([x in get_text(d, ps).lower() for x in check_words]):
        ps += 1
        pe += 1

    ### CHECK THAT PAGE AFTER LAST IS REFERENCES
    Ref_Words = ['references', 'bibliography', "r e f e r e n c e s", "b i b l i o g r a p h y"]
    if not any([x in get_text(d, pe + 1).lower() for x in Ref_Words]):

        ### IF NOT, TRY NEXT PAGE (OR TWO) AND UPDATED LAST PAGE NUMBER
        if any([x in get_text(d, pe + 2).lower() for x in Ref_Words]):
            pe += 1
        elif any([x in get_text(d, pe + 3).lower() for x in Ref_Words]):
            pe += 2

        ### CHECK THEY DIDN'T GO UNDER THE PAGE LIMIT
        if any([x in get_text(d, pe).lower() for x in Ref_Words]):
            pe -= 1
        elif any([x in get_text(d, pe - 1).lower() for x in Ref_Words]):
            pe -= 2
        elif any([x in get_text(d, pe - 2).lower() for x in Ref_Words]):
            pe -= 3
        elif any([x in get_text(d, pe - 3).lower() for x in Ref_Words]):
            pe -= 4

    ### PRINT TO SCREEN (ACCOUNTING FOR ZERO-INDEXING)
    print("\n\tTotal pages = {},  Start page = {},   End page = {}".format(pn, ps + 1, pe + 1))

    return pn, ps, pe


def get_proposal_info(doc):

    """
    PURPOSE:   grab PI name and proposal number from cover page
    INPUTS:    doc  = fitz Document object
    OUTPUTS:   pi_first = PI first name (str)
               pi_last = PI last name (str)
               pn = proposal number assigned by NSPIRES (str)

    """

    ### GET COVER PAGE
    cp = get_text(doc, 0)

    ### GET PI NAME
    pi_name = ((cp[cp.index('Principal Investigator'):cp.index('E-mail Address')]).split('\n')[1]).split(' ')
    pi_first, pi_last = pi_name[0].title(), pi_name[-1].title()

    ### GET PROPOSAL NUMBER
    pn = ((cp[cp.index('Proposal Number'):cp.index('NASA PROCEDURE FOR')]).split('\n')[1]).split(' ')[0]

    return pi_first, pi_last, pn


def split_text(t):

    ### REMOVE PUNCTUATION
    replace_with_space = ['/', '+', '#', '!', '\n', '.', ',', ';', ')', '(', '[', ']']
    for i, val in enumerate(replace_with_space):
        t = t.replace(val, ' ')

    # ### SPLIT HYPHENATED WORDS
    # replace_with_space = ['-']
    # for i, val in enumerate(replace_with_space):
    #     t = t.replace(val, ' ')

    ### TOKENIZE SENTENCES (BREAK INTO WORDS)
    # t = t.split(' ')
    t = nltk.word_tokenize(t)

    ### REMOVE ANY WORDS CONTAINING NUMBERS
    t = [x for x in t if not any(c.isdigit() for c in x)]
    
    ### REMOVE ANY WORDS THAT HAVE <= 2 LETTERS (ALSO REMOVES SPACES)
    t = [x for x in t if len(x) > 2]

    ### CHANGE ALL WORDS TO LOWER CASE
    t = [x.lower() for x in t]

    return t


def clean_text(t, lmf, rmf, rmf2=[]):
    
    
    """
    PURPOSE:   clean up extracted text
    INPUTS:    
    OUTPUTS:   t  = cleaned-up text
    
    """
    
    ### LEMMATIZE WITH NLTK
    lem = WordNetLemmatizer()
    t_lem=[]
    for w in t:
        t_lem.append(lem.lemmatize(w, 'v'))
    t = np.array(t_lem)

    ### LEMMATIZE SPECIAL ASTRO WORDS
    dfl = pd.read_csv(lmf)     
    for i, val in enumerate(np.arange(len(dfl))):
        w = dfl.iloc[i][~pd.isna(dfl.iloc[i])].values
        for n, nval in enumerate(w):
            if n == 0:
                continue
            else:
                idx = np.where(t == nval)
                t[idx] = w[0]

    ### REMOVE STOP & COMMON WORDS
    t = t.tolist()
    dfr = pd.read_csv(rmf) 
    rm = dfr['word'].values 
    for i, val in enumerate(rm):
        t = list(filter((val).__ne__, t))
    t = list(filter(('').__ne__, t))
    
    ### REMOVE WORDS FOR THIS PROGRAM
    if len(rmf2) > 0:
        rflag = 1
        dfr2 = pd.read_csv(rmf2) 
        rm2 = dfr2['word'].values 
        for i, val in enumerate(rm2):
            t = list(filter((val).__ne__, t))
    else:
        rflag = 0

    ### REMOVE WORDS THAT CONTAIN IRREGULAR CHARACTERS
    t = [x for x in t if len(re.findall(r'[^a-zA-Z0-9\._-]', x)) == 0 ]

    return t, rflag


def plot_wc(opath, fn, wc, nbins=25):

    ### SETUP PLOT
    mpl.rc('xtick', labelsize=10)
    mpl.rc('ytick', labelsize=10)
    mpl.rc('xtick.major', size=5, pad=7, width=2)
    mpl.rc('ytick.major', size=5, pad=7, width=2)
    mpl.rc('xtick.minor', width=2)
    mpl.rc('ytick.minor', width=2)
    mpl.rc('axes', linewidth=2)
    mpl.rc('lines', markersize=5)

    ### SETUP AXES
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title('Distribution of Word Counts', size=10)
    ax.set_xlabel('Number of Words in Proposal')

    ### GET MEDIAN AND ROBUST SIGMA OF DATA
    mu, std = np.median(wc), mad_std(wc)

    ### PLOT HISTOGRAM WITH GAUSSIAN FIT
    na, ba, pa = ax.hist(wc, bins=nbins, density=True)
    xf = np.linspace(0, np.max(wc) + 1000, 100)
    yf = norm.pdf(xf, mu, std)
    ax.plot(xf, yf, 'k')
    [ax.axvline(x, color='gray', linestyle=":") for x in [mu - std*2, mu + std*2]]

    ### CLEANUP
    fig.savefig(os.path.join(opath, 'pp_wc_dist.pdf'), bbox_inches='tight', dpi=100, alpha=True, rasterized=True)
    plt.close('all')

    ### RETURN PROPOSALS WITH CURIOUS WORD COUNTS
    # ind = np.where( (wc > mu + 2 * std) | (wc < mu - 2 * std) )


def plot_top_words(prefix, opath, wc, mc, expsig=3, mcb=[]):

    """
    PURPOSE:   plot most used words
    INPUTS:    ppath = path of file with name of proposal
               opath = output path
               wc    = total word count
               mc    = dictionary with most used words and their counts
               mcb   = dictionary with most used bigrams
    OUTPUTS:   top words according to exponential cut-off
    
    """

    ### GET DATA TO PLOT
    yvals = np.array(list(mc.values()))
    xvals = np.arange(0, len(yvals))
    xlabs = np.array(list(mc.keys()))
    
    ### SETUP PLOT
    mpl.rc('xtick', labelsize=10)
    mpl.rc('ytick', labelsize=10)
    mpl.rc('xtick.major', size=5, pad=7, width=2)
    mpl.rc('ytick.major', size=5, pad=7, width=2)
    mpl.rc('xtick.minor', width=2)
    mpl.rc('ytick.minor', width=2)
    mpl.rc('axes', linewidth=2)
    mpl.rc('lines', markersize=5)

    ### SETUP AXES
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Number of Uses', fontsize=12, labelpad=10)
    ax.xaxis.set_ticks(np.arange(len(xvals)))
    ax.set_xticklabels(xlabs, rotation='vertical', size=8)

    ### FIT EXPONENIAL
    params_guess = (np.max(yvals), np.min(yvals), 0.5)
    fp, fc = curve_fit(exp_fit, xvals, yvals, p0=params_guess)
    x_fit = np.linspace(0, len(yvals), 100)
    y_fit = exp_fit(x_fit, *fp)

    ### PLOT DATA AND FIT
    ax.scatter(xvals, yvals)
    ax.plot(x_fit, y_fit)

    ### MAKE EXPONENTIAL CUTOFFS
    half_life = np.log(2) / fp[2]
    e_time = half_life / np.log(2)
    ind = np.min([int(round(expsig * half_life)), len(xvals)-1])
    ax.axvline(xvals[ind], linestyle=":", color='gray')
    print("\n\tKeywords: " + textwrap.shorten(str(xlabs[0:ind+1]), 60))

    ### CLEANUP
    fig.savefig(os.path.join(opath, prefix + '.pdf'), bbox_inches='tight', dpi=100, alpha=True, rasterized=True)
    plt.close('all')

    ### SAVE TOP WORDS
    top = xlabs[0:ind+1] 

    return top.tolist()


def exp_fit(x, a, b, c):

    return a * np.exp(-c * x) + b


# ====================== Set Inputs =======================

### SET THINGS TO DO
Find_Keywords = False                                                     # GRAB KEYWORDS 
Make_LDA_Models = False                                                   # MAKE LDA MODELS
Apply_LDA_Models = True                                                   # APPLY LDA MODELS

### SET I/O PATHS
PDF_Path  = '../panels/XRP/XRP_Proposals_2014_2020/XRP_Proposals_2020'    # PATH TO PROPOSAL PDFs
Out_Path  = '../panels/XRP20_NLP'                                         # PATH TO NPL OUTPUTS

### SET FILES FOR CLEANING TEXT
Lemm_File = '../panels/nlp_words - lemmatize.csv'                         # ASTRO LEMMATIZE WORDS
Remv_File = '../panels/nlp_words - stop.csv'                              # NLP WORDS TO REMOVE
Remv_Prog_File = '../panels/nlp_words - xrp.csv'                          # NLP WORDS TO REMOVE


# ====================== Main Code ========================

### DO PRE-PROCESSING
if Find_Keywords:

    ### GET LIST OF PDF FILES
    PDF_Files = np.sort(glob.glob(os.path.join(PDF_Path, '*.pdf')))

    ### ARRAYS TO FILL
    File_Names_All, PI_Names_All, Prop_Nb_All, Files_Skipped_All, Text_Proposal_All, Text_Clean_All = [], [], [], [], [], []
    MC_Words_All, Key_Words_All, Vocab_All, ML_Count_All = [], [], [], []

    ### LOOP THROUGH ALL PROPOSALS
    for p, pval in enumerate(PDF_Files):

        ### OPEN PDF DOCUMENT
        Doc = fitz.open(pval)
        
        ### GET PI NAME AND PROPOSAL NUMBER
        PI_First, PI_Last, Prop_Nb = get_proposal_info(Doc)
        print(colored(f'\n\n\n\t{Prop_Nb}\t{PI_Last}', 'green', attrs=['bold']))
        
        ### GET PAGES OF S/T/M PROPOSAL
        try:
            Page_Num, Page_Start, Page_End = get_pages(Doc)         
        except RuntimeError:
            print("\n\tCould not read PDF, did not save")
            Files_Skipped_All.append(pval)
            continue

        ### GET TEXT OF FIRST PAGE TO CHECK
        print("\n\tSample of first page:\t" + textwrap.shorten((get_text(Doc, Page_Start)[100:130]), 40))
        print("\tSample of mid page:\t"     + textwrap.shorten((get_text(Doc, Page_Start + 8)[100:130]), 40))
        print("\tSample of last page:\t"    + textwrap.shorten((get_text(Doc, Page_End)[100:130]), 40))
        
        ### GRAB TEXT OF ENTIRE PROPOSAL
        Text_Proposal = ''
        for i, val in enumerate(np.arange(Page_Start, Page_End + 1)):    
            Text_Proposal = Text_Proposal + ' ' + get_text(Doc, val)

        ### SPLIT INTO WORDS
        Text_Split = split_text(Text_Proposal)
        if len(Text_Split) == 0:
            print("\n\tCould not read PDF, did not save")
            Files_Skipped_All.append(pval)
            continue

        ### CLEAN UP TEXT
        Text_Clean, RFlag = clean_text(Text_Split, Lemm_File, Remv_File, Remv_Prog_File)
        print("\n\tTotal Word Count:\t{}".format(len(Text_Split)))
        print("\tCleaned Word Count:\t{}".format(len(Text_Clean)))

        if (len(Text_Clean) > 1000):

            ### IDENTIFY MOST USED WORDS
            FD = nltk.FreqDist(Text_Clean)
            MC_Words = dict(FD.most_common(50))

            # ### IDENTIFY MOST USED BIGRAMS (MUST OCCUR AT LEAST 10 TIMES)
            # BGM = nltk.collocations.BigramAssocMeasures()
            # BGF = BigramCollocationFinder.from_words(Text_Clean)
            # BGF.apply_freq_filter(10)
            # MCB = BGF.ngram_fd
            # tmp = BGF.score_ngrams(BGM.raw_freq)

            ### IDENTIFY KEYWORDS
            PreFix = f'kw_rf{str(RFlag).zfill(2)}_{Prop_Nb}_{PI_Last}'
            Key_Words = plot_top_words(PreFix, Out_Path, len(Text_Split), MC_Words)

            ### SAVE THINGS
            File_Names_All.append(pval)
            PI_Names_All.append(PI_Last)
            Prop_Nb_All.append(Prop_Nb)
            Text_Clean_All.append(Text_Clean)
            Text_Proposal_All.append(Text_Split)
            MC_Words_All.append(list(MC_Words.keys()))
            Key_Words_All.append(Key_Words)
            Vocab_All = Vocab_All + Text_Clean

            ### CHECK IF ML MENTIONED
            ML_Words = ["machine learning", "deep learning", "artificial intelligence"]
            ML_Count_All.append(np.sum(np.array([(Text_Proposal.lower()).count(x) for x in ML_Words])))

        else:

            Files_Skipped_All.append(pval)
            print("\n\tCould not read PDF, did not save")

    ### PLOT DISTRIBUTION OF WORD COUNTS
    plot_wc(Out_Path, File_Names_All, [len(x) for x in Text_Proposal_All])

    ### SAVE INFORMATION FOR NLP MODELS
    # Vocab = np.unique(np.array(Vocab_All))
    pickle.dump(Text_Clean_All, open(os.path.join(Out_Path, 'text_clean.pkl'), 'wb'))
    d = {'Prop_Nb': Prop_Nb_All, 'PI_Name': PI_Names_All, 'Keywords': Key_Words_All}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(Out_Path, 'keywords.csv'), index=False)



### TO DO: PARAMETERIZE ALL THESE AND SAVE DICT/CORP + MODELS W/RANDOM SEED SET FOR EASY LOADING
### VIZUALIZE CATEGORIZATIONS

if Make_LDA_Models:

    ### READ IN DATA FROM PRE-PROCESSING
    Text_Clean_All = pickle.load(open(os.path.join(Out_Path, 'text_clean.pkl'), 'rb')) 
    df = pd.read_csv(os.path.join(Out_Path, 'keywords.csv'))

    ### RUN LDA MODEL
    ### set passes to high value; useful to see corpus many time for small datasets
    ### chunksize is how many are seen at once; can effect results but not sure how
    ### alpha close to zero = fewer topics per document; auto = will be tuned automatically.
    ### alpha and eta can be thought of as smoothing parameters when we compute how much each document "likes" a topic (alpha) or how much each topic "likes" a word (eta)
    ### higher alpha makes the document preferences "smoother" over topics, and a higher eta makes the topic preferences "smoother" over words.
    ### when alpha is low, most of the weight in the topic distribution for this article goes to a single topic, but when it is high, the weight is much more evenly distributed across the topics.
    ### low eta results in higher weight placed on the top words and lower weight placed on the bottom words for each topic
    ### DEFAULTS: num_topics=100, id2word=None, distributed=False, chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001)
    NumTopics = 3
    Dict = corpora.Dictionary(Text_Clean_All)
    Corp = [Dict.doc2bow(text) for text in Text_Clean_All]
    logging.basicConfig(filename=os.path.join(Out_Path, 'LDA_gensim.log'), format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
    LDAM = gensim.models.ldamodel.LdaModel(Corp, num_topics = NumTopics, id2word=Dict, passes=50, iterations=150,
                                        #    eta = 'auto', alpha='auto', eval_every=1)
                                           eta = [0.001]*len(Dict.keys()), alpha=[0.001]*NumTopics, eval_every=10)
    topics = LDAM.print_topics(num_words=6)
    for topic in topics:
        print(topic)

    ### PLOT CONVERGENCE
    p = re.compile(r"(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    matches = [p.findall(l) for l in open(os.path.join(Out_Path, 'LDA_gensim.log'))]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    perplexity = [float(t[1]) for t in tuples]
    liklihood = [float(t[0]) for t in tuples]
    iterations = list(range(0,len(tuples)*10,10))
    plt.plot(iterations,liklihood,c="black")
    plt.ylabel("log liklihood")
    plt.xlabel("iteration")
    plt.title("Topic Model Convergence")
    plt.grid()
    plt.savefig(os.path.join(Out_Path, "LDA_convergence_liklihood.pdf"))
    plt.close()

    ### MAKE NICE VISUALIZATION
    vis = pyLDAvis.gensim.prepare(topic_model=LDAM, corpus=Corp, dictionary=Dict)
    pyLDAvis.show(vis)
    ### A good topic model will have fairly big, non-overlapping bubbles scattered throughout the chart instead of being clustered in one quadrant.
    ### A model with too many topics, will typically have many overlaps, small sized bubbles clustered in one region of the chart.
    ### area of circle represents the importance of each topic over the entire corpus, the distance between the center of circles indicate the similarity between topics

    ### SAVE MODEL IF GOOD
    PreFixLDA = "LDA_NT-" + str(NumTopics) + '_'
    pickle.dump(Corp, open(os.path.join(Out_Path, PreFixLDA + 'corpus.pkl'), 'wb'))
    Dict.save(os.path.join(Out_Path, PreFixLDA + 'dict.gensim'))
    LDAM.save(os.path.join(Out_Path, PreFixLDA + 'model5.gensim'))


if Apply_LDA_Models:

    LDAM = gensim.models.ldamodel.LdaModel.load(os.path.join(Out_Path, 'LDA_NT-3_model5.gensim'))

    ### FIND A CATEGORY FOR A DOC (0 Boss, 96 Cleeves, 14 Hasegawa; 37 Ertel; 9 Mann; 30 Morley; 40 Bergin)
    idx = 14
    dd = Dict.doc2bow(Text_Clean_All[idx])
    # cc = [Dict.doc2bow(text) for text in Text_All]
    print("\n")
    print(File_Names_All[idx])
    print(Key_Words_All[idx])
    print(LDAM.get_document_topics(dd))

    for i, val in enumerate(df['Prob_Nb']):
        dd = Dict.doc2bow(Text_Clean_All[i])
        print(df['Prop_Nb'].iloc[i], LDAM.get_document_topics(dd))
        pdb.set_trace()


