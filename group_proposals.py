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
    PURPOSE:   extract text from a PDF document
    INPUTS:    d = PDF document file from fitz
               pn = page number to read (int)
    OUTPUTS:   t  = text of page (str)

    """
                
    ### LOAD PAGE
    p = d.loadPage(int(pn))

    ### GET RAW TEXT
    t = p.getText("text")
    
    return t


def get_pages(d, pl):

    """
    PURPOSE:   find start and end pages of proposal text
               [assumes TOC after budget & authors used full page limit]
    INPUTS:    d  = PDF document file from fitz
               pl = page limit of call
    OUTPUTS:   pn = number of pages
               ps = start page number
               pe = end page number

    """

    ### GET NUMBER OF PAGES
    pn = d.pageCount

    ### LOOP THROUGH PDF PAGES
    ps = 0
    for i, val in enumerate(np.arange(pn)):
            
        ### READ IN TEXT FROM THIS PAGE AND NEXT PAGE
        t1 = get_text(d, val)
        t2 = get_text(d, val + 1)
        
        ### FIND PROPOSAL START USING END OF SECTION X
        if ('SECTION X - Budget' in t1) & ('SECTION X - Budget' not in t2):
            
            ### PROPOSAL USUALLY STARTS 2 PAGES AFTER (I.E., INCLDUES TOC)
            ps += val + 2
            
            ### ASSUMES AUTHORS USED FULL PAGE LIMIT
            pe  = ps + (pl - 1)
            
        ### EXIT LOOP IF START PAGE FOUND
        if ps != 0:
            break 

    ### PRINT TO SCREEN (ACCOUNTING FOR ZERO-INDEXING)
    print("\n\tTotal pages = {},  Start page = {},   End page = {}".format(pn, ps + 1, pe + 1))

    return pn, ps, pe


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
    
    return t, rflag


def plot_wc(opath, fn, wc_a, wc_c, nbins=25):

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
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(8, 4))
    axs[0].set_title('All Words', size=10)
    axs[1].set_title('Cleaned Words', size=10)

    ### GET MEDIAN AND ROBUST SIGMA OF DATA
    mu_a, std_a = np.median(wc_a), mad_std(wc_a)
    mu_c, std_c = np.median(wc_c), mad_std(wc_c)

    ### PLOT HISTOGRAM
    na, ba, pa = axs[0].hist(wc_a, bins=nbins, density=True)
    nc, bc, pc = axs[1].hist(wc_c, bins=nbins, density=True)

    ### PLOT FITS
    xf_a = np.linspace(0, np.max(wc_a) + 1000, 100)
    yf_a = norm.pdf(xf_a, mu_a, std_a)
    xf_c = np.linspace(0, np.max(wc_c) + 1000, 100)
    yf_c = norm.pdf(xf_c, mu_c, std_c)
    axs[0].plot(xf_a, yf_a, 'k')
    axs[1].plot(xf_c, yf_c, 'k')
    [axs[0].axvline(x, color='gray', linestyle=":") for x in [mu_a - std_a*2, mu_a + std_a*2]]
    [axs[1].axvline(x, color='gray', linestyle=":") for x in [mu_c - std_c*2, mu_c + std_c*2]]

    ### CLEANUP
    fig.savefig(os.path.join(opath, 'pp_wc_dist.pdf'), bbox_inches='tight', dpi=100, alpha=True, rasterized=True)
    plt.close('all')

    ### RETURN SUSPICIOUS PROPOSALS
    ind_a = np.where( (wc_a > mu_a + 2 * std_a) | (wc_a < mu_a - 2 * std_a) )
    ind_c = np.where( (wc_c > mu_c + 2 * std_c) | (wc_c < mu_c - 2 * std_c) )

    return fn[ind_a], fn[ind_c]


def plot_top(prefix, opath, wc, mc, mcb, expsig):

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
    print("\n\tTop words: " + textwrap.shorten(str(xlabs[0:ind+1]), 60))

    ### CLEANUP
    fig.savefig(os.path.join(opath, prefix + '.pdf'), bbox_inches='tight', dpi=100, alpha=True, rasterized=True)
    plt.close('all')

    return xlabs[0:ind+1]


def exp_fit(x, a, b, c):

    return a * np.exp(-c * x) + b


# ====================== Set Inputs =======================

### SET 
Process_Proposals = True                                # DO NLP PRE-PROCESSING
Make_LDA_Models = False                                 # MAKE ML MODELS

### SET INPUTS THAT ARE PROGRAM-SPECIFIC
PDF_Path  = './XRP20_Proposals'                         # PATH TO PROPOSAL PDFs
NLP_Path  = './XRP20_NLP'                               # PATH TO NPL OUTPUTS
Remv_Prog_File = './nlp_words - xrp.csv'                # NLP WORDS TO REMOVE

### GENERAL INPUTS THAT USUALLY DON'T CHANGE
Page_Lim  = 15                                          # PROPOSAL PAGE LIMIT
ExpMult  = 3                                            # CUTOFF FOR EXPNENTIAL DROP
Lemm_File = './nlp_words - lemmatize.csv'               # ASTRO LEMMATIZE WORDS
Remv_File = './nlp_words - stop.csv'                    # NLP WORDS TO REMOVE


# ====================== Main Code ========================

if Process_Proposals:

    ### GRAB INFO IF IT DOESN'T ALREADY EXIST
    PDF_Files = np.sort(glob.glob(os.path.join(PDF_Path, '*.pdf')))
    Files, WC_All, WC_Clean, Text_All, Vocab_All, MC_All, MC_Top, Files_Skip  = [], [], [], [], [], [], [], []
    for p, pval in enumerate(PDF_Files):

        ### OPEN PDF FILE
        Prop_Name = (pval.split('/')[-1]).split('.pdf')[0]
        Doc = fitz.open(pval)
        print(colored("\n\n\n\t" + Prop_Name, 'green', attrs=['bold']))

        ### GET PAGES OF PROPOSAL
        try:
            Page_Num, Page_Start, Page_End = get_pages(Doc, Page_Lim)
        except RuntimeError:
            print("\tCould not read PDF")
            print("\n\t!!!!!!!!!DID NOT SAVE!!!!!!!!!!!!!!!!")
            Files_Skip.append(pval)
            continue

        ### GET TEXT OF FIRST PAGE TO CHECK
        print("\n\tSample of first page:\t" + textwrap.shorten((get_text(Doc, Page_Start)[100:130]), 40))
        print("\tSample of mid page:\t"     + textwrap.shorten((get_text(Doc, Page_Start + 8)[100:130]), 40))
        print("\tSample of last page:\t"    + textwrap.shorten((get_text(Doc, Page_End)[100:130]), 40))
        
        ### GRAB TEXT OF ENTIRE PROPOSAL
        Text_Proposal = ''
        for i, val in enumerate(np.arange(Page_Start, Page_End)):    
            Text_Proposal = Text_Proposal + ' ' + get_text(Doc, val)

        ### SPLIT INTO WORDS
        Text_Split = split_text(Text_Proposal)

        ### CLEAN UP TEXT
        Text_Clean, RFlag = clean_text(Text_Split, Lemm_File, Remv_File, Remv_Prog_File)
        # Text_Clean, RFlag = clean_text(Text_Split, Lemm_File, Remv_File)

        ### IDENTIFY MOST USED WORDS
        FD = nltk.FreqDist(Text_Clean)
        MC = dict(FD.most_common(50))
        MC_All.append(list(MC.keys()))

        ### IDENTIFY MOST USED BIGRAMS (MUST OCCUR AT LEAST 10 TIMES)
        BGM = nltk.collocations.BigramAssocMeasures()
        BGF = BigramCollocationFinder.from_words(Text_Clean)
        BGF.apply_freq_filter(10)
        MCB = BGF.ngram_fd
        tmp = BGF.score_ngrams(BGM.raw_freq)

        print("\n\tTotal Word Count:\t{}".format(len(Text_Split)))
        print("\tCleaned Word Count:\t{}".format(len(Text_Clean)))

        ### SAVE PROPOSAL PRE-PROCESSING FOR THOSE THAT MADE IT THROUGH
        PreFix = 'pp_rf' + str(RFlag).zfill(2) + '_' + pval.split('/')[-1][0:-4]
        top = plot_top(PreFix, NLP_Path, len(Text_Split), MC, MCB, ExpMult)
        if (len(Text_Clean) > 1000) & (not any('�' in word for word in top)):
            Text_All.append(Text_Clean)
            WC_All.append(len(Text_Split))
            WC_Clean.append(len(Text_Clean))
            Files.append(pval)
            MC_Top.append(top.tolist())
            Vocab_All = Vocab_All + Text_Clean
            # np.save(os.path.join(NLP_Path, PreFix + '.npy'), top)
        else:
            Files_Skip.append(pval)
            print("\n\t!!!!!!!!!DID NOT SAVE!!!!!!!!!!!!!!!!")

    Bad_All, Bad_Clean = plot_wc(NLP_Path, np.array(Files), np.array(WC_All), np.array(WC_Clean))
    Vocab = np.unique(np.array(Vocab_All))


### TO DO: PARAMETERIZE ALL THESE AND SAVE DICT/CORP + MODELS W/RANDOM SEED SET FOR EASY LOADING
### VIZUALIZE CATEGORIZATIONS

if Make_LDA_Models:
                            
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
    Dict = corpora.Dictionary(Text_All)
    Corp = [Dict.doc2bow(text) for text in Text_All]
    logging.basicConfig(filename=os.path.join(NLP_Path, 'LDA_gensim.log'), format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
    LDAM = gensim.models.ldamodel.LdaModel(Corp, num_topics = NumTopics, id2word=Dict, passes=50, iterations=150,
                                        #    eta = 'auto', alpha='auto', eval_every=1)
                                           eta = [0.001]*len(Dict.keys()), alpha=[0.001]*NumTopics, eval_every=10)
    topics = LDAM.print_topics(num_words=6)
    for topic in topics:
        print(topic)

    ### PLOT CONVERGENCE
    p = re.compile(r"(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    matches = [p.findall(l) for l in open(os.path.join(NLP_Path, 'LDA_gensim.log'))]
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
    plt.savefig(os.path.join(NLP_Path, "LDA_convergence_liklihood.pdf"))
    plt.close()

    ### MAKE NICE VISUALIZATION
    vis = pyLDAvis.gensim.prepare(topic_model=LDAM, corpus=Corp, dictionary=Dict)
    pyLDAvis.show(vis)
    ### A good topic model will have fairly big, non-overlapping bubbles scattered throughout the chart instead of being clustered in one quadrant.
    ### A model with too many topics, will typically have many overlaps, small sized bubbles clustered in one region of the chart.
    ### area of circle represents the importance of each topic over the entire corpus, the distance between the center of circles indicate the similarity between topics

    ### SAVE MODEL IF GOOD
    PreFixLDA = "LDA_NT-" + str(NumTopics) + '_'
    pickle.dump(Corp, open(os.path.join(NLP_Path, PreFixLDA + 'corpus.pkl'), 'wb'))
    Dict.save(os.path.join(NLP_Path, PreFixLDA + 'dict.gensim'))
    LDAM.save(os.path.join(NLP_Path, PreFixLDA + 'model5.gensim'))

    ### FIND A CATEGORY FOR A DOC (95 Cleeves, 14 Hasegawa; 37 Ertel; 9 Mann)
    idx = 9
    dd = Dict.doc2bow(Text_All[idx])
    # cc = [Dict.doc2bow(text) for text in Text_All]
    print("\n")
    print(Files[idx])
    print(MC_Top[idx])
    print(LDAM.get_document_topics(dd))

