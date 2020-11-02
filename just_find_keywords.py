import os, sys, pdb, glob
import numpy as np
import pandas as pd

import textwrap
from termcolor import colored

import fitz
fitz.TOOLS.mupdf_display_errors(False)

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

    check_words = ["contents", "budget", "cost", "costs",
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
    Ref_Words = ['references', 'bibliography']
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


# ====================== Set Inputs =======================

### KEYWORDS TO SEARCH FOR
Keywords = ["machine learning", "deep learning", "artificial intelligence", "neural network", "active learning", "autoencoders"]
Page_Limit = 7

### GET LIST OF PDF FILES
PDF_Path = '/Volumes/MAnsdell/NASA_Proposals/NESSF/NESSF_Proposals_2019/NESSF19'
PDF_Files = np.sort(glob.glob(os.path.join(PDF_Path, '*.pdf')))

### LOOP THROUGH ALL PROPOSALS
File_Names_All, Prop_Nb_All, Files_Skipped_All, Keywords_Count_All = [], [], [], []
for p, pval in enumerate(PDF_Files):

    ### OPEN PDF DOCUMENT
    Doc = fitz.open(pval)
    
    ### GET PI NAME AND PROPOSAL NUMBER
    PI_First, PI_Last, Prop_Nb = get_proposal_info(Doc)
    print(colored(f'\n\n\n\t{Prop_Nb}\t{PI_Last}', 'green', attrs=['bold']))
    
    ### GET PAGES OF S/T/M PROPOSAL
    try:
        Page_Num, Page_Start, Page_End = get_pages(Doc, pl=Page_Limit)         
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
        Text_Proposal = Text_Proposal.replace('\n', '')

    ### COUNT NUMBER OF KEYWORD MENTIONS
    Prop_Nb_All.append(Prop_Nb)
    if Text_Proposal.count(" ") < 6500:
        Keywords_Count_All.append([(Text_Proposal.lower()).count(x) for x in Keywords])

    ### CATCH PROPOSALS WITH LOTS OF SPACES BETWEEN LETTERS
    else:
        kw = [x.replace(' ', '') for x in Keywords]
        Text_Proposal = Text_Proposal.replace(' ', '')
        Keywords_Count_All.append([(Text_Proposal.lower()).count(x) for x in kw])


### MAKE KEYWORD DATAFRAME
Prop_Nb_All = np.array(Prop_Nb_All)
Keywords_Count_All = np.array(Keywords_Count_All)
df_kw = pd.DataFrame(data=Keywords_Count_All, columns=Keywords, index=Prop_Nb_All)
# df_kw.to_csv('keyword_outputs.csv', index=False)

### MAKE TOTAL KEYWORD COUNTS
Keywords_Count_Total = np.array([np.sum(x) for x in Keywords_Count_All])
ind1 = np.where(Keywords_Count_Total > 0)
ind2 = np.where(Keywords_Count_Total > 3)
print("\n\nNumber of proposals that mention keywords:\t\t" + str(len(ind1[0])))
print("Number of proposals that mention keywords > 3 times:\t" + str(len(ind2[0])))

### WARNING IF SKIPPED PROPOSALS
if len(Files_Skipped_All) > 0:
    print(colored(f'\n\n{len(Files_Skipped_All)} files could not be read', 'red', attrs=['bold']))
