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


# ====================== Set Inputs =======================

### KEYWORDS TO SEARCH FOR
Keywords = ["machine learning", "deep learning", "artificial intelligence"]

### SET I/O PATHS
PDF_Path = './MyPDFs'
Out_Path  = './MyOutputs'

### GET LIST OF PDF FILES
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
    for i, val in enumerate(np.arange(Page_Start, Page_End)):    
        Text_Proposal = Text_Proposal + ' ' + get_text(Doc, val)

    Prop_Nb_All.append(Prop_Nb)
    Keywords_Count_All.append(np.sum(np.array([(Text_Proposal.lower()).count(x) for x in Keywords])))

ind1 = np.where(np.array(Keywords_Count_All) > 0)
ind2 = np.where(np.array(Keywords_Count_All) > 3)

print("\n\tNumber of proposals that mention keywords:\t" + str(len(ind1[0])))
print("\tNumber of proposals that mention keywords >3 times:\t" + str(len(ind2[0])))