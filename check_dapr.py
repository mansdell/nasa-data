"""

Script to check DAPR compliance in NSPIRES proposals

"""


# ============== Import Packages ================

import sys, os, glob, re
import numpy as np
import pandas as pd

import textwrap
from termcolor import colored
import matplotlib as mpl
import matplotlib.pyplot as plt

import fitz 
fitz.TOOLS.mupdf_display_errors(False)
from collections import Counter
import datetime
import unicodedata
import gender_guesser.detector as gender


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


def get_fonts(doc, pn):

    """
    PURPOSE:   get font sizes used in the proposal
    INPUTS:    doc = fitz Document object
               pn  = page number to grab fonts (int)
    OUTPUTS:   df  = dictionary with font sizes, types, colors, and associated text

    """

    ### LOAD PAGE
    page = doc.loadPage(int(pn))

    ### READ PAGE TEXT AS DICTIONARY (BLOCKS == PARAGRAPHS)
    blocks = page.getText("dict", flags=11)["blocks"]

    ### ITERATE THROUGH TEXT BLOCKS
    fn, fs, fc, ft = [], [], [], []
    for b in blocks:
        ### ITERATE THROUGH TEXT LINES
        for l in b["lines"]:
            ### ITERATE THROUGH TEXT SPANS
            for s in l["spans"]:
                fn.append(s["font"])
                fs.append(s["size"])
                fc.append(s["color"])
                ft.append(s["text"])

    d = {'Page': np.repeat(pn, len(fn)), 'Font': fn, 'Size': fs, 'Color': fc, 'Text': ft}
    df = pd.DataFrame (d, columns = ['Page', 'Font', 'Size', 'Color', 'Text'])

    return df


def get_pages(d):

    ### GET TOTAL NUMBER OF PAGES IN PDF
    pn = d.pageCount

    ### LOOP THROUGH PDF PAGES
    stm_start, stm_end, ref_start, ref_end, dmp_start, dmp_end, rel_start, rel_end, fte_start, fte_end = 0, -100, -100, -100, -100, -100, -100, -100, -100, -100
    for i, val in enumerate(np.arange(10, pn-1)):
            
        ### READ IN TEXT FROM THIS PAGE AND NEXT PAGE
        t1 = get_text(d, val).replace('\n', '')[0:500]
        t2 = get_text(d, val + 1).replace('\n', '')[0:500]

        ### UGH SOMETIMES THIS MATTERS
        t1 = t1.replace('   ', ' ')
        t1 = t1.replace('  ', ' ')
        t2 = t2.replace('   ', ' ')
        t2 = t2.replace('  ', ' ')
        
        ### FIND STM END AND REFERENCES START
        if (('reference' in t2.lower()) & ('reference' not in t1.lower())) | (('bibliography' in t2.lower()) & ('bibliography' not in t1.lower())):
            stm_end = val
            ref_start = val + 1

        ### FIND REF END AND DMP START
        if ('data management' in t2.lower()) & ('data management' not in t1.lower()):
            ref_end = val
            dmp_start = val + 1

        ### FIND DMP END AND RELEVANCE START
        if ('relevance' in t2.lower()) & ('relevance' not in t1.lower()):
            dmp_end = val
            rel_start = val + 1

        ### FIND BUDGET INFO 
        # if (('work effort' in t2.lower()) | ('budget' in t2.lower())) & (('work effort' not in t1.lower()) | ('budget' not in t1.lower())):
        #     if rel_start != -100:
        #         rel_end = val 
        #     else:
        #         dmp_end = val
        #     fte_start = val + 1
        #     fte_end = pn - 1
        #     if stm_end != -100:
        #         break

    ### IF COULDN'T FIND END OF STM, ASSUME 16 PAGES (CONSERVATIVE SINCE USUAL HAS 1-2 PAGES OF FRONT MATTER)
    if stm_end == -100:
        stm_end = np.min([15, pn])
    
    ### IF COULDN'T START OF REFERENCES, BUT COULD FIND END, ASSUME REFS START RIGHT AFTER STM SECTION
    if (ref_end != -100) & (ref_start == -100):
        ref_start = stm_end + 1

    ### LIMIT REFERENCES TO 5 PAGES
    ### (SINCE IT DETERMINES END BY START OF DMP, AND SOMETIMES DMP IS ELSEWHERE)
    # tc = 'white'
    # if ref_end - ref_start > 5:
    #     ref_end, tc = ref_start + 3, 'red'

    ### ASSUME THAT BUDGET STUFF COMES AFTER RELEVANCE (OR WHATEVER WAS BEFORE THAT IF NO RELEVANCE) 
    ### AND THAT RELEVANCE IS ONLY ONE PAGE
    if rel_start != -100:
        rel_end, fte_start, fte_end, tc2 = rel_start, rel_start + 1, pn-1, 'white'
    elif dmp_end != -100:
        fte_start, fte_end, tc2 = dmp_end+1, pn-1, 'yellow'
    elif dmp_start != -100:
        fte_start, fte_end, tc2 = dmp_start+1, pn-1, 'yellow'
    elif ref_end != -100:
        fte_start, fte_end, tc2 = ref_end+1, pn-1, 'yellow'
    else:
        fte_start, fte_end, tc2 = stm_end+1, pn-1, 'yellow'

    print(f"\n\tPage Guesses:\n")
    print(f"\t\tSTM = {stm_start+1, stm_end+1}")
    print(f"\t\tRef = {ref_start+1, ref_end+1}")
    print(f"\t\tDMP = {dmp_start+1, dmp_end+1}")
    print(f"\t\tRel = {rel_start+1, rel_end+1}")
    print(colored("\t\tFTE = (" + str(fte_start+1) + ", " + str(fte_end+1) + ")", tc2))

    return [stm_start, stm_end], [ref_start, ref_end], [dmp_start, dmp_end], [rel_start, rel_end], [fte_start, fte_end]


def get_median_font(doc, ps, pe):

    ### GRAB FONT SIZE & CPI
    cpi = []
    for i, val in enumerate(np.arange(ps, pe)):
        cpi.append(len(get_text(doc, val)) / 44 / 6.5)
        if i ==0:
            df = get_fonts(doc, val)
        else:
            df = df.append(get_fonts(doc, val), ignore_index=True)
    cpi = np.array(cpi)

    if len(df) == 0:
        return 0

    ### MEDIAN FONT SIZE (PRINT WARNING IF LESS THAN 12 PT)
    ### only use text > 50 characters (excludes random smaller text)
    mfs = round(np.median(df[df['Text'].apply(lambda x: len(x) > 50)]["Size"]), 1)  
    if mfs <= 11.8:
        print("\n\tMed. font size: ", colored(str(mfs), 'yellow'))
    else:
        print("\n\tMed. font size: " + str(mfs))

    return mfs


def check_ref_type(doc, ps, pe):

    ### GRAB TEXT OF STM
    tp = ''
    for n, nval in enumerate(np.arange(ps, pe)):    
        tp = tp + ' ' + get_text(doc, nval)
    tp = tp.lower()

    ### CHECK FOR DAPR COMPLIANCE
    n_brac = len([i.start() for i in re.finditer(']', tp)])
    n_etal = len([i.start() for i in re.finditer('et al', tp)])
    if n_brac < 10:
        print("\n\t# [] refs:\t", colored(str(n_brac), 'yellow'))
    else:
        print("\n\t# [] refs:\t", str(n_brac))
    if n_etal > 10:
        print("\t# et al. refs:\t", colored(str(n_etal), 'yellow'), '\n')
    else:
        print("\t# et al. refs:\t", str(n_etal), '\n')

    return n_brac, n_etal
        

def check_dapr_words(doc, ps_file, ref_pages):

    ### GET PI INFO
    dfp = pd.read_csv(ps_file)
    pi_name = (dfp[dfp['Proposal Number'] == Prop_Nb]['PI Last Name'].values[0]).split(',')[0]
    pi_orgs = (dfp[dfp['Proposal Number'] == Prop_Nb]['Linked Org'].values[0]).split(', ')
    pi_orgs.append(dfp[dfp['Proposal Number'] == Prop_Nb]['PI Company Name'].values[0])
    pi_orgs = np.unique(pi_orgs).tolist()
    pi_city = (dfp[dfp['Proposal Number'] == Prop_Nb]['PI City'].values[0]).split(',')[0]

    ### GET ALL DAPR WORDS
    dw = ['our group', 'our team', 'our work', 'our previous', 'our prior', 'my group', 'my team', 'university', 'department', 'dept.', 'institute', 'institution']
    dw = dw + pi_orgs + [pi_name] + [pi_city]

    ### GET PAGE NUMBERS WHERE DAPR WORDS APPEAR
    ### IGNORES REFERENCE SECTION, IF KNOWN
    dwc = []
    for i, ival in enumerate(dw):
        if pd.isnull(ival):
            continue
        pn = []
        for n, nval in enumerate(np.arange(0, doc.pageCount)):
            if (nval >= np.min(ref_pages)) & (nval <= np.max(ref_pages)) & (np.min(ref_pages) > 5):
                continue
            tp = (get_text(doc, nval)).lower()
            # if (' ' + ival.lower() + ' ' in tp):
            if (ival.lower() in tp):
                pn.append(nval) 
        dwc.append(pn)

    ### PRINT FINDINGS TO SCREEN
    for m, mval in enumerate(dwc):
        if len(mval) > 0:
             print(f'\t"{dw[m]}" found {len(mval)} times on pages {np.unique(mval)+1}')

    # ### INDEX START POINT IN TEXT
    # idx = []
    # for i, val in enumerate(dw):
    #     idx.append([i.start() for i in re.finditer(' ' + val.lower() + ' ', tp)])

    return dw, dwc


# ====================== Main Code ========================

### SET IN/OUT PATHS
PDF_Path  = './pdfs-anon'
Out_Path  = '.' 

### GET LIST OF PDF FILES
PDF_Files = np.sort(glob.glob(os.path.join(PDF_Path, '*anonproposal.pdf')))
PS_File = './ProposalMaster.csv'

### ARRAYS TO FILL
Prop_Nb_All, PI_First_All, PI_Last_All, Font_All, Budget_All = [], [], [], [], []
PhD_Year_All, PhD_Page_All, Zipcode_All, Gender_All, Org_All, Org_Type_All, CoI_Gender_All = [], [], [], [], [], [], []

### LOOP THROUGH ALL PROPOSALS
for p, pval in enumerate(PDF_Files):

    ### GET PROPOSAL FILE NAME
    Prop_Nb = (pval.split('/')[-1]).split('_anon')[0]
    print(colored(f'\n\n\n\t{Prop_Nb}', 'green', attrs=['bold']))

    ### GET PAGES OF PROPOSAL
    Doc = fitz.open(pval)
    STM_Pages, Ref_Pages, DMP_Pages, Rel_Pages, FTE_Pages = get_pages(Doc)

    ### CHECK FONT SIZE COMPLIANCE
    Font_Size = get_median_font(Doc, STM_Pages[0], STM_Pages[1])

    ### CHECK DAPR REFERENCING COMPLIANCE
    N_Brac, N_EtAl = check_ref_type(Doc, STM_Pages[0], STM_Pages[1])

    ### CHECK DAPR WORDS
    DW, DWC = check_dapr_words(Doc, PS_File, Ref_Pages)

