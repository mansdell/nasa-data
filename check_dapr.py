"""

Script to check DAPR compliance in NSPIRES proposals

"""


# ============== Import Packages ================

import sys, os, glob, re, pdb
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
                
    ### LOAD PAGE
    p = d.load_page(int(pn))

    ### GET RAW TEXT
    t = p.get_text("text")

    ### FIX ENCODING
    t = t.encode('utf-8', 'replace').decode()
    
    return t


def get_fonts(doc, pn):

    ### LOAD PAGE
    page = doc.load_page(int(pn))

    ### READ PAGE TEXT AS DICTIONARY (BLOCKS == PARAGRAPHS)
    blocks = page.get_text("dict", flags=11)["blocks"]

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
        print("\n\t# [] refs:\t", colored(str(n_brac), 'red'))
    else:
        print("\n\t# [] refs:\t", str(n_brac))
    if n_etal > 10:
        print("\t# et al. refs:\t", colored(str(n_etal), 'yellow'), '\n')
    else:
        print("\t# et al. refs:\t", str(n_etal), '\n')

    return n_brac, n_etal
        

def check_dapr_words(doc, ps_file, pn, stm_pages, ref_pages):

    ### LOAD PROPOSAL MASTER FILE FROM NSPIRES
    dfp = pd.read_csv(ps_file)

    ### GET PI INFO (iNSPIRES FORMAT)
    pi_name = (dfp[dfp['Response number'] == pn]['PI Last name'].values[0]).split(',')[0]
    pi_orgs = (dfp[dfp['Response number'] == pn]['Linked Org'].values[0]).split(', ')
    pi_orgs.append(dfp[dfp['Response number'] == pn]['Company name'].values[0])
    pi_orgs = np.unique(pi_orgs).tolist()
    pi_city = (dfp[dfp['Response number'] == pn]['City'].values[0]).split(',')[0]

    ### REMOVE 'nan' ENTRIES THAT CAN HAPPEN FOR ORG LIST
    if 'nan' in pi_orgs:
        pi_orgs.remove('nan')

    ### GET PI INFO (OTHER FORMAT)
    # pi_name = (dfp[dfp['Response number'] == pn]['PI Last name'].values[0]).split(',')[0]
    # pi_orgs = (dfp[dfp['Response number'] == pn]['Linked Org'].values[0]).split(', ')
    # pi_orgs.append(dfp[dfp['Response number'] == pn]['Company name'].values[0])
    # pi_orgs = np.unique(pi_orgs).tolist()
    # pi_city = (dfp[dfp['Response number'] == pn]['City'].values[0]).split(',')[0]

    ### GET ALL DAPR WORDS
    dw = ['our group', 'our team', 'our work', 'our previous', 'my group', 'my team', 'university', 'department', 'dept.', ' she ', ' he ', ' her ', ' his ']
    dw = dw + pi_orgs + [pi_name] + [pi_city]
    dw = np.unique(dw).tolist()

    ### GET PAGE NUMBERS WHERE DAPR WORDS APPEAR
    ### IGNORES REFERENCE SECTION, IF KNOWN
    dwp = []
    for i, ival in enumerate(dw):
        if pd.isnull(ival):
            continue
        pn = []
        for n, nval in enumerate(np.arange(stm_pages[0], doc.pageCount)):
            if (nval >= np.min(ref_pages)) & (nval <= np.max(ref_pages)) & (np.min(ref_pages) > 5):
                continue
            tp = (get_text(doc, nval)).lower()
            # if (' ' + ival.lower() + ' ' in tp) | (' ' + ival.lower() + "'" in tp) | (' ' + ival.lower() + "." in tp):
            if (ival.lower() in tp):
                pn.append(nval) 
        dwp.append(pn)

    ### RECORD NUMBER OF TIMES EACH WORD FOUND AND UNIQUE PAGE NUMBERS
    # ### PRINT FINDINGS TO SCREEN
    dww, dwcc, dwpp = [], [], []
    for m, mval in enumerate(dwp):
        if len(mval) > 0:
             print(f'\t"{dw[m]}" found {len(mval)} times on pages {np.unique(mval)+1}')
             dww.append(dw[m])
             dwcc.append(len(mval))
             dwpp.append((np.unique(mval)+1).tolist())

    return dww, dwcc, dwpp


def get_pages(d):

    ### GET TOTAL NUMBER OF PAGES IN PDF
    pn = d.pageCount

    ### LOOP THROUGH PDF PAGES
    stm_start, stm_end, ref_start, ref_end, ref_end_bu = 0, -100, -100, -100, -100
    tcr, tcs, pflag = 'white', 'white', ''
    for i, val in enumerate(np.arange(5, pn-1)):
            
        ### READ IN TEXT FROM THIS PAGE AND NEXT PAGE
        t1 = get_text(d, val).replace('\n', '').replace('\t', ' ').replace('   ', ' ').replace('  ', ' ')[0:500]
        t2 = get_text(d, val + 1).replace('\n', '').replace('\t', ' ').replace('   ', ' ').replace('  ', ' ')[0:500]
        t1 = t1.lower()
        t2 = t2.lower()

        ### FIND START OF STM IF FULL NSPIRES PROPOSAL
        if ('section x - budget' in t1) & ('section x - budget' not in t2):
            stm_start = val + 1
            continue

        ### FIND STM END AND REFERENCES START
        if (stm_start != -100) & (('reference' in t2) & ('reference' not in t1)) | (('bibliography' in t2) & ('bibliography' not in t1)):
            stm_end = val
            ref_start = val + 1
            continue

        ### REFERENCE END BACK-UP
        if (ref_start != -100) & (('budget' in t2) & ('budget' not in t1)):
            ref_end_bu = val
            
        ### FIND REF END 
        w1, w2, w3, w4, w5, w6 = 'data management', 'budget justification', 'work plan', 'budget narrative', 'work effort', 'total budget'
        if (ref_start != -100) & ((w1 in t2) & (w1 not in t1)) | ((w2 in t2) & (w2 not in t1)) | ((w3 in t2) & (w3 not in t1)) | ((w4 in t2) & (w4 not in t1)) | ((w5 in t2) & (w5 not in t1)) | ((w6 in t2) & (w6 not in t1)):
            ref_end = val
            if (ref_start != -100) & (ref_end > ref_start) & (stm_end - stm_start > 10):
                break
    
    ### FIX SOME THINGS BASED ON COMMON SENSE
    if ref_end < ref_start:
        ### USE SIMPLE "BUDGET" FLAG IF WE HAVE TO
        ref_end = ref_end_bu
        if ref_end < ref_start:
            ref_end = -100
    if stm_end - stm_start <= 5:
        ### IF STM SECTION REALLY SHORT, ASSUME PTOT PAGES AND FLAG
        ptot = 15
        stm_end, tcs = np.min([stm_start+ptot-1, pn]), 'yellow'
    if (ref_end != -100) & (ref_start == -100) & (stm_end != -100):
        ### IF FOUND END BUT NOT START OF REFERENCES, ASSUME REFS START RIGHT AFTER STM BUT FLAG
        ref_start, tcr = stm_end + 1, 'yellow'
    if ref_end == -100:
        ### IF COULDN'T FIND REF END, ASSUME GOES TO END OF PDF (SOMETIMES THIS IS TRUE) AND FLAG
        ref_end, tcr = pn-1, 'yellow'
    if (tcr == 'yellow') | (tcs == 'yellow'):
        pflag='YES'


    ### IF PROPOSAL INCOMPLETE (E.G., WITHDRAWN) RETURN NOTHING
    if pn - stm_start < 3:
        return [], [], 0, ''

    ### OTHERWISE, RETURN PAGE GUESSES
    else:    
        print(f"\n\tPage Guesses:\n")
        print(colored(f"\t\tSTM = {stm_start+1, stm_end+1}", tcs))
        print(colored(f"\t\tRef = {ref_start+1, ref_end+1}", tcr))
        return [stm_start, stm_end], [ref_start, ref_end], pn, pflag


# ====================== Main Code ========================

### SET PATH TO PDFs
PDF_Path  = 'proposals'

### GET LIST OF PDF FILES
PDF_Files = np.sort(glob.glob(os.path.join(PDF_Path, '*anonproposal.pdf')))
PS_File = 'proposal_master.csv'

### ARRAYS TO FILL
Prop_Nb_All, Font_Size_All, N_Brac_All, N_EtAl_All = [], [], [], []
STM_Pages_All, Ref_Pages_All, pFlag_All = [], [], []
DW_All, DWC_All, DWP_All = [], [], []

### LOOP THROUGH ALL PROPOSALS
for p, pval in enumerate(PDF_Files):

    ### GET PROPOSAL FILE NAME
    Prop_Nb = '21-'+pval.split('-')[-2]+'-'+(pval.split('-')[-1]).split('_')[0]
    # Prop_Nb = '21-'+pval.split('-')[-2].split('_2')[0]+'-' + (pval.split('-')[-1]).split('_')[0]
    # Prop_Nb = '20-'+pval.split('_')[3]
    print(colored(f'\n\n\n\t{Prop_Nb}', 'green', attrs=['bold']))

    ### GET PAGES OF PROPOSAL
    pval = str(pval)
    Doc = fitz.open(pval)
    STM_Pages, Ref_Pages, Tot_Pages, pFlag = get_pages(Doc)
    if Tot_Pages == 0:
        print(f'\n\tProposal incomplete, skipping')
        continue

    ### CHECK FONT SIZE COMPLIANCE
    Font_Size = get_median_font(Doc, STM_Pages[0], STM_Pages[1])

    ### CHECK DAPR REFERENCING COMPLIANCE
    N_Brac, N_EtAl = check_ref_type(Doc, STM_Pages[0], STM_Pages[1])

    ### CHECK DAPR WORDS
    DW, DWC, DWP = check_dapr_words(Doc, PS_File, Prop_Nb, STM_Pages, Ref_Pages)

    ### RECORD STUFF
    Prop_Nb_All.append(Prop_Nb)
    Font_Size_All.append(Font_Size)
    N_Brac_All.append(N_Brac)
    N_EtAl_All.append(N_EtAl)
    STM_Pages_All.append((np.array(STM_Pages) + 1).tolist())
    Ref_Pages_All.append((np.array(Ref_Pages) + 1).tolist())
    pFlag_All.append(pFlag)
    DW_All.append(DW)
    DWC_All.append(DWC)
    DWP_All.append(DWP)

### OUTPUT TO DESKTOP
d = {'Prop_Nb': Prop_Nb_All, 'Font Size': Font_Size_All, 'N_Brac': N_Brac_All, 'N_EtAl':N_EtAl_All,
     'STM_Pages': STM_Pages_All, 'Ref Pages': Ref_Pages_All, 'Flag Pages': pFlag_All,
     'DAPR_Words': DW_All, 'DAPR_Word_Count': DWC_All, 'DAPR_Word_Pages': DWP_All}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(os.path.expanduser("~/Desktop"), 'dapr_checks.csv'), index=False)
