"""

Script to check DAPR compliance in NSPIRES proposals

"""


# ============== Import Packages ================

import sys, os, glob, re, pdb
import numpy as np
import pandas as pd
import argparse

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

    ### GET NUMBER OF BRACKETED REFERENCES
    n_brac = 0
    i_brac = [i.start() for i in re.finditer(']', tp)]
    for i, val in enumerate(i_brac):
        if tp[val-1].isnumeric():
            n_brac += 1

    ### ALSO GET NUMBER OF POSSIBLE PARENTHETICAL REFERENCES
    ### MATCHES REQUIRE NUMBER WITHIN PARENTHASES < 200 (ASSUMES <200 REFS; HELPS CATCH YEARS IN PARENTHESIS)
    n_para = 0
    para_vals = [x for x in re.findall('\(([^)]+)', tp) if x.isnumeric()]
    for i, val in enumerate(para_vals):
        if int(val) < 200:
            n_para += 1

    ### CHECK FOR NUMBER OF ET AL REFERENCES
    n_etal = len([i.start() for i in re.finditer(r'\bet al\b', tp)])

    ### PRINT TO SCREEN
    if n_brac < 10:
        print("\n\t# [] refs:\t", colored(str(n_brac), 'red'))
        if n_para > 20:
            print("\tUsed () instead of []? # () refs:\t", colored(str(n_para), 'red'))
    else:
        print("\n\t# [] refs:\t", str(n_brac))
    if n_etal > 10:
        print("\t# et al. refs:\t", colored(str(n_etal), 'yellow'), '\n')
    else:
        print("\t# et al. refs:\t", str(n_etal), '\n')

    return n_brac, n_etal, n_para
        

def check_dapr_words(doc, ps_file, pn, stm_pages, ref_pages):

    ### LOAD PROPOSAL MASTER FILE FROM NSPIRES
    dfp = pd.read_csv(ps_file)

    ### FIGURE OUT WHICH COLUMN NAMES TO USE (DIFFERENT BETWEEN DIVISIONS)
    colnames = ['Response number', 'PI Last name', 'Linked Org', 'Company name', 'City']
    if colnames[0] not in  np.array(dfp.columns):
        colnames = ['Proposal Number', 'PI Last Name', 'Linked Org', 'PI Company Name', 'PI City']
        
    ### CHECK IF MISMATCH BETWEEN PROPOSAL NUMBER PARSED FROM PDF FILE AND WHAT IS USED IN PROPOSAL MASTER
    if len (dfp[dfp[colnames[0]] == pn]) == 0:
        print("\n\tNo matches found in Proposal Master for this proposal number")
        print("\tCheck for differences in proposal number format between PDF filenames and Proposal Master")
        print(f"\tTest: {pn} vs. {dfp[colnames[0]][0]} --> Update Prop_Nb if needed")
        print("\tQuitting program\n")
        sys.exit()

    ### GET PI INFO (iNSPIRES FORMAT)
    pi_name = (dfp[dfp[colnames[0]] == pn][colnames[1]].values[0]).split(',')
    pi_orgs = (dfp[dfp[colnames[0]] == pn][colnames[2]].values[0]).split(', ')
    pi_orgs.append(dfp[dfp[colnames[0]] == pn][colnames[3]].values[0])
    pi_city = (dfp[dfp[colnames[0]] == pn][colnames[4]].values[0]).split(',')[0]

    ### GET OTHER TEAM MEMBER NAMES
    for i, val in enumerate(np.arange(14)+1):

        ### MATCH THE TEAM MEMBER COLUMN NAME (HAS CHANGED BETWEEN YEARS AND/OR DIVISIONS)
        if 'Member - 1 Member name; Role; Email; Relationship_org; Phone' in dfp.columns:
            col = f'Member - {val} Member name; Role; Email; Relationship_org; Phone'
            idx = [0, 0, 3]
        elif 'Member - 1 Member SUID; Name; Role; Email; Organization; Phone':
            col = f"Member - {val} Member SUID; Name; Role; Email; Organization; Phone"
            idx = [0, 1, 4]
        else:
            print("Team member column name not found")
            sys.exit()

        ### GRAB INFO
        if col not in dfp.columns:
            break
        if pd.isnull(dfp[dfp[colnames[0]] == pn][col].values[0]):
            break
        else:
            tm_name = dfp[dfp[colnames[0]] == pn][col].values[idx[0]].split('; ')[idx[1]].split(', ')[0]
            tm_orgs = dfp[dfp[colnames[0]] == pn][col].values[idx[0]].split('; ')[idx[2]].split(', ')[0]
            pi_name.append(tm_name)
            pi_orgs.append(tm_orgs)
    
    ### CLEAN THINGS UP
    pi_orgs = np.unique(pi_orgs).tolist()
    pi_name = np.unique(pi_name).tolist()
    pi_city = np.unique(pi_city).tolist()
    if 'nan' in pi_orgs:
        pi_orgs.remove('nan')
    if '' in pi_orgs:
        pi_orgs.remove('')
    if ';' in pi_orgs:
        pi_orgs.remove(';')

    ### GET PI INFO (OTHER FORMAT)
    # pi_name = (dfp[dfp['Response number'] == pn]['PI Last name'].values[0]).split(',')[0]
    # pi_orgs = (dfp[dfp['Response number'] == pn]['Linked Org'].values[0]).split(', ')
    # pi_orgs.append(dfp[dfp['Response number'] == pn]['Company name'].values[0])
    # pi_orgs = np.unique(pi_orgs).tolist()
    # pi_city = (dfp[dfp['Response number'] == pn]['City'].values[0]).split(',')[0]

    ### GET ALL DAPR WORDS
    # dw = ['our group', 'our team', 'our work', 'our previous', 'my group', 'my team', 'university', 'department', 'dept.', ' she ', ' he ', ' her ', ' his ']
    dw_gp = ['she', 'he', 'her', 'hers', 'his', 'him']
    dw = dw_gp + pi_orgs + pi_name + pi_city
    dw = np.unique(dw).tolist()
        
    ### GET PAGE NUMBERS WHERE DAPR WORDS APPEAR
    ### IGNORES REFERENCE SECTION, IF KNOWN
    dwp, dwc, dww = [], [], []
    for i, ival in enumerate(dw):
        if pd.isnull(ival):
            continue
        for n, nval in enumerate(np.arange(stm_pages[0], doc.page_count)):

            if (nval >= np.min(ref_pages)) & (nval <= np.max(ref_pages)) & (np.min(ref_pages) > 5):
                continue
            tp = (get_text(doc, nval)).lower()
            wi = [[i.start(), i.end()] for i in re.finditer(r'\b' + re.escape(ival.lower()) + r'\b', tp)]

            for m, mval in enumerate(wi):

                ### CHECK IF GENDER PRONOUN CATCHES ARE ACTUALLY HE/SHE, HIM/HER, ETC.
                ### ONLY SAVE DW INFO IF NOT
                if ival in dw_gp:
                    if not (tp[mval[0]-1] == '/') | (tp[mval[1]] == '/'):

                        ### ONLY SAVE FLAGS FOR FIRST OCCURENCE ON PAGE
                        if m == 0:
                            dwp.append(nval)
                            dwc.append(len(wi)) 
                            dww.append(ival)
                            print(f'\t"{ival}" found {len(wi)} times on pages {nval+1}')
                else:

                    ### ONLY SAVE FLAGS FOR FIRST OCCURENCE ON PAGE
                    if m == 0:
                        dwp.append(nval)
                        dwc.append(len(wi)) 
                        dww.append(ival)      
                        print(f'\t"{ival}" found {len(wi)} times on pages {nval+1}')
                    
    return dww, dwc, dwp, pi_name
    

def get_pages(d, stm_pl=15):

    ### GET TOTAL NUMBER OF PAGES IN PDF
    pn = d.page_count

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
        if (stm_start != -100) & (('references' in t2) & ('references' not in t1)) | (('bibliography' in t2) & ('bibliography' not in t1)) | (('citations' in t2) & ('citations' not in t1)):
            stm_end = val
            ref_start = val + 1

        ### FIND REF END 
        # w1, w2, w3, w4, w5, w6, w7 = 'data management', 'budget justification', 'work plan', 'budget narrative', 'work effort', 'total budget'
        # if (ref_start != -100) & ((w1 in t2) & (w1 not in t1)) | ((w2 in t2) & (w2 not in t1)) | ((w3 in t2) & (w3 not in t1)) | ((w4 in t2) & (w4 not in t1)) | ((w5 in t2) & (w5 not in t1)) | ((w6 in t2) & (w6 not in t1)):
        w1, w2, w3, w4, w5, w6 = 'budget justification', 'budget narrative', 'total budget', 'table of work effort', 'data management', 'table of personnel'
        if (ref_start != -100) & ((w1 in t2) & (w1 not in t1)) | ((w2 in t2) & (w2 not in t1)) | ((w3 in t2) & (w3 not in t1)) | ((w4 in t2) & (w4 not in t1)) | ((w5 in t2) & (w5 not in t1)) | ((w6 in t2) & (w6 not in t1)):
            ref_end = val
            if (ref_start != -100) & (ref_end > ref_start) & (stm_end - stm_start > stm_pl-5):
                break

    ### FIX SOME THINGS BASED ON COMMON SENSE
    if ref_end < ref_start:
        ### USE SIMPLE "BUDGET" FLAG IF WE HAVE TO
        ref_end = ref_end_bu
        if ref_end < ref_start:
            ref_end = -100
    if stm_end - stm_start <= 5:
        ### IF STM SECTION REALLY SHORT, ASSUME ALL PAGES USED AND FLAG
        stm_end, tcs = np.min([stm_start+stm_pl-1, pn]), 'yellow'
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

### GET ARGUMENTS
### NOTE PDF_SUFFIX USES "REMAINDER" SO IT CAN HANDLE STRINGS STARTING WITH "-"
parser = argparse.ArgumentParser()
parser.add_argument("PDF_Path", type=str, help="path to anonymized proposal PDF")
parser.add_argument("PDF_Suffix", type=str, help="suffix of anonymized proposal PDF (what is before .pdf but after proposal number)", nargs=argparse.REMAINDER)
parser.add_argument("PM_Path", type=str, help="path to team info (NSPIRES cover pages or .csv file)")
args = parser.parse_args()

### GET LIST OF PDF FILES
### CHANGE IF NRESS USED DIFFERENT SUFFIX
PDF_Files = np.sort(glob.glob(os.path.join(args.PDF_Path, '*' + args.PDF_Suffix[0] + '.pdf')))
if len(PDF_Files) == 0:
    print("\nNo files found in folder set by PDF_Path\nCheck directory path in PDF_Path and PDF suffix in PDF_Files\nQuitting program\n")
    sys.exit()

### GET PROPOSAL MASTER
if os.path.isfile(args.PM_Path) == False:
    print("\nNo Proposal Master file found in path set by PS_File\nCheck path for Proposal Master\nQuitting program\n")
    sys.exit()  

### ARRAYS TO FILL
Prop_Nb_All, TMN_All, Font_Size_All, N_Brac_All, N_EtAl_All, N_Para_All = [], [], [], [], [], []
STM_Pages_All, Ref_Pages_All, pFlag_All = [], [], []
DW_All, DWC_All, DWP_All = [], [], []

### LOOP THROUGH ALL PROPOSALS
for p, pval in enumerate(PDF_Files):

    ### GET PROPOSAL FILE NAME
    Prop_Nb = pval.split('/')[-1].split(args.PDF_Suffix[0])[0]
    print(colored(f'\n\n\n\t{Prop_Nb}', 'green', attrs=['bold']))

    ### GET PAGES OF PROPOSAL
    pval = str(pval)
    Doc = fitz.open(pval)
    STM_Pages, Ref_Pages, Tot_Pages, pFlag = get_pages(Doc)        
    if Tot_Pages == 0:
        print(f'\n\tProposal incomplete, skipping')
        continue

    ### FIX ANY PROPOSALS THAT COULDN'T FIND REFERENCE SECTION (PAGE NUMBERS ARE FROM PDF, NOT PYTHON ZERO BASE)
    Prop_Nb_Fix, Ref_Pages_Fix = ['23-HSR23_2-0002', '23-HSR23_2-0184'], [[35,39], [33,33]]
    if Prop_Nb in Prop_Nb_Fix:
        print(colored(f"\t\tRef_Fixed = {Ref_Pages_Fix[Prop_Nb_Fix.index(Prop_Nb)][0], Ref_Pages_Fix[Prop_Nb_Fix.index(Prop_Nb)][1]}", 'yellow'))
        Ref_Pages = [Ref_Pages_Fix[Prop_Nb_Fix.index(Prop_Nb)][0]-1, Ref_Pages_Fix[Prop_Nb_Fix.index(Prop_Nb)][1]-1]

    ### CHECK FONT SIZE COMPLIANCE 
    Font_Size = get_median_font(Doc, STM_Pages[0], STM_Pages[1])

    ### CHECK DAPR REFERENCING COMPLIANCE
    N_Brac, N_EtAl, N_Para = check_ref_type(Doc, STM_Pages[0], STM_Pages[1])

    ### CHECK DAPR WORDS (AND GRAB TEAM MEMBER NAMES)
    DW, DWC, DWP, TMN = check_dapr_words(Doc, args.PM_Path, Prop_Nb, STM_Pages, Ref_Pages)

    ### RECORD STUFF
    Prop_Nb_All.append(Prop_Nb)
    Font_Size_All.append(Font_Size)
    N_Brac_All.append(N_Brac)
    N_EtAl_All.append(N_EtAl)
    N_Para_All.append(N_Para)
    STM_Pages_All.append((np.array(STM_Pages) + 1).tolist())
    Ref_Pages_All.append((np.array(Ref_Pages) + 1).tolist())
    pFlag_All.append(pFlag)
    DW_All.append(DW)
    DWC_All.append(DWC)
    DWP_All.append((np.array(DWP) + 1).tolist())
    TMN_All.append(TMN)


### OUTPUT TO DESKTOP
d = {'Prop_Nb': Prop_Nb_All, 'Team Members': TMN_All, 'Font Size': Font_Size_All, 
     'N_Brac': N_Brac_All, 'N_EtAl':N_EtAl_All, 'N_Para':N_Para_All,
     'STM_Pages': STM_Pages_All, 'Ref Pages': Ref_Pages_All, 'Flag Pages': pFlag_All,
     'DAPR_Words': DW_All, 'DAPR_Word_Count': DWC_All, 'DAPR_Word_Pages': DWP_All}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(os.path.expanduser("~/Desktop"), 'dapr_checks.csv'), index=False)
