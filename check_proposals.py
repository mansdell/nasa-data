"""

Script to read and analyze data in NSPIRES-formatted PDF

"""


# ============== Import Packages ================

import sys, os, glob, pdb
import numpy as np
import pandas as pd

import textwrap
from termcolor import colored

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import fitz 
fitz.TOOLS.mupdf_display_errors(False)
from collections import Counter
import datetime
import unicodedata


# ============== Define Functions ===============

def get_text(d, pn):

    """
    PURPOSE:   extract text from a given page of a PDF document
    INPUTS:    d = fitz Document object
               pn = page number to read (int)
    OUTPUTS:   t  = text of page (str)

    """
                
    ### LOAD PAGE
    p = d.loadPage(int(pn))

    ### GET RAW TEXT
    t = p.getText("text")
    
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

    ### GET TOTAL NUMBER OF PAGES IN PDF
    pn = d.pageCount

    ### WORDS THAT INDICATE EXTRA STUFF BEFORE PROPOSAL STARTS
    check_words = ["contents", "c o n t e n t s", "budget", "cost", "costs",
                   "submitted to", "purposely left blank", "restrictive notice"]

    ### LOOP THROUGH PDF PAGES
    ps = 0
    for i, val in enumerate(np.arange(pn)):
            
        ### READ IN TEXT FROM THIS PAGE AND NEXT PAGE
        t1 = get_text(d, val)
        t2 = get_text(d, val + 1)
        
        ### FIND PROPOSAL START USING END OF SECTION X IN NSPIRES
        if ('SECTION X - Budget' in t1) & ('SECTION X - Budget' not in t2):

            ### SET START PAGE
            ps = val + 1

            ### ATTEMPT TO CORRECT FOR (ASSUMED-TO-BE SHORT) COVER PAGES
            if len(t2) < 500:
                ps += 1
                t2 = get_text(d, val + 2)

            ### ATTEMP TO ACCOUNT FOR TOC OR EXTRA SUMMARIES
            if any([x in t2.lower() for x in check_words]):
                ps += 1

            ### SET END PAGE ASSUMING AUTHORS USED FULL PAGE LIMIT
            pe  = ps + (pl - 1) 
                        
        ### EXIT LOOP IF START PAGE FOUND
        if ps != 0:
            break 

    ### ATTEMPT TO CORRECT FOR TOC > 1 PAGE OR SUMMARIES THAT WEREN'T CAUGHT ABOVE
    if any([x in get_text(d, ps).lower() for x in check_words]):
        ps += 1
        pe += 1

    ### CHECK THAT PAGE AFTER pe IS REFERENCES
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

    # ## PRINT WARNING IF WENT OVER PAGE LIMIT (THIS WASN'T VERY RELIABLE)
    # if pe - ps > 14:
    #     print(colored("\n\t!!!!! PAGE LIMIT WARNING -- OVER !!!!!", 'blue'))
    # if pe - ps < 13:
    #     print(colored("\n\t!!!!! PAGE LIMIT WARNING -- UNDER !!!!!", 'blue'))

    return pn, ps, pe


def get_fonts(doc, pn):

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
            


def get_phd_data(doc, ps, pi):

    """

    Finds line in PI's CV with PhD info 
    Assumes first CV in proposal in for the PI
    Searches for the words "phd" and grabs nearby information that contains digits
    Assumes those digits are a year
    Returns empty list [] if no information found

    """

    ### OPEN LOG FOR PHD DATA
    phd_data = []

    ### WORDS TO IDENTIFY CV OR PDH
    cv_words = ["curriculum", "vitae", "biographic", "sketches", "cv", "education", "professional"]
    phd_words = ["phd", "ph.d", "d.phil", "philosophy", "dr.rer.nat."]

    ### LOOP THROUGH PDF PAGES
    pn = doc.pageCount
    for i, val in enumerate(np.arange(ps+1, pn)):

        ### EXTRACT/FIX TEXT
        text = get_text(doc, val)
        for r, rval in enumerate([' ', '\n']):
            text = text.replace(rval, '')
        text = text.lower() ### PUT IN LOWER CASE
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")  ### REMOVE ACCENTS

        ### FIX PI NAME
        pi = pi.lower()
        pi = pi.replace(' ', '')
        pi = pi.replace('_', '')

        ### FIND PI's CV IN PROPOSAL
        if (any([x in text for x in cv_words])) & (pi in text[0:int(len(text)/4)]):

            ### ADVANCED: USE STRUCTURE OF PDF PAGE
            page = doc.loadPage(int(val))
            blocks = page.getText("dict", flags=11)["blocks"]
            for b in blocks:
                for l, lval in enumerate(b["lines"]):
                    for s, sval in enumerate(lval["spans"]):

                        # if (val == 48) & ("Ph.D.att" in sval["text"]) :
                        # # if (val == 48) :
                        #     print(sval["text"])
                        #     print("")
                        #     pdb.set_trace()

                        ### CHECK IN PHD MENTION:
                        if any([x in sval["text"].lower().replace(" ","") for x in phd_words]):
                            
                            ### CHECK IF YEAR FOUND
                            if any(i.isdigit() for i in sval["text"]):
                                phd_data.append(sval["text"])
                            
                            ### IF NOT, CHECK ALL OTHER SPANS IF NO YEAR FOUND
                            else:
                                ncheck = np.arange(0, len(lval["spans"]))
                                for c, cval in enumerate(ncheck):
                                    if any(i.isdigit() for i in lval["spans"][cval]["text"]):
                                        phd_data.append(lval["spans"][cval]["text"] + ' ' + sval["text"])
       
                            ### IF NOT, CHECK LINES JUST BEFORE/AFTER MENTION FOR YEAR
                            dl = 2
                            while (len(phd_data) == 0) & (dl < 8):
                                tmp = ''
                                ncheck = np.arange( np.max([0, l-dl]), np.min([len(b["lines"]), l+dl]) )
                                for c, cval in enumerate(ncheck):
                                    ncheck2 = np.arange(0, len(b["lines"][cval]["spans"]))
                                    for c2, cval2 in enumerate(ncheck2):
                                        tmptmp = b["lines"][cval]["spans"][cval2]["text"]
                                        if any(i.isdigit() for i in tmptmp):
                                            tmp = tmp + ' ' + tmptmp
                                if len(tmp) > 0:
                                    phd_data.append(tmp + ' ' + sval["text"])
                                dl += 1

                            # ### CHECK EVEN FURTHER ONLY IF YEAR NOT FOUND YET
                            # if len(phd_data) == 0:

                            #     tmp = ''
                            #     ncheck = np.arange( np.max([0, l-3]), np.min([len(b["lines"]), l+3]) )
                            #     for c, cval in enumerate(ncheck):
                            #         tmptmp = b["lines"][cval]["spans"][0]["text"]
                            #         if any(i.isdigit() for i in tmptmp):
                            #             tmp = tmp + ' ' + tmptmp
                            #     if len(tmp) > 0:
                            #         phd_data.append(tmp + ' ' + sval["text"])

                            # if (val == 48) & ("Ph. D." in sval["text"]) :    
                            #     pdb.set_trace()

            ## SIMPLE: JUST GRAB WHATEVER TEXT IS AROUND MENTION OF PHD (IN FIRST HALF OF CV)    
            ### NEED TO MAKE SMARTER SO DOESN'T TAKE NEARBY YEAR THAT IS MORE RECENT        
            # if len(phd_data) == 0:

            #     ### ONLY GET TEXT IN FIRST HALF OF CV
            #     text = text[0:int(len(text)/1.5)]

            #     ### INDEX LOCATION OF PHD MENTIONS
            #     idx = np.sort([text.index(x) for x in phd_words if x in text])
            #     for x in idx:
            #         if any(i.isdigit() for i in text[x-100:x+100]):
            #             phd_data.append(text[x-100:x+100].replace('\n',' '))
            #             print("TESTING")

            # if val == 43:
            #     print(phd_data)
            #     pdb.set_trace()

            if len(phd_data) > 0:
                return phd_data

    return phd_data


def guess_phd_year(info):

    ### LOOK AT EACH STRING OF PHD INFO PULLED FROM CV
    yrs = []
    for p, pval in enumerate(info):

        ### FOR STORING YEARS JUST IN THIS STRING
        tmp = []

        ### INDEX ALL INTEGERS IN THIS STRING
        dd = np.array([i for i, c in enumerate(pval) if c.isdigit()])

        ### FIND BREAKS IF MULTIPLE YEARS
        ind_dd = np.where(np.roll(dd, 1) - dd != -1)[0]

        ### EXTRACT YEARS 
        for d, dval in enumerate(ind_dd[1:]):
            tmp.append(pval[dd[ind_dd[d]]: dd[ind_dd[d+1]-1]+1])
        if len(ind_dd) > 0:
            tmp.append(pval[dd[ind_dd[-1]]:dd[-1]+1])

        ### ONLY KEEP NUMBERS THAT ARE == 4 DIGITS
        ind_keep = [i for (i,j) in enumerate(tmp) if len(j) == 4]
        tmp = np.array(tmp)[ind_keep].tolist()

        ### REMOVE FUTURE YEARS (TO CATCH EXPECTED PHDs)
        ind_keep = [i for (i,j) in enumerate(tmp) if float(j) <= datetime.datetime.now().year]
        tmp = np.array(tmp)[ind_keep].tolist()
        
        ### ONLY TAKE MOST RECENT (FOR WHEN YEAR SPANS ARE GIVEN)
        if len(tmp) > 0:
            yrs.append(str(np.max(np.array(tmp).astype(int))))

    ### PRINT INFO
    if len(yrs) > 0:

        ### GUESS YEAR
        # yr_guess = str(np.max(np.array(yrs).astype(int)))   ### JUST TAKE MAX
        yr_guess = yrs[0]                                   ### TAKE FIRST MENTION
        
        for p, pval in enumerate(info):
            if p == 0:
                print("\n\tPhD Year (Guess):\t" + yr_guess)
                print("\tPhD Text:\t\t" + info[p][0:40])
            else:
                print("\t\t\t\t" + info[p][0:40])

        return yr_guess, yrs

    return None, None


# ====================== Set Inputs =======================

PDF_Path  = '../panels/XRP/XRP_Proposals_2014_2020/XRP_Proposals_2020'    # PATH TO PROPOSAL PDFs
Out_Path  = '../panels/XRP20_Fonts'                                       # PATH TO OUTPUT
Guess_PhD = False                                                         # GUESS PHD YEAR FROM CV?

# ====================== Main Code ========================

PDF_Files = np.sort(glob.glob(os.path.join(PDF_Path, '*.pdf')))
Prop_Name_All, PhD_Year_All, PhD_Info_All = [], [], []
for p, pval in enumerate(PDF_Files):

    ### OPEN PDF FILE
    Prop_Name = (pval.split('/')[-1]).split('.pdf')[0]
    Doc = fitz.open(pval)
    print(colored("\n\n\n\t" + Prop_Name, 'green', attrs=['bold']))

    ### GET PI NAME (ASSUMES FORMAT OF NAMES!!!)
    PI_Name = ((pval.split('/')[-1]).split('.pdf')[0]).split('-')[-1]

    ### GET PAGES OF PROPOSAL (DOES NOT ACCOUNT FOR ZERO INDEXING; NEED TO ADD 1 WHEN PRINTING)
    try:
        Page_Num, Page_Start, Page_End = get_pages(Doc)
    except RuntimeError:
        print("\tCould not read PDF")
        print(colored("\n\t!!!!!!!!!DID NOT SAVE!!!!!!!!!!!!!!!!", "orange"))
        Files_Skipped.append(pval)
        continue

    ### GET TEXT OF FIRST PAGE TO CHECK
    print("\n\tSample of first page:\t" + textwrap.shorten((get_text(Doc, Page_Start)[100:130]), 40))
    print("\tSample of mid page:\t"     + textwrap.shorten((get_text(Doc, Page_Start + 8)[100:130]), 40))
    print("\tSample of last page:\t"    + textwrap.shorten((get_text(Doc, Page_End)[100:130]), 40))
            
    ### GRAB FONT SIZE & CPI
    cpi = []
    for i, val in enumerate(np.arange(Page_Start, Page_End)):
        cpi.append(len(get_text(Doc, val)) / 44 / 6.5)
        if i ==0:
            df = get_fonts(Doc, val)
        else:
            df = df.append(get_fonts(Doc, val), ignore_index=True)
    cpi = np.array(cpi)

    ### PRINT WARNINGS IF NEEDED (typical text font < 11.8 or CPI > 15.5 on > 1 page)
    CMF = Counter(df['Font'].values).most_common(1)[0][0]
    MFS = round(np.median(df[df['Text'].apply(lambda x: len(x) > 50)]["Size"]), 1)  ## only use text > 50 characters (excludes random smaller text; see histograms for all)
    CPC, CPI = len(cpi[cpi > 15.5]), [round(x, 1) for x in cpi[cpi > 15.5]]
    print("\n\tMost common font:\t" + CMF)
    if MFS <= 11.8:
        print("\tMedian font size:\t", colored(str(MFS), 'yellow'))
    else:
        print("\tMedian font size:\t" + str(MFS))
    if CPC > 1:
        print("\tPages with CPI > 15.5:\t", textwrap.shorten(colored((np.arange(Page_Start, Page_End)[cpi > 15.5] + 1).tolist(), 'yellow'), 70))
        print("\t\t\t\t", textwrap.shorten(colored(CPI, 'yellow'), 70))
    if (MFS <= 11.8) | (CPC > 1):
        print(colored("\n\t!!!!! COMPLIANCE WARNING!!!!!", 'red'))

    ### PLOT HISTOGRAM OF FONTS
    mpl.rc('xtick', labelsize=10)
    mpl.rc('ytick', labelsize=10)
    mpl.rc('xtick.major', size=5, pad=7, width=2)
    mpl.rc('ytick.major', size=5, pad=7, width=2)
    mpl.rc('xtick.minor', width=2)
    mpl.rc('ytick.minor', width=2)
    mpl.rc('axes', linewidth=2)
    mpl.rc('lines', markersize=5)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_title(Prop_Name + "   Median Font = " + str(MFS) + "pt    CPI = " + str(round(np.median(cpi[cpi > 8]), 1)), size=11)
    ax.set_xlabel('Font Size', size=10)
    ax.set_ylabel('Density', size=10)
    ax.axvspan(11.8, 12.2, alpha=0.5, color='gray')
    ax.hist(df["Size"], bins=np.arange(5.4, 18, 0.4), density=True)
    fig.savefig(os.path.join(Out_Path, 'fc_' + pval.split('/')[-1]), bbox_inches='tight', dpi=100, alpha=True, rasterized=True)
    plt.close('all')

    ### GUESS PHD YEAR FROM CV
    if Guess_PhD:

        ### GET PHD INFO AND YAER
        PhD_Info = get_phd_data(Doc, Page_End, PI_Name)
        PhD_Year, _ = guess_phd_year(PhD_Info)

        ### SAVE STUFF
        PhD_Year_All.append(PhD_Year)
        PhD_Info_All.append(PhD_Info)
        Prop_Name_All.append(Prop_Name)
