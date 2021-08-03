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

    ### CHECK THAT PAGE AFTER END PAGE IS REFERENCES
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
            


def get_phd_data(doc, ps, pi):

    """
    PURPOSE:   finds line(s) in PI's CV with PhD info 
               assumes PI's CV is the first in proposal and format is YYYY
    INPUTS:    doc = fitz Document object
               ps  = start page of end material (int)
               pi  = name of PI (str)
    OUTPUTS:   returns potential text containing PhD year (list of str)
               returns empty list [] if no information found

    """

    ### OPEN LOG FOR PHD DATA
    phd_data, phd_page = [], []

    ### WORDS TO IDENTIFY CV OR PhD
    cv_words = ["curriculum", "vitae", "biographic", "sketches", "cv", "education", "professional", "experience"]
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

            ### USE STRUCTURE OF PDF PAGE
            page = doc.loadPage(int(val))
            blocks = page.getText("dict", flags=11)["blocks"]
            for b in blocks:
                for l, lval in enumerate(b["lines"]):
                    for s, sval in enumerate(lval["spans"]):

                        ### CHECK IN PHD MENTION:
                        if any([x in sval["text"].lower().replace(" ","") for x in phd_words]):
                            
                            ### CHECK IF YEAR FOUND
                            if any(i.isdigit() for i in sval["text"]):
                                phd_data.append(sval["text"])
                                phd_page.append(val)
                            
                            ### IF NOT, CHECK ALL OTHER SPANS IF NO YEAR FOUND
                            else:
                                ncheck = np.arange(0, len(lval["spans"]))
                                for c, cval in enumerate(ncheck):
                                    if any(i.isdigit() for i in lval["spans"][cval]["text"]):
                                        phd_data.append(lval["spans"][cval]["text"] + ' ' + sval["text"])
                                        phd_page.append(val)

                            ### IF NOT, CHECK LINES JUST BEFORE/AFTER MENTION FOR YEAR
                            dl = 2
                            while (len(phd_data) == 0) & (dl < 15):

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
                                    phd_page.append(val)
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


            ## SIMPLE: JUST GRAB WHATEVER TEXT IS AROUND MENTION OF PHD (IN FIRST HALF OF CV)    
            ### [THIS SIMPLE APPROACH OFTEN GRABS THE WRONG DATE]       
            # if len(phd_data) == 0:

            #     ### ONLY GET TEXT IN FIRST HALF OF CV
            #     text = text[0:int(len(text)/1.5)]

            #     ### INDEX LOCATION OF PHD MENTIONS
            #     idx = np.sort([text.index(x) for x in phd_words if x in text])
            #     for x in idx:
            #         if any(i.isdigit() for i in text[x-100:x+100]):
            #             phd_data.append(text[x-100:x+100].replace('\n',' '))
            #             print("TESTING")
            
            if len(phd_data) > 0:
                return phd_data, phd_page

    return phd_data, phd_page


def guess_phd_year(info, page):

    """
    PURPOSE:   guess the PhD year of PI
    INPUTS:    info grabbed from proposal by get_phd_data (list of str or None)
    OUTPUTS:   yr_guess = best-guess of PhD year of PI (str or None)
               yrs = all years extracted (list of str or None)

    """

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

        ### ONLY KEEP NUMBERS THAT ARE == 4 DIGITS AND START WITH 1 OR 2
        ind_keep = [i for (i,j) in enumerate(tmp) if (len(j) == 4) & (1 <= int(j[0]) <=2)]
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
        yr_guess = yrs[0]                                     ### TAKE FIRST MENTION
        
        for p, pval in enumerate(info):
            if p == 0:
                print(f"\n\tPhD Year (Guess):\t{yr_guess}")
                print(f"\tPhD Text [{page[p]+1}]:\t\t{info[p]}")
            else:
                print(f"\t\t\t\t{info[p]}")

        return yr_guess, yrs

    return None, None


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
    pi_first, pi_last = pi_name[0], pi_name[-1]

    ### GET PROPOSAL NUMBER
    pn = ((cp[cp.index('Proposal Number'):cp.index('NASA PROCEDURE FOR')]).split('\n')[1]).split(' ')[0]

    return pi_first, pi_last, pn


def check_compliance(doc, ps, pe):

    """
    PURPOSE:   check font size and counts-per-inch 
    INPUTS:    doc = fitz Document object
               ps  = start page of proposal (int)
               pe  = end page of proposals (int)
    OUTPUTS:   mfs = median font size of proposal (int)
  
    """

    ### GRAB FONT SIZE & CPI PER LINE
    for i, val in enumerate(np.arange(ps, pe + 1)):
        t = get_text(doc, val)
        ln = t.split('\n')
        ln = [x for x in ln if len(x) > 50] ## TRY TO ONLY KEEP REAL LINES
        if i ==0:
            df = get_fonts(doc, val)
            cpi = [round(len(x)/6.5,2) for x in ln[2:-2]]  ### TRY TO AVOID HEADERS/FOOTERS
            lns, lpi = ln[2:-2], [round(len(ln)/9, 2)]
        else:
            df = df.append(get_fonts(doc, val), ignore_index=True)
            cpi = cpi + [round(len(x)/6.5,2) for x in ln[2:-2]]
            lns = lns + ln[2:-2]
            lpi.append(round(len(ln)/9, 2))
    cpi, lns, lpi = np.array(cpi), np.array(lns), np.array(lpi)

    ### RETURN IF COULDN'T READ
    if len(df) == 0:
        return 0

    ### MEDIAN FONT SIZE (PRINT WARNING IF LESS THAN 12 PT)
    ### only use text > 50 characters (excludes random smaller text; see histograms for all)
    mfs = round(np.median(df[df['Text'].apply(lambda x: len(x) > 50)]["Size"]), 1)  
    if mfs <= 11.8:
        print("\n\tMedian font size:\t", colored(str(mfs), 'yellow'))
    else:
        print("\n\tMedian font size:\t" + str(mfs))

    ### MOST COMMON FONT TYPE USED
    # cft = Counter(df['Font'].values).most_common(1)[0][0]
    # print("\n\tMost common font:\t" + cft)

    ### COUNTS PER INCH
    cpi_max, lpi_max = 16.0, 5.5
    ind_cpi, ind_lpi = np.where(cpi > cpi_max), np.where(lpi > lpi_max)
    cpi, lns, lpi, pgs = cpi[ind_cpi], lns[ind_cpi], lpi[ind_lpi], (np.arange(ps, pe+1)+1)[ind_lpi].tolist()
    if len(lpi) >= 1:
        print(f"\tPages w/LPI > {lpi_max}:\t", colored(len(lpi), 'red'), colored(lpi, 'red'), colored(pgs, 'red'))
    else:
        print(f"\tPages w/LPI > {lpi_max}:\t", len(lpi), lpi)
    if len(cpi) >= 8:
        print(f"\n\tLines w/CPI > {cpi_max}:\t", colored(len(cpi), 'yellow'))
        [print('\t\t\t\t',textwrap.shorten(x, 60)) for x in lns]
    else:
        print(f"\tLines w/CPI > {cpi_max}:\t", len(cpi))

    # ### PLOT HISTOGRAM OF FONTS
    # mpl.rc('xtick', labelsize=10)
    # mpl.rc('ytick', labelsize=10)
    # mpl.rc('xtick.major', size=5, pad=7, width=2)
    # mpl.rc('ytick.major', size=5, pad=7, width=2)
    # mpl.rc('xtick.minor', width=2)
    # mpl.rc('ytick.minor', width=2)
    # mpl.rc('axes', linewidth=2)
    # mpl.rc('lines', markersize=5)
    # fig = plt.figure(figsize=(6, 4))
    # ax = fig.add_subplot(111)
    # ax.set_title(Prop_Name + "   Median Font = " + str(MFS) + "pt    CPI = " + str(round(np.median(cpi[cpi > 8]), 1)), size=11)
    # ax.set_xlabel('Font Size', size=10)
    # ax.set_ylabel('Density', size=10)
    # ax.axvspan(11.8, 12.2, alpha=0.5, color='gray')
    # ax.hist(df["Size"], bins=np.arange(5.4, 18, 0.4), density=True)
    # fig.savefig(os.path.join(out_path, 'fc_' + pval.split('/')[-1]), bbox_inches='tight', dpi=100, alpha=True, rasterized=True)
    # plt.close('all')

    return mfs, cpi, lns, lpi, pgs


def get_demographics(doc, pi_first):

    """
    PURPOSE:   grab and/or guess demographic information from cover page
    INPUTS:    doc  = fitz Document object
               pi_first = PI first name (str)
    OUTPUTS:   gndr = guessed gender of PI based on first name (str)
               org_type = organization type of PI (str)
               zip_code = zip code of PI (int)
               coi = gender counts of coi's (str; {#male}_{#female})

    """

    ### GET COVER PAGE
    cp = get_text(doc, 0)

    ### GUESS GENDER BASED ON FIRST NAME
    gdDB = gender.Detector()
    gndr = gdDB.get_gender(pi_first.title())

    ### GRAB ZIP CODE FROM COVER PAGE 
    zip_code = ((cp[cp.index('Postal Code'):cp.index('Country Code')]).split('\n')[1]).split('-')[0]

    ### GRAB ORG NAME
    org_name = ((cp[cp.index('Organization Name (Standard/Legal Name)'):cp.index('Company Division')]).split('\n')[1]).split('-')[0]

    ### GRAB BUDGET, IF AVAILABLE
    try:
        bt = float((((cp[cp.index('Total Budget'):cp.index('Year 1 Budget')]).split('\n')[1]).split('-')[0]).replace(',',''))
        b1 = float((((cp[cp.index('Year 1 Budget'):cp.index('Year 2 Budget')]).split('\n')[1]).split('-')[0]).replace(',',''))
        b2 = float((((cp[cp.index('Year 2 Budget'):cp.index('Year 3 Budget')]).split('\n')[1]).split('-')[0]).replace(',',''))
        b3 = float((((cp[cp.index('Year 3 Budget'):cp.index('Year 4 Budget')]).split('\n')[1]).split('-')[0]).replace(',',''))
        b4 = float((((cp[cp.index('Year 4 Budget'):cp.index('SECTION II -')]).split('\n')[1]).split('-')[0]).replace(',',''))
    except ValueError:
        bt, b1, b2, b3, b4 = -99.0, -99.0, -99.0, -99.0, -99.0

    ### PRINT OUT
    print(f'\n\tGender (Guess):\t\t{gndr} ({pi_first})')
    print(f'\tZipcode:\t\t{zip_code}')

    ### GRAB CO-I GENDERS (AND ANY SCIENCE PIs) FROM NEXT TWO COVER PAGES
    coi_gndr, spi = [], []
    cp2 = get_text(doc, 1)
    if "SECTION VI - Team Members" in get_text(doc, 2):
        cp2 = cp2 + get_text(doc, 2)
    while 'Co-I' in cp2:
        if 'Co-I/Science PI' in cp2:
            cp2 = cp2[cp2.index('Co-I/Science PI'):]
            spi_first, spi_last = ((cp2[cp2.index('Co-I/Science PI'):cp2.index('Contact')]).split('\n')[-2]).split(' ')[0], ((cp2[cp2.index('Co-I/Science PI'):cp2.index('Contact')]).split('\n')[-2]).split(' ')[-1]
            spi_gndr = gdDB.get_gender(spi_first.title())
            spi = [spi_first, spi_last, spi_gndr]
            cp2 = cp2[cp2.index('Phone'):]
        else:
            cp2 = cp2[cp2.index('Co-I'):]
            cn = ((cp2[cp2.index('Co-I'):cp2.index('Contact')]).split('\n')[-2]).split(' ')[0]
            coi_gndr.append(gdDB.get_gender(cn.title()))
            cp2 = cp2[cp2.index('Phone'):]
    nm = coi_gndr.count('male') + coi_gndr.count('mostly_male')
    nf = coi_gndr.count('female') + coi_gndr.count('mostly_female')
    coi = f'{nm}_{nf}'

    ### GRAB ORG & ORG TYPE
    for i, val in enumerate(np.arange(20)):            
        text = get_text(doc, val)  
        if ('Question 2 : Type of institution' in text):
            org_type = (((text[text.index('Question 2'):text.index('Question 3')]).split('\n')[-2]).split(':')[-1]).strip()
            break
    if org_type == '':
        org_type = 'Not Specified'

    return gndr, org_name, org_type, zip_code, coi, [bt, b1, b2, b3, b4], spi


# ====================== Main Code ========================

### SET IN/OUT PATHS
PDF_Path  = '../panels/XRP/XRP_Proposals_2014_2021/XRP_Proposals_2021'
Out_Path  = '../panels/XRP' 

### GET LIST OF PDF FILES
PDF_Files = np.sort(glob.glob(os.path.join(PDF_Path, '*.pdf')))

### ARRAYS TO FILL
Prop_Nb_All, PI_First_All, PI_Last_All, Budget_All = [], [], [], []
Font_All, CPI_All, CPI_Lines_All, LPI_All, LPI_Pages_All = [], [], [], [], []
PhD_Year_All, PhD_Page_All, Zipcode_All, Gender_All, Org_All, Org_Type_All, CoI_Gender_All = [], [], [], [], [], [], []

### LOOP THROUGH ALL PROPOSALS
for p, pval in enumerate(PDF_Files):

    ### OPEN PDF DOCUMENT
    pval = str(pval)
    Doc = fitz.open(pval)

    ### GET PI NAME AND PROPOSAL NUMBER
    PI_First, PI_Last, Prop_Nb = get_proposal_info(Doc)
    print(colored(f'\n\n\n\t{Prop_Nb}\t{PI_Last}', 'green', attrs=['bold']))

    ### GET PAGES OF S/T/M PROPOSAL
    try:
        Page_Num, Page_Start, Page_End = get_pages(Doc)         
    except RuntimeError:
        print("\tCould not read PDF, did not save")
        continue

    ### PRINT SOME TEXT TO CHECK
    print("\n\tSample of first page:\t" + textwrap.shorten((get_text(Doc, Page_Start)[300:400]), 60))
    print("\tSample of mid page:\t"     + textwrap.shorten((get_text(Doc, Page_Start + 8)[300:400]), 60))
    print("\tSample of last page:\t"    + textwrap.shorten((get_text(Doc, Page_End)[300:400]), 60))  
    
    ### CHECK FONT/TEXT COMPLIANCE
    Font_Size, CPI, CPI_Lines, LPI, LPI_Pages = check_compliance(Doc, Page_Start, Page_End)

    ### GUESS PHD YEAR FROM CV
    PhD_Info, PhD_Page = get_phd_data(Doc, Page_End, PI_Last)
    PhD_Year, _ = guess_phd_year(PhD_Info, PhD_Page)

    ### DEMOGRAPHIC INFORMATION
    PI_Gender, PI_Org, PI_Org_Type, PI_Zip, CoI_Gender, Budget, sPI = get_demographics(Doc, PI_First)

    ### SAVE STUFF
    Prop_Nb_All.append(Prop_Nb)
    PI_Last_All.append(PI_Last)
    PI_First_All.append(PI_First)
    Font_All.append(Font_Size)
    CPI_All.append(CPI)
    CPI_Lines_All.append(CPI_Lines)
    LPI_All.append(LPI)
    LPI_Pages_All.append(LPI_Pages)
    PhD_Year_All.append(PhD_Year)
    PhD_Page_All.append(list(np.unique(PhD_Page)))
    Zipcode_All.append(PI_Zip)
    Gender_All.append(PI_Gender)
    Org_Type_All.append(PI_Org_Type)
    Org_All.append(PI_Org)
    CoI_Gender_All.append(CoI_Gender)
    Budget_All.append(Budget)

    ### SAVE SCIENCE PI INFO, IF NEEDED
    if len(sPI) != 0:
        sPI_PhD_Info, sPI_PhD_Page = get_phd_data(Doc, Page_End, sPI[1])
        sPI_PhD_Year, _ = guess_phd_year(sPI_PhD_Info, sPI_PhD_Page)
        Prop_Nb_All.append(Prop_Nb+'_SciencePI')
        PI_Last_All.append(sPI[1])
        PI_First_All.append(sPI[0])
        Font_All.append(Font_Size)
        CPI_All.append(CPI)
        CPI_Lines_All.append(CPI_Lines)
        LPI_All.append(LPI)
        LPI_Pages_All.append(LPI_Pages)
        PhD_Year_All.append(sPI_PhD_Year)
        PhD_Page_All.append(list(np.unique(sPI_PhD_Page)))
        Zipcode_All.append(PI_Zip)
        Gender_All.append(sPI[2])
        Org_Type_All.append(PI_Org_Type)
        Org_All.append(PI_Org)
        CoI_Gender_All.append(CoI_Gender)
        Budget_All.append(Budget)     

d = {'Prop_Nb': Prop_Nb_All, 'PI_Last': PI_Last_All, 'PI_First': PI_First_All, 
     'Font_Size': Font_All, 'CPI': CPI_All, 'CPI_Lines': CPI_Lines_All, 'LPI': LPI_All, 'LPI_Pages': LPI_Pages_All,
     'PhD_Year': PhD_Year_All, 'PhD_Page': PhD_Page_All, 'Gender': Gender_All, 'Zipcode': Zipcode_All, 
     'Org_Name': Org_All, 'Org_Type': Org_Type_All, 'CoI_Gender': CoI_Gender_All, 'Budget_Total_Years': Budget_All}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(Out_Path, 'outputs.csv'), index=False)
