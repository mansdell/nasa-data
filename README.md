# Background

This directory contains some code to pull and analyze data from [NSPIRES](https://nspires.nasaprs.com/external/)-formatted proposals submitted to NASA [ROSES](https://science.nasa.gov/researchers/roses-blogs) calls, including checks on compliance with certain NASA policies.

Code author: Megan Ansdell [@mansdell](https://github.com/mansdell)

# Setup

### Required Non-standard Packages

[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/): a useful package for importing PDF text (which confusingly is imported as "import fitz"). To install, use: 
```
pip install PyMuPDF==1.21.1
```
[gender-guesser](https://pypi.org/project/gender-guesser/): only needed for one part of for check_proposals.py

# Description

### check_proposals.py

This code reads in an NSPIRES-formatted PDF submitted to a NASA ROSES call, attempts to find the "Scientific / Technical / Management" section (hereafter "the proposal"), and then grabs/checks a variety of useful things that are output into a csv file. These things are described below. 

The code requires 1 input, which is the path to the full proposal PDFs generated by NSPIRES (in quotation marks). The code can also take 1 optional input, which is the page limit of the proposal (as an integer) that is otherwise defaulted to 15.

Example command line input with default page limit: 
```
    python check_proposals.py "/Users/mansdell/Proposals/"
```

Example command line input with user-specified page limit: 
```
    python check_proposals.py "/Users/mansdell/Proposals/" --Page_Limit 6
```

The code outputs its findings to the terminal as it checks each proposal. When all proposals are checked, the code will also output a final CSV file named "format_checks.csv" to your Desktop. The information includes:

* PI name and proposal number
  - These are taken from the cover page of the NSPIRES-formatted PDF
  
* Font size (useful for checking compliance)
  - The median font size used in the proposal is calculated, and a warning is given when <=11.8 pt (e.g., for checking compliance)
  - A histogram of the font sizes (based on each PDF ["span"](https://pymupdf.readthedocs.io/en/latest/faq.html#how-to-analyze-font-characteristics)) can be saved to the specified output directory (right now this isn't an active feature, which is a fancy way of saying I commented it out).
  
* Lines per inch (LPI) and characters per inch (CPI)
  - LPI is calculated per page and a warning is given for LPI > 5.5 with the page number
  - CPI is calculated per line and a warning is given for CPI > 16.0 with the line text
  - Note that PDF formats are weird, so these calculations are difficult and results should be checked.
 
* PhD Year (useful for identifying early career proposers)
  - The PhD year of the PI is extracted from the CV that is included within the PDF after the main proposal 
  - When extracted, the PhD year is correct in ~95% of cases (in some cases, no PhD year can be found, or a PhD year isn't provided in the proposal)
  - The text from which the year was guessed, and the page of the proposal from which it was extracted, are printed to the screen and useful for double checking

* Demographic information
  - Inferred gender of the PI based on the first name using [gender-guesser](https://pypi.org/project/gender-guesser/)
  - Zipcode of the PI (useful for geographic analysis)
  - Organization type (specified by the PI via NSPIRES)
  - Number of male and female Co-I's (based on inferred gender, as for the PI)

  
### check_dapr.py

This code reads in an anonymized proposal submitted to a ROSES program that follows Dual-Anonymouse Peer Reivew (DAPR); can be redacted NSPIRES-generated PDF or just the anonymized proposal PDF. The code attempts to find the different sections of the proposal (STM, DMP, Relevance, Budget) and then checks a variety of things to make sure it is DAPR compliant. The outputs are described below.

The code requires 3 inputs (in this order) with quotation marks around them:

1) REQUIRED: Path to the anonymized proposal PDFs. This can also be the "redacted" PDFs with NSPIRES front-matter.
2) REQUIRED: Suffix of proposal PDFs (what comes *before* ".pdf" but *after* the proposal number)<br> e.g., for "23-XRP23_2-0003_Redacted.pdf" the suffix would be "_Redacted"
4) REQUIRED: Path to "Proposal Master" report from i-NSPIRES in CSV format (*not* Excel)
    
Example command line input: 
```
    python check_dapr.py "/Users/mansdell/Proposals/" "_Redacted" "/Users/mansdell/ProposalMasters/ProposalMaster.csv"
```

The code outputs its findings to the terminal as it checks each proposal. When all proposals are checked, the code will also output a final CSV file named "dapr_checks.csv" to your Desktop. The information includes:

* Page ranges for proposal sections
  - These assume the following order: STM, References, DMP, Relevance, Budget
  - They're usually correct, but sometimes they're not; this only really matters for searching for the PI name but avoiding the References section
  - The value -99 is reported if the page limits could not be found
  
* Median font size
  - The median font size used in the proposal is calculated, and a warning is given when <=11.8 pt (e.g., for checking compliance)
  - This is the same as for check_proposal.py

* Reference format
  - DAPR proposals are supposed to use bracketed number references
  - Reports number of brackets found in proposal and number of "et al." usages in proposal (the former number should be high, the latter low)
  
* Forbidden DAPR words
  - DAPR proposal shouldn't include references to previous work, institutions/departments/universities/cities, PI or Co-I names, etc.
  - Reports number of times such things are found and page numbers on which they are found

### group_proposals.py

This code reads in the NSPIRES-formatted PDF, attempts to find the 15-page proposal text, performs some basic Natural Language Processing (NLP) pre-processing of the text, identifies key words, then attempts to group the proposals according to topic. The outputs are described below (TBD).


# Disclaimer

This is not an official NASA product. 
