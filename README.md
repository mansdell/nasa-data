# Background

This directory contains some code to pull and analyze data from [NSPIRES](https://nspires.nasaprs.com/external/)-formatted proposals submitted to NASA [ROSES](https://science.nasa.gov/researchers/roses-blogs) calls.

Code author: Megan Ansdell [@mansdell](https://github.com/mansdell)

# Setup

### Required Non-standard Packages

[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/): a useful package for importing PDF text (which confusingly is imported as "import fitz")

[gender-guesser](https://pypi.org/project/gender-guesser/): only needed for one part of for check_proposals.py

# Description

### check_proposals.py

This code reads in an NSPIRES-formatted PDF submitted to a NASA ROSES call, attempts to find the "Scientific / Technical / Management" section (hereafter "the proposal"), and then grabs/checks a variety of useful things that are output into a csv file. These things are described below with tips on how to interpret the code output.

* PI name and proposal number
  - These are taken from the cover page of the NSPIRES-formatted PDF
  
* Font size (useful for checking compliance)
  - The median font size used in the proposal is calculated, and a warning is given when <=11.8 pt (e.g., for checking compliance)
  - A histogram of the font sizes (based on each PDF ["span"](https://pymupdf.readthedocs.io/en/latest/faq.html#how-to-analyze-font-characteristics)) can be saved to the specified output directory
  
* PhD Year (useful for identifying early career proposers)
  - The PhD year of the PI is extracted from the CV that is included within the PDF after the main proposal 
  - When extracted, the PhD year is correct in ~95% of cases (in some cases, no PhD year can be found)
  - Useful for identfy

* Demographic information
  - Guessed gender of the PI based on the first name
  - Zipcode of the PI 
  - Organizational code from NSPIRES 

### group_proposals.py

This code reads in the NSPIRES-formatted PDF, attempts to find the 15-page proposal text, performs some basic Natural Language Processing (NLP) pre-processing of the text, identifies key words, then attempts to group the proposals according to topic. The outputs are described below (TBD).


# Disclaimer

This is not an official NASA product. 
