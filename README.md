# Background

This directory contains some code to pull and analyze data from proposals submitted to NASA ROSES calls.

Code author: Megan Ansdell [@mansdell](https://github.com/mansdell)

# Setup

### Required Non-standard Packages

[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/): a useful package for importing PDF text (which confusingly is imported as "import fitz")

# Outputs

### check_proposals.py

This code reads in the NSPIRES-formatted PDF, attempts to find the 15-page proposal text, and then checks a variety of useful things. These things are described below with tips on how to interpret the output (TBD)

### group_proposals.py

This code reads in the NSPIRES-formatted PDF, attempts to find the 15-page proposal text, performs some basic Natural Language Processing (NLP) pre-processing of the text, identifies key words, then attempts to group the proposals according to topic. The outputs are described below (TBD).


# Disclaimer

This is not an official NASA product. 
