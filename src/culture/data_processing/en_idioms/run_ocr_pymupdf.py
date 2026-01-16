import os
import pymupdf

culture_root = os.environ["CULTURE_ROOT"]

doc = pymupdf.open(culture_root + "/culture/data/en_idiom_oxford_dictionary.pdf") # open a document
out = open(culture_root + "/culture/data/ocr_outputs_pymupdf_en_idiom_oxford_dictionary.txt", "wb") # create a text output
for page in doc: # iterate the document pages
    text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
    out.write(text) # write text of page
    out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
out.close()