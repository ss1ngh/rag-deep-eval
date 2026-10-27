from pymupdf import open
import re
from typing import List
from pathlib import Path


def extract_paragraphs(pdf_path : str | Path) -> List[str]:
    doc = open(pdf_path)
    paragraphs = []


    for page in doc:

        #get blocks from the page
        blocks = page.get_text("blocks")


        #sort the blocks from top to bottom
        blocks = sorted(blocks, key=lambda b:(b[1], b[0]))

        '''
        returns
        [
            1,
            2.
            3.
            24,
            text
        ]
        '''
        for block in blocks:
            text = block[4]

            #replace line breaks inside the paragraph with space
            text = re.sub(r"\r+", " ", text).strip()


            if text:
                paragraphs.append(text)

    return paragraphs

# print (extract_paragraphs("../pdfs/penguins.pdf")) 
