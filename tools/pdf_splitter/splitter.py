import os
import re
from pypdf import PdfReader, PdfWriter


def sanitize_filename(name):
    """
    Removes characters that are invalid for filenames and replaces spaces.
    """
    # Remove invalid characters
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    return name


def split_pdf_by_chapters(source_pdf_path, output_directory):
    """
    Splits a PDF into multiple files based on a predefined chapter structure.

    Args:
        source_pdf_path (str): The path to the source PDF file.
        output_directory (str): The folder where the split PDFs will be saved.
    """
    # --- Chapter data based on the provided PDF structure ---
    # Each dictionary contains the title and the 1-based page range.
    chapters = [
        {"number": 1, "title": "Introduction", "start_page": 22, "end_page": 43},
        {
            "number": 2,
            "title": "Information Security Risk Assessment Basics",
            "start_page": 44,
            "end_page": 58,
        },
        {"number": 3, "title": "Project Definition", "start_page": 60, "end_page": 93},
        {
            "number": 4,
            "title": "Security Risk Assessment Preparation",
            "start_page": 94,
            "end_page": 131,
        },
        {"number": 5, "title": "Data Gathering", "start_page": 132, "end_page": 164},
        {
            "number": 6,
            "title": "Administrative Data Gathering",
            "start_page": 166,
            "end_page": 235,
        },
        {
            "number": 7,
            "title": "Technical Data Gathering",
            "start_page": 236,
            "end_page": 307,
        },
        {
            "number": 8,
            "title": "Physical Data Gathering",
            "start_page": 308,
            "end_page": 384,
        },
        {
            "number": 9,
            "title": "Security Risk Analysis",
            "start_page": 386,
            "end_page": 400,
        },
        {
            "number": 10,
            "title": "Security Risk Mitigation",
            "start_page": 402,
            "end_page": 414,
        },
        {
            "number": 11,
            "title": "Security Risk Assessment Reporting",
            "start_page": 416,
            "end_page": 428,
        },
        {
            "number": 12,
            "title": "Security Risk Assessment Project Management",
            "start_page": 430,
            "end_page": 454,
        },
        {
            "number": 13,
            "title": "Security Risk Assessment Approaches",
            "start_page": 456,
            "end_page": 474,
        },
    ]

    # --- Script Logic ---

    # Check if the source PDF file exists
    if not os.path.exists(source_pdf_path):
        print(f"Error: The file '{source_pdf_path}' was not found.")
        print(
            "Please make sure the script is in the same directory as the PDF, or update the 'source_pdf' variable."
        )
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: '{output_directory}'")

    # Open the source PDF file
    with open(source_pdf_path, "rb") as infile:
        reader = PdfReader(infile)
        total_pages = len(reader.pages)
        print(f"Source PDF '{source_pdf_path}' opened. Total pages: {total_pages}\n")

        # Process each chapter
        for chapter in chapters:
            writer = PdfWriter()
            chapter_num = chapter["number"]
            chapter_title = chapter["title"]
            start_page = chapter["start_page"]
            end_page = chapter["end_page"]

            print(
                f"Processing Chapter {chapter_num}: '{chapter_title}' (Pages {start_page}-{end_page})..."
            )

            # Page numbers in PyPDF2 are 0-indexed, so we subtract 1
            start_index = start_page - 1
            # The range function's end is exclusive, so the end_page is correct
            end_index = end_page

            if start_index >= total_pages:
                print(
                    f"  -> Warning: Start page {start_page} is out of bounds. Skipping chapter."
                )
                continue

            # Add the pages for the current chapter to the writer object
            for page_num in range(start_index, end_index):
                if page_num < total_pages:
                    writer.add_page(reader.pages[page_num])
                else:
                    # This handles cases where the specified end_page is beyond the PDF's length
                    print(
                        f"  -> Warning: Page {page_num + 1} not found in PDF. Stopping at the last page."
                    )
                    break

            # If pages were added, save them to a new PDF file
            if len(writer.pages) > 0:
                sanitized_title = sanitize_filename(chapter_title)
                output_filename = os.path.join(
                    output_directory, f"Chapter_{chapter_num}_{sanitized_title}.pdf"
                )

                with open(output_filename, "wb") as outfile:
                    writer.write(outfile)
                print(f"  -> Saved '{output_filename}' successfully.")
            else:
                print(
                    f"  -> No pages were found for Chapter {chapter_num}. PDF not created."
                )


# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    # Change these variables to match your file and desired output folder
    source_pdf = "../../raw/The Security Risk Assessment Handbook_ A Complete Guide for Performing Security Risk Assessments, Second Edition.pdf"
    output_folder = "Split_Chapters"

    # Run the splitting function
    split_pdf_by_chapters(source_pdf, output_folder)

    print("\nPDF splitting process complete!")
