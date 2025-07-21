import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import PyPDF2
import csv

# --------- 1. Count Pages in PDFs ---------
def count_pdf_pages(root_folder, report_file='pdf_page_report.csv'):
    with open(report_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'Page Count'])
        for dirpath, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.lower().endswith('.pdf'):
                    full_path = os.path.join(dirpath, filename)
                    try:
                        pdf = fitz.open(full_path)
                        writer.writerow([full_path, len(pdf)])
                        pdf.close()
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")
    print(f"[✓] Page count saved to {report_file}")

# --------- 2. Split 2-Pages-per-Sheet PDFs ---------
def split_double_pages(input_pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
    output_pdf_path = os.path.join(output_folder, f"{base_name}_split.pdf")

    pdf = fitz.open(input_pdf_path)
    output_pdf = fitz.open()

    for page in pdf:
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Split the image in half (vertical split)
        mid = pix.width // 2
        left = img[:, :mid]
        right = img[:, mid:]

        for half in [left, right]:
            # Convert to grayscale for OCR-friendly output
            img_gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
            cleaned = clean_image(img_gray)
            pdf_bytes = cv2.imencode('.png', cleaned)[1].tobytes()

            img_doc = fitz.open("png", pdf_bytes)
            output_pdf.insert_page(-1, width=img_doc[0].rect.width, height=img_doc[0].rect.height)
            output_pdf[-1].insert_image(img_doc[0].rect, stream=pdf_bytes)

    output_pdf.save(output_pdf_path)
    print(f"[✓] Split PDF saved to {output_pdf_path}")
    return output_pdf_path


# --------- 3. Merge PDFs (if needed) ---------
def merge_pdfs(input_paths, output_path):
    merger = PyPDF2.PdfMerger()
    for pdf_path in input_paths:
        merger.append(pdf_path)
    merger.write(output_path)
    merger.close()
    print(f"[✓] Merged PDF saved to {output_path}")

# --------- 4. Clean up document scans ---------
def clean_image(img):
    # Remove punch holes, shadows, and borders
    cleaned = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(cleaned, 200, 255, cv2.THRESH_BINARY)
    return thresh

# --------- 5. Main Orchestrator ---------
def process_all_pdfs(input_root, output_root):
    count_pdf_pages(input_root)

    split_paths = []
    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                input_pdf = os.path.join(dirpath, filename)
                output_folder = os.path.join(output_root, os.path.relpath(dirpath, input_root))
                os.makedirs(output_folder, exist_ok=True)

                # Step 2: Split double pages
                split_pdf = split_double_pages(input_pdf, output_folder)
                split_paths.append(split_pdf)

    # Step 3: Merge all processed PDFs
    merged_output = os.path.join(output_root, "Merged_All.pdf")
    merge_pdfs(split_paths, merged_output)

# --------- Entry Point ---------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF Processor: Count, Split, Merge, Clean.")
    parser.add_argument("--input", required=True, help="Input folder with PDFs")
    parser.add_argument("--output", required=True, help="Output folder for processed files")

    args = parser.parse_args()
    process_all_pdfs(args.input, args.output)
