import os

pdf_path = r"c:\Users\poter\Downloads\Languages.pdf"
print("File exists:", os.path.exists(pdf_path), "Size:", os.path.getsize(pdf_path))

# Try pypdf / pypdf2 / fitz / pdfminer
for lib in ["pypdf", "PyPDF2", "fitz", "pdfplumber"]:
    try:
        mod = __import__(lib)
        print("Imported:", lib)
        if lib in ["pypdf", "PyPDF2"]:
            reader = mod.PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                print(f"--- PAGE {i+1} ---")
                print(page.extract_text())
        elif lib == "fitz":
            doc = mod.open(pdf_path)
            for i, page in enumerate(doc):
                print(f"--- PAGE {i+1} ---")
                print(page.get_text())
        break
    except ImportError:
        pass
    except Exception as e:
        print("Error with", lib, e)
