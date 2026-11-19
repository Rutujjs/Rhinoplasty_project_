for mod in ["fitz", "reportlab", "fpdf", "pypdf"]:
    try:
        __import__(mod)
        print("Installed:", mod)
    except ImportError:
        print("Not installed:", mod)
