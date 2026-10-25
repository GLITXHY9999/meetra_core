"""Audit Kaizen_Final_Project_Report.pdf for errors, mismatches, and formatting issues."""
import pypdf

reader = pypdf.PdfReader("Kaizen_Final_Project_Report.pdf")
pages = [p.extract_text() or "" for p in reader.pages]

print(f"Total pages: {len(pages)}")

toc_entries = [
    ("CHAPTER 1: INTRODUCTION", 7),
    ("1.1 Identification of Client", 7),
    ("1.2 Problem Statement", 8),
    ("1.3 Objectives", 8),
    ("1.4 Timeline", 9),
    ("1.5 Team Work Division", 9),
    ("1.6 Organization of the Report", 11),
    ("CHAPTER 2: BACKGROUND STUDY", 12),
    ("2.1 Key Concepts Used", 12),
    ("2.2 Existing Solutions and Similar Work", 13),
    ("2.3 Summary and Link to This Project", 13),
    ("CHAPTER 3: METHODOLOGY", 15),
    ("3.1 Dataset Description", 15),
    ("3.2 Data Preprocessing", 16),
    ("3.3 Model Selection", 17),
    ("3.4 System Flow Diagram", 17),
    ("3.5 User Interface Design", 18),
    ("3.6 Tools and Technologies", 18),
    ("CHAPTER 4: RESULTS ANALYSIS AND VALIDATION", 20),
    ("4.1 Exploratory Data Analysis (EDA)", 20),
    ("4.2 Experimental Setup", 21),
    ("4.3 Model Performance", 21),
    ("4.4 Result Analysis", 22),
    ("4.5 User Interface Implementation and Testing", 24),
    ("CHAPTER 5: CONCLUSION AND FUTURE WORK", 26),
    ("5.1 Conclusion", 26),
    ("5.2 Learning Outcomes", 26),
    ("5.3 Future Work", 27),
    ("REFERENCES", 28),
    ("APPENDIX", 29),
    ("USER MANUAL", 31)
]

print("\n=== TABLE OF CONTENTS ACCURACY AUDIT ===")
print(f"{'Section / Heading':<50} | {'TOC Page':<10} | {'Actual Page':<12} | Status")
print("-" * 85)

all_match = True
for title, expected_page in toc_entries:
    # check if the heading exists on expected page
    content = pages[expected_page - 1]
    # check prefix or clean match
    check_str = title.split(" (")[0].strip()
    if check_str.lower() in content.lower():
        status = "MATCH (VERIFIED)"
        act_str = str(expected_page)
    else:
        # scan for actual
        actuals = [i + 1 for i, p in enumerate(pages[4:], start=5) if check_str.lower() in p.lower()]
        act_str = str(actuals[0]) if actuals else "NOT FOUND"
        status = f"MISMATCH ({actuals[0] - expected_page:+d})" if actuals else "NOT FOUND"
        all_match = False
    print(f"{title:<50} | {expected_page:<10} | {act_str:<12} | {status}")

print("\n=== LIST OF FIGURES AUDIT ===")
figures = [
    ("Figure 3.1", 17),
    ("Figure 3.2", 18),
    ("Figure 4.1", 20),
    ("Figure 4.2", 23),
    ("Figure 4.3", 24)
]
for fig, expected in figures:
    content = pages[expected - 1]
    stat = "MATCH (VERIFIED)" if fig.lower() in content.lower() else "MISMATCH"
    if stat != "MATCH (VERIFIED)":
        all_match = False
    print(f"{fig:<15} | Listed: {expected:<4} | Actual: {expected if stat == 'MATCH (VERIFIED)' else 'ERROR':<4} | {stat}")

print("\n=== LIST OF TABLES AUDIT ===")
tables = [
    ("Table 1.1", 7),
    ("Table 1.2", 9),
    ("Table 1.3", 9),
    ("Table 2.1", 13),
    ("Table 3.1", 15),
    ("Table 3.2", 18),
    ("Table 4.1", 21),
    ("Table 4.2", 25),
    ("Table B.1", 30)
]
for tab, expected in tables:
    content = pages[expected - 1]
    stat = "MATCH (VERIFIED)" if tab.lower() in content.lower() else "MISMATCH"
    if stat != "MATCH (VERIFIED)":
        all_match = False
    print(f"{tab:<15} | Listed: {expected:<4} | Actual: {expected if stat == 'MATCH (VERIFIED)' else 'ERROR':<4} | {stat}")

print(f"\nOVERALL AUDIT PASSED: {all_match}")
