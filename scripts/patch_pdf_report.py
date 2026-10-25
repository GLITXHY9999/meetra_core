import pymupdf

def patch_report(input_pdf, output_pdf):
    doc = pymupdf.open(input_pdf)
    print(f"Loaded {input_pdf} with {len(doc)} pages.")

    # 1. Update Certificate (Page 2, index 1)
    p2 = doc[1]
    # Redact "Academic Designation"
    # bbox: [133.6, 403.7, 251.0, 417.0]
    p2_rect = pymupdf.Rect(120, 400, 265, 420)
    p2.add_redact_annot(p2_rect, fill=(1, 1, 1))
    p2.apply_redactions()
    
    # Insert "Project Mentor" centered under SUPERVISOR
    # SUPERVISOR center is ~192.35
    text_pm = "Project Mentor"
    w_pm = pymupdf.get_text_length(text_pm, fontname="Times-Bold", fontsize=12.0)
    p2.insert_text((192.35 - w_pm / 2, 414.7), text_pm, fontname="Times-Bold", fontsize=12.0, color=(0, 0, 0))
    print("Page 2 (Certificate): 'Academic Designation' replaced with 'Project Mentor'.")

    # 2. Update Table of Contents (Page 3, index 2)
    p3 = doc[2]
    # We have 33 lines of numbers in TOC
    # Lines that need update:
    # Line 7 (index 6): 1.6 Organization: 10 -> 11 (Regular)
    # Line 8 (index 7): CH 2 Background: 11 -> 12 (Bold)
    # Line 9 (index 8): 2.1 Key Concepts: 11 -> 12 (Regular)
    # Line 10 (index 9): 2.2 Existing Solutions: 12 -> 13 (Regular)
    # Line 11 (index 10): 2.3 Summary and Link: 12 -> 13 (Regular)
    # Line 12 (index 11): CH 3 Methodology: 14 -> 15 (Bold)
    # Line 13 (index 12): 3.1 Dataset Description: 14 -> 15 (Regular)
    # Line 14 (index 13): 3.2 Preprocessing: 15 -> 16 (Regular)
    # Line 15 (index 14): 3.3 Model Selection: 16 -> 17 (Regular)
    # Line 16 (index 15): 3.4 System Flow: 16 -> 17 (Regular)
    # Line 17 (index 16): 3.5 UI Design: 17 -> 18 (Regular)
    # Line 18 (index 17): 3.6 Tools: 17 -> 18 (Regular)
    # Line 19 (index 18): CH 4 Results: 19 -> 20 (Bold)
    # Line 20 (index 19): 4.1 EDA: 19 -> 20 (Regular)
    # Line 21 (index 20): 4.2 Setup: 20 -> 21 (Regular)
    # Line 22 (index 21): 4.3 Performance: 20 -> 21 (Regular)
    # Line 23 (index 22): 4.4 Result Analysis: 21 -> 22 (Regular)
    # Line 24 (index 23): 4.5 UI Implementation: 23 -> 24 (Regular)
    # Line 25 (index 24): CH 5 Conclusion: 25 -> 26 (Bold)
    # Line 26 (index 25): 5.1 Conclusion: 25 -> 26 (Regular)
    # Line 27 (index 26): 5.2 Learning Outcomes: 25 -> 26 (Regular)
    # Line 28 (index 27): 5.3 Future Work: 26 -> 27 (Regular)
    # Line 29 (index 28): REFERENCES: 27 -> 28 (Bold)
    # Line 30 (index 29): APPENDIX: 28 -> 29 (Bold)
    # Line 31 (index 30): Appendix A: 28 -> 29 (Regular)
    # Line 32 (index 31): Appendix B: 29 -> 30 (Regular)
    # Line 33 (index 32): USER MANUAL: 30 -> 31 (Bold)

    p3_updates = [
        # (line_index, old_num, new_num, is_bold, origin_y)
        (6, "10", "11", False, 210.98),
        (7, "11", "12", True,  227.06),
        (8, "11", "12", False, 242.78),
        (9, "12", "13", False, 258.65),
        (10, "12", "13", False, 274.49),
        (11, "14", "15", True,  290.69),
        (12, "14", "15", False, 306.29),
        (13, "15", "16", False, 322.13),
        (14, "16", "17", False, 338.09),
        (15, "16", "17", False, 353.93),
        (16, "17", "18", False, 369.89),
        (17, "17", "18", False, 385.73),
        (18, "19", "20", True,  401.81),
        (19, "19", "20", False, 417.55),
        (20, "20", "21", False, 433.39),
        (21, "20", "21", False, 449.23),
        (22, "21", "22", False, 465.19),
        (23, "23", "24", False, 481.03),
        (24, "25", "26", True,  497.23),
        (25, "25", "26", False, 512.83),
        (26, "25", "26", False, 528.67),
        (27, "26", "27", False, 544.63),
        (28, "27", "28", True,  560.71),
        (29, "28", "29", True,  576.55),
        (30, "28", "29", False, 592.30),
        (31, "29", "30", False, 608.14),
        (32, "30", "31", True,  624.34),
    ]

    for idx, old_num, new_num, is_bold, orig_y in p3_updates:
        # Redact old number at x in [518, 534], y in [orig_y - 11, orig_y + 3]
        rect = pymupdf.Rect(518, orig_y - 10.5, 534, orig_y + 2.5)
        p3.add_redact_annot(rect, fill=(1, 1, 1))
    p3.apply_redactions()

    for idx, old_num, new_num, is_bold, orig_y in p3_updates:
        fontname = "Times-Bold" if is_bold else "Times-Roman"
        w = pymupdf.get_text_length(new_num, fontname=fontname, fontsize=10.6)
        x = 531.1 - w
        p3.insert_text((x, orig_y), new_num, fontname=fontname, fontsize=10.6, color=(0, 0, 0))
    print("Page 3 (TOC): 27 pagination entries synchronized.")

    # 3. Update Page 4 (List of Figures & Tables)
    p4 = doc[3]
    # Figures:
    # Figure 3.1: 16 -> 17 (orig_y = 140.30)
    # Figure 3.2: 17 -> 18 (orig_y = 164.06)
    # Figure 4.1: 19 -> 20 (orig_y = 187.70)
    # Figure 4.2: 21 -> 23 (orig_y = 211.46)
    # Figure 4.3: 23 -> 24 (orig_y = 235.22)
    # Tables:
    # Table 2.1: 12 -> 13 (orig_y = 420.67)
    # Table 3.1: 14 -> 15 (orig_y = 444.31)
    # Table 3.2: 17 -> 18 (orig_y = 468.07)
    # Table 4.1: 20 -> 21 (orig_y = 491.83)
    # Table 4.2: 24 -> 25 (orig_y = 515.47)
    # Table B.1: 29 -> 30 (orig_y = 539.23)

    p4_updates = [
        ("Figure 3.1", "16", "17", 140.30),
        ("Figure 3.2", "17", "18", 164.06),
        ("Figure 4.1", "19", "20", 187.70),
        ("Figure 4.2", "21", "23", 211.46),
        ("Figure 4.3", "23", "24", 235.22),
        ("Table 2.1",  "12", "13", 420.67),
        ("Table 3.1",  "14", "15", 444.31),
        ("Table 3.2",  "17", "18", 468.07),
        ("Table 4.1",  "20", "21", 491.83),
        ("Table 4.2",  "24", "25", 515.47),
        ("Table B.1",  "29", "30", 539.23),
    ]

    for label, old_num, new_num, orig_y in p4_updates:
        rect = pymupdf.Rect(518, orig_y - 10.0, 534, orig_y + 2.5)
        p4.add_redact_annot(rect, fill=(1, 1, 1))
    p4.apply_redactions()

    for label, old_num, new_num, orig_y in p4_updates:
        w = pymupdf.get_text_length(new_num, fontname="Times-Roman", fontsize=10.0)
        x = 530.56 - w
        p4.insert_text((x, orig_y), new_num, fontname="Times-Roman", fontsize=10.0, color=(0, 0, 0))
    print("Page 4 (List of Figures & Tables): 11 pagination entries synchronized.")

    # 4. Insert professional footer page numbers on all pages
    # Preliminary pages (2 to 6): Roman numerals ii, iii, iv, v, vi
    roman_map = {2: "ii", 3: "iii", 4: "iv", 5: "v", 6: "vi"}
    for p_num in range(2, 7):
        p = doc[p_num - 1]
        rom = roman_map[p_num]
        w = pymupdf.get_text_length(rom, fontname="Times-Roman", fontsize=10.0)
        p.insert_text((306.0 - w / 2, 750.0), rom, fontname="Times-Roman", fontsize=10.0, color=(0, 0, 0))
    
    # Body pages (7 to 32): Arabic numerals 7 to 32
    for p_num in range(7, len(doc) + 1):
        p = doc[p_num - 1]
        page_str = str(p_num)
        w = pymupdf.get_text_length(page_str, fontname="Times-Roman", fontsize=10.0)
        p.insert_text((306.0 - w / 2, 750.0), page_str, fontname="Times-Roman", fontsize=10.0, color=(0, 0, 0))
    print(f"Added page numbers to footers across pages 2 to {len(doc)}.")

    # Save output PDF
    doc.save(output_pdf, garbage=4, deflate=True)
    print(f"Saved patched PDF to {output_pdf}")

if __name__ == '__main__':
    patch_report('Kaizen_Final_Project_Report.pdf', 'scratch/Kaizen_Report_Fixed_Test.pdf')
