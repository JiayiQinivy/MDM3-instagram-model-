#!/usr/bin/env python3
"""
Script to populate the MDM3 Instagram Model template with actual project content.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET
import re

# Constants
WORKSPACE = Path(
    "/Users/salahbaaziz/Library/CloudStorage/OneDrive-SharedLibraries-Onedrive-UniversityofBristol/Engineering Math/Year 3/MDM3/TB3/Phase C/MDM3-instagram-model-/Salah"
)
TEMPLATE_PATH = WORKSPACE / "template.pptx"
OUTPUT_PATH = WORKSPACE / "MDM3_Instagram_Model_Presentation.pptx"
RF_OUTPUTS = WORKSPACE / "random_forest_outputs"

# Namespaces
NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

# Register namespaces
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)
ET.register_namespace("", "http://schemas.openxmlformats.org/presentationml/2006/main")

# Slide content definitions
SLIDE_CONTENT = {
    1: {  # Title slide
        "title": "Nonlinear Threshold Effects of Social Media Usage on Psychological Risk",
        "subtitle": "MDM3 Machine Learning Project",
        "description": "An Analysis Using Random Forest, Linear GAM, and Logistic GAM Models",
    },
    3: {  # Table of Contents
        "sections": [
            ("01", "Introduction & Objectives", "Research goals and hypotheses"),
            ("02", "Literature Review", "Social media and mental health research"),
            ("03", "Methodology", "Dataset, features, and modeling approach"),
            ("04", "Analysis & Results", "Feature selection and model performance"),
            ("05", "Discussion & Conclusions", "Key findings and implications"),
        ]
    },
    4: {  # Statement slide
        "section": "01",
        "title": "Research Statement",
        "content": "This research investigates the nonlinear relationships between Instagram usage patterns and psychological risk factors, examining how different usage behaviors may threshold at certain levels to produce disproportionate effects on mental well-being.",
    },
    5: {  # Purpose Statement
        "objectives": [
            (
                "Objective 1",
                "Identify key Instagram usage features that most strongly predict psychological risk",
            ),
            (
                "Objective 2",
                "Detect nonlinear threshold effects in the relationship between social media usage and stress",
            ),
            (
                "Objective 3",
                "Compare performance of Random Forest, Linear GAM, and Logistic GAM models",
            ),
            (
                "Objective 4",
                "Rank features using ensemble methods including Boruta, SHAP, and permutation importance",
            ),
        ]
    },
    6: {  # Dataset Overview
        "big_number": "1,547,896",
        "label": "Total Samples in Dataset",
        "description": "Instagram usage and lifestyle data with 54 features including demographics, behavioral metrics, and health indicators",
    },
    11: {  # Literature Review
        "title": "Literature Review",
        "references": [
            "Raudenbush & Bryk (2002). Hierarchical Linear Models",
            "Hastie et al. (2009). The Elements of Statistical Learning",
            "Breiman (2001). Random Forests. Machine Learning",
            "SHAP Documentation. TreeExplainer for model interpretability",
            "Boruta: A wrapper algorithm for feature selection",
        ],
    },
    12: {  # Theoretical Framework
        "title": "Theoretical Framework",
        "framework": {
            "Target Variable": "psychological_risk = z(stress) - z(happiness)",
            "Key Features": "17 Instagram usage metrics + 12 lifestyle factors",
            "Model Types": "Regression (stress prediction) + Classification (risk threshold)",
        },
    },
    13: {  # Schedule
        "phases": [
            ("Phase 1", "Data Preprocessing & Visualization"),
            ("Phase 2", "Feature Selection (12 methods)"),
            ("Phase 3", "Model Training & Evaluation"),
            ("Phase 4", "Threshold Analysis & Conclusions"),
        ]
    },
    14: {  # Methodology
        "methodology": [
            ("Dataset", "1.5M+ samples, 54 features"),
            ("Target", "Psychological risk score"),
            ("Methods", "Random Forest, Linear GAM, Logistic GAM"),
            (
                "Feature Selection",
                "12 methods: MI, Spearman, VIF, ElasticNet, ExtraTrees, LGB, SHAP, RFECV, Boruta, Stability, Permutation",
            ),
        ]
    },
    17: {  # Analysis Results - Feature Importance
        "title": "Feature Importance Analysis",
        "top_features": [
            ("daily_active_minutes_instagram", "22.3%"),
            ("likes_given_per_day", "16.2%"),
            ("stories_viewed_per_day", "10.7%"),
            ("time_on_feed_per_day", "10.6%"),
            ("time_on_reels_per_day", "7.6%"),
        ],
    },
    18: {  # Discussion
        "discussions": [
            (
                "Finding 1",
                "Daily active minutes is the strongest predictor of psychological risk",
            ),
            (
                "Finding 2",
                "Behavioral engagement metrics (likes, comments) show high predictive power",
            ),
            ("Finding 3", "Time on Reels shows distinct nonlinear threshold patterns"),
        ]
    },
    19: {  # Conclusions
        "conclusions": [
            "Random Forest achieves R² = 0.734 for regression, AUC = 0.924 for classification",
            "Feature selection confirms core behavioral drivers: active time, engagement metrics",
            "Nonlinear threshold effects detected in passive consumption behaviors",
            "Age moderates the relationship between usage patterns and psychological outcomes",
        ]
    },
    20: {  # Thank You
        "email": "salah.baaziz@bristol.ac.uk",
        "contact": "University of Bristol",
    },
}


def create_populated_pptx():
    """Main function to create the populated presentation."""

    # Copy template to output
    print(f"Copying template to {OUTPUT_PATH}...")
    shutil.copy2(TEMPLATE_PATH, OUTPUT_PATH)

    # Extract the PPTX
    extract_dir = Path("/tmp/pptx_work")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(OUTPUT_PATH, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Process each slide
    slides_dir = extract_dir / "ppt" / "slides"

    # Slide 1 - Title
    modify_title_slide(slides_dir / "slide1.xml")

    # Slide 3 - Table of Contents
    modify_toc_slide(slides_dir / "slide3.xml")

    # Slide 4 - Statement
    modify_statement_slide(slides_dir / "slide4.xml")

    # Slide 5 - Objectives
    modify_objectives_slide(slides_dir / "slide5.xml")

    # Slide 6 - Dataset Overview
    modify_dataset_slide(slides_dir / "slide6.xml")

    # Slide 11 - Literature Review
    modify_literature_slide(slides_dir / "slide11.xml")

    # Slide 12 - Theoretical Framework
    modify_framework_slide(slides_dir / "slide12.xml")

    # Slide 14 - Methodology
    modify_methodology_slide(slides_dir / "slide14.xml")

    # Slide 17 - Feature Importance Results
    modify_feature_importance_slide(slides_dir / "slide17.xml")

    # Slide 18 - Discussion
    modify_discussion_slide(slides_dir / "slide18.xml")

    # Slide 19 - Conclusions
    modify_conclusions_slide(slides_dir / "slide19.xml")

    # Slide 20 - Thank You
    modify_thankyou_slide(slides_dir / "slide20.xml")

    # Copy images to media folder
    copy_images_to_pptx(extract_dir)

    # Repackage PPTX
    repackage_pptx(extract_dir, OUTPUT_PATH)

    print(f"\nPresentation created successfully: {OUTPUT_PATH}")
    return str(OUTPUT_PATH)


def modify_text_element(element, text):
    """Modify text in a shape."""
    for t_elem in element.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
    ):
        if t_elem.text:
            t_elem.text = text
            break


def find_text_element(element, search_text):
    """Find element containing specific text."""
    for t_elem in element.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
    ):
        if t_elem.text and search_text.lower() in t_elem.text.lower():
            return element
    return None


def modify_title_slide(slide_path):
    """Modify the title slide."""
    print("Modifying title slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    # Find and modify text elements
    texts_found = []
    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text:
                texts_found.append(t.text.strip()[:50])

                # Replace placeholder text
                if "Thesis Defense" in t.text:
                    t.text = "Nonlinear Threshold Effects"
                elif "Doctor Of Philosophy" in t.text:
                    t.text = "MDM3 Machine Learning Project"
                elif "Here is where" in t.text:
                    t.text = "Instagram Usage & Psychological Risk Analysis\nUsing Random Forest, Linear GAM & Logistic GAM Models"

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated title slide")


def modify_toc_slide(slide_path):
    """Modify the table of contents slide."""
    print("Modifying table of contents...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    # Replace section names
    replacements = {
        "You can describe the topic of the section here": "",
        "Statement": "Introduction & Objectives",
        "Methodology": "Literature Review",
        "Analysis": "Dataset & Features",
        "Hypothesis": "Modeling Approach",
        "Conclusions": "Results & Discussion",
        "Objectives": "Feature Selection",
    }

    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text:
                for old, new in replacements.items():
                    if old in t.text:
                        t.text = t.text.replace(old, new)
                        break

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated table of contents")


def modify_statement_slide(slide_path):
    """Modify the statement slide."""
    print("Modifying statement slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    content = """This research investigates the nonlinear relationships between Instagram usage patterns and psychological risk factors, examining how different usage behaviors may threshold at certain levels to produce disproportionate effects on mental well-being.

Key Research Questions:
• Which Instagram usage behaviors most strongly predict psychological risk?
• Do threshold effects exist where usage crosses certain levels?
• How do lifestyle factors moderate these relationships?"""

    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text and "You can describe" in t.text:
                t.text = content

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated statement slide")


def modify_objectives_slide(slide_path):
    """Modify the objectives slide."""
    print("Modifying objectives slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    objectives = [
        (
            "Objective 1",
            "Identify key Instagram usage features that most strongly predict psychological risk",
        ),
        (
            "Objective 2",
            "Detect nonlinear threshold effects in social media usage and stress relationship",
        ),
        (
            "Objective 3",
            "Compare Random Forest, Linear GAM, and Logistic GAM model performance",
        ),
        (
            "Objective 4",
            "Rank features using ensemble methods: Boruta, SHAP, and permutation importance",
        ),
    ]

    # Replace placeholder content
    count = 0
    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text:
                if "What about Mercury" in t.text and count < len(objectives):
                    obj = objectives[count]
                    t.text = f"{obj[0]}: {obj[1]}"
                    count += 1
                elif "Mercury is the closest" in t.text and count < len(objectives):
                    t.text = objectives[count % len(objectives)][1]
                    count += 1

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated objectives slide")


def modify_dataset_slide(slide_path):
    """Modify the dataset overview slide."""
    print("Modifying dataset overview slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text:
                if "98,300,000" in t.text or "Big numbers" in t.text:
                    t.text = "1,547,896"
                elif "catch your audience" in t.text:
                    t.text = "Samples analyzed with comprehensive Instagram usage and lifestyle metrics"

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated dataset slide")


def modify_literature_slide(slide_path):
    """Modify the literature review slide."""
    print("Modifying literature review slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    references = [
        "Raudenbush & Bryk (2002). Hierarchical Linear Models. SAGE Publications.",
        "Hastie et al. (2009). The Elements of Statistical Learning. Springer.",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
        "Lundberg et al. (2020). Local Interpretable Model-agnostic Explanations (LIME).",
        "Strobl et al. (2007). Bias in random forest variable importance measures.",
    ]

    count = 0
    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text and ("AUTHOR." in t.text or "Mercury is small" in t.text):
                if count < len(references):
                    t.text = references[count]
                    count += 1

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated literature review slide")


def modify_framework_slide(slide_path):
    """Modify the theoretical framework slide."""
    print("Modifying theoretical framework slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    framework_items = [
        "Target: psychological_risk = z(stress) - z(happiness)",
        "Features: 17 Instagram metrics + 12 lifestyle factors",
        "Models: Regression + Binary Classification",
        "Evaluation: RMSE, R², AUC, F1 Score",
    ]

    count = 0
    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text and ("Mercury is small" in t.text or "Theory" in t.text):
                if count < len(framework_items):
                    t.text = framework_items[count]
                    count += 1

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated framework slide")


def modify_methodology_slide(slide_path):
    """Modify the methodology slide."""
    print("Modifying methodology slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    methods = [
        "Dataset: 1,547,896 samples, 54 features",
        "Preprocessing: Label encoding, median imputation",
        "Feature Selection: 12 methods (MI, Spearman, VIF, ElasticNet, ExtraTrees, LGB, SHAP, RFECV, Boruta, Stability, Permutation)",
        "Models: Random Forest, Linear GAM, Logistic GAM",
        "Validation: Train/test split, cross-validation",
    ]

    count = 0
    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text and "Mercury is the closest" in t.text:
                if count < len(methods):
                    t.text = methods[count]
                    count += 1

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated methodology slide")


def modify_feature_importance_slide(slide_path):
    """Modify the feature importance results slide."""
    print("Modifying feature importance slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    features = [
        ("daily_active_minutes_instagram", "22.3%"),
        ("likes_given_per_day", "16.2%"),
        ("stories_viewed_per_day", "10.7%"),
        ("time_on_feed_per_day", "10.6%"),
        ("time_on_reels_per_day", "7.6%"),
        ("comments_written_per_day", "7.5%"),
        ("passive_consumption_index", "4.9%"),
    ]

    # Update text content
    count = 0
    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text:
                if "Mercury" in t.text and count < len(features):
                    feat = features[count]
                    t.text = f"{feat[0]}: {feat[1]}"
                    count += 1
                elif "50%" in t.text:
                    t.text = "Top 7 features account for ~80% of importance"

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated feature importance slide")


def modify_discussion_slide(slide_path):
    """Modify the discussion slide."""
    print("Modifying discussion slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    discussions = [
        (
            "Finding 1",
            "Daily active minutes is the strongest predictor of psychological risk",
        ),
        (
            "Finding 2",
            "Behavioral engagement (likes, comments) shows high predictive power",
        ),
        ("Finding 3", "Time on Reels shows distinct nonlinear threshold patterns"),
    ]

    count = 0
    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text:
                if "Discussion" in t.text and count < len(discussions):
                    t.text = f"{discussions[count][0]}: {discussions[count][1]}"
                    count += 1
                elif "Ceres is located" in t.text:
                    t.text = "Nonlinear effects detected at specific usage thresholds"

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated discussion slide")


def modify_conclusions_slide(slide_path):
    """Modify the conclusions slide."""
    print("Modifying conclusions slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    conclusions = [
        "Random Forest: R² = 0.734 (regression), AUC = 0.924 (classification)",
        "Feature selection confirms core behavioral drivers",
        "Nonlinear threshold effects detected in passive consumption",
        "Age moderates usage-outcome relationships",
    ]

    count = 0
    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text and ("Mars is full" in t.text or "Despite being red" in t.text):
                if count < len(conclusions):
                    t.text = conclusions[count]
                    count += 1

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated conclusions slide")


def modify_thankyou_slide(slide_path):
    """Modify the thank you slide."""
    print("Modifying thank you slide...")
    tree = ET.parse(slide_path)
    root = tree.getroot()

    for sp in root.findall(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}sp"
    ):
        for t in sp.findall(
            ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
        ):
            if t.text:
                if "youremail@freepik.com" in t.text:
                    t.text = "salah.baaziz@bristol.ac.uk"
                elif "yourwebsite.com" in t.text:
                    t.text = "University of Bristol"
                elif "+34 654 321 432" in t.text:
                    t.text = "MDM3 Machine Learning Project"

    tree.write(slide_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Updated thank you slide")


def copy_images_to_pptx(extract_dir):
    """Copy images from random_forest_outputs to the PPTX media folder."""
    print("\nCopying images to presentation...")

    media_dir = extract_dir / "ppt" / "media"
    media_dir.mkdir(exist_ok=True)

    # Copy available images
    if RF_OUTPUTS.exists():
        for img_file in RF_OUTPUTS.glob("*.png"):
            dest = media_dir / img_file.name
            shutil.copy2(img_file, dest)
            print(f"  Copied: {img_file.name}")

    # Update Content_Types.xml if needed
    content_types_path = extract_dir / "[Content_Types].xml"
    if content_types_path.exists():
        tree = ET.parse(content_types_path)
        root = tree.getroot()

        # Check if png extension is registered
        ct_ns = "{http://schemas.openxmlformats.org/package/2006/content-types}"
        has_png = False
        for override in root.findall(f"{ct_ns}Override"):
            if "png" in override.get("Extension", ""):
                has_png = True
                break

        if not has_png:
            # Add PNG content type
            default = ET.SubElement(root, f"{ct_ns}Default")
            default.set("Extension", "png")
            default.set("ContentType", "image/png")

        tree.write(content_types_path, xml_declaration=True, encoding="UTF-8")


def repackage_pptx(extract_dir, output_path):
    """Repackage the extracted files back into a PPTX."""
    print(f"\nRepackaging presentation to {output_path}...")

    # Remove old file
    if output_path.exists():
        os.remove(output_path)

    # Collect all files first
    all_files = []
    for root_dir, dirs, files in os.walk(extract_dir):
        for file in files:
            file_path = os.path.join(root_dir, file)
            arcname = os.path.relpath(file_path, extract_dir)
            all_files.append((file_path, arcname))

    print(f"  Found {len(all_files)} files to package")

    # Create new PPTX
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path, arcname in all_files:
            try:
                zipf.write(file_path, arcname)
            except Exception as e:
                print(f"  Warning: Could not add {arcname}: {e}")

    print("Repackaging complete!")


if __name__ == "__main__":
    try:
        result = create_populated_pptx()
        print(f"\n✅ Success! Your presentation has been created at:")
        print(f"   {result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
