import ollama
import uuid
from datetime import date
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── DB setup ────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ── Detection inputs (PLEASE REPLACE WITH MODEL OUTPUTS!!!!!) ──────────────────────
priority          = "YELLOW"
patient_age_group = "child"

# --- Malocclusion (from OMNI model) ---
detected_labels = ["TM"]                        # labels detected by OMNI model
image_views     = ["frontal", "maxillary"]      # views used

# --- Gingivitis (from MGI model) ---
mgi_scores = {
    "maxilla": {
        "overall":  1,            # MGI score for upper jaw (0-4)
        "anterior": "mild",       # front teeth region
        "premolar": "healthy",    # premolar region
        "molar":    "healthy",    # molar region
    },
    "mandible": {
        "overall":  0,            # MGI score for lower jaw (0-4)
        "anterior": "healthy",
        "premolar": "healthy",
        "molar":    "healthy",
    },
}
ohi_provided = "Yes"              # were oral hygiene instructions given?

mgi_scores = {                         # from gingivitis model
    "maxilla":  {"overall": 1, "anterior": "mild", "premolar": "healthy", "molar": "healthy"},
    "mandible": {"overall": 0, "anterior": "healthy", "premolar": "healthy", "molar": "healthy"},
}


# ── Label maps ───────────────────────────────────────────────────────────────
label_map = {
    "TT": "Tooth Torsion",        "DO": "Deep Overjet",
    "TM": "Tooth Misalignment",   "MR": "Mandibular Retrusion",
    "IOA": "Invisible Orthodontic Attachment",
    "CFOA": "Cast Fixed Orthodontic Appliances",
    "OB": "Orthodontic Brace",    "FOD": "Fixed Orthodontic Device",
    "HT": "Healthy Teeth",        "TE": "Tooth Emergence",
}
mgi_text_map = {
    0: "Healthy gingiva",           1: "Mild localised inflammation",
    2: "Mild generalised inflammation", 3: "Moderate inflammation",
    4: "Severe inflammation",
}
condition_labels  = {"TT", "DO", "TM", "MR"}
appliance_labels  = {"IOA", "CFOA", "OB", "FOD"}
normal_labels     = {"HT", "TE"}

# ── Referral logic ───────────────────────────────────────────────────────────
overall_mgi = max(mgi_scores["maxilla"]["overall"], mgi_scores["mandible"]["overall"])

has_condition = any(l in condition_labels for l in detected_labels)
has_appliance = any(l in appliance_labels for l in detected_labels)

if has_condition:
    mal_refer    = "Yes"
    refer_to     = "Orthodontist" + (" + OMFS" if "MR" in detected_labels else "")
    if overall_mgi >= 3:
        refer_to += " + General Dentist / Periodontist"
    mal_urgency  = "Urgent" if "DO" in detected_labels else "Within 4 weeks"
elif has_appliance and has_condition:
    mal_refer    = "Yes"
    refer_to     = "Notify treating orthodontist"
    mal_urgency  = "Routine"
else:
    mal_refer    = "No"
    refer_to     = "—"
    mal_urgency  = "—"

if overall_mgi == 0:   gingi_refer = "No referral (MGI 0–1)"
elif overall_mgi == 1: gingi_refer = "No referral (MGI 0–1) — OHI advised"
elif overall_mgi == 2: gingi_refer = "Routine dental review within 4–6 weeks (MGI 2)"
elif overall_mgi == 3: gingi_refer = "Refer within 2–4 weeks — scaling indicated (MGI 3)"
else:                  gingi_refer = "URGENT referral within 24–72 hours (MGI 4)"

# ── Helper: checkbox line ────────────────────────────────────────────────────
def box(label, detected_labels):
    return "[X]" if label in detected_labels else "[ ]"

# ── Pre-fill the template ─────────────────────────────────────────────────────
def build_partial_note():
    lines = []
    lines.append("NDCS DENTAL SCREENING — AI-ASSISTED REFERRAL NOTE")
    lines.append("-" * 50)
    lines.append(f"Date of Screening : {date.today().isoformat()}")
    lines.append(f"Screening ID      : {uuid.uuid4().hex[:8].upper()}")
    lines.append(f"Triage Priority   : {priority}")
    lines.append(f"Patient Age Group : {patient_age_group}")
    lines.append("")

    lines.append("--- MALOCCLUSION FINDINGS ---")
    lines.append("Model Used   : OMNI-based Malocclusion Detector")
    lines.append(f"Image Views  : {', '.join(image_views)}")
    lines.append("")
    lines.append("Detected Conditions:")

    # TT
    tt = box("TT", detected_labels)
    lines.append(f"  {tt} TT — Tooth Torsion")
    if "TT" in detected_labels:
        lines.append( "        Location : [FIELD: jaw quadrant(s) and tooth position(s)]")
        lines.append( "        Severity : [FIELD: single tooth / multiple teeth / severe rotation]")

    # DO
    do_ = box("DO", detected_labels)
    lines.append(f"  {do_} DO — Deep Overjet")
    if "DO" in detected_labels:
        lines.append( "        Location : Anterior region (maxilla + mandible)")
        lines.append( "        Severity : [FIELD: moderate / severe]")

    # TM
    tm = box("TM", detected_labels)
    lines.append(f"  {tm} TM — Tooth Misalignment")
    if "TM" in detected_labels:
        lines.append( "        Location : [FIELD: jaw region(s) and segment(s)]")
        lines.append( "        Type     : [FIELD: crowding / spacing / arch deviation]")

    # MR
    mr = box("MR", detected_labels)
    lines.append(f"  {mr} MR — Mandibular Retrusion")
    if "MR" in detected_labels:
        lines.append( "        Location : Whole mandible (skeletal)")
        lines.append( "        Notes    : Class II relationship suspected. Cephalometric assessment recommended.")

    lines.append("")
    lines.append("Appliances Detected:")
    for code, desc in [("IOA","Clear aligner attachment"), ("CFOA","Cast fixed appliance"),
                        ("OB","Orthodontic brace"),         ("FOD","Fixed orthodontic device")]:
        lines.append(f"  {box(code, detected_labels)} {code} — {desc}")

    lines.append("")
    lines.append("Developmental / Normal Findings:")
    lines.append(f"  {box('HT', detected_labels)} HT — Healthy teeth, no malocclusion")
    lines.append(f"  {box('TE', detected_labels)} TE — Tooth emergence (normal development)")

    lines.append("")
    lines.append(f"Malocclusion Referral Required : {mal_refer}")
    if mal_refer == "Yes":
        lines.append(f"  Refer to : {refer_to}")
        lines.append(f"  Urgency  : {mal_urgency}")

    lines.append("")
    lines.append("--- GINGIVITIS FINDINGS ---")
    lines.append("Model Used : MGI-based Gingivitis Detector")
    lines.append("")

    for jaw in ["maxilla", "mandible"]:
        jaw_label = "Maxilla (upper jaw)" if jaw == "maxilla" else "Mandible (lower jaw)"
        s = mgi_scores[jaw]
        lines.append(f"  {jaw_label}: MGI Score {s['overall']} — {mgi_text_map[s['overall']]}")
        lines.append(f"    Anterior  : {s['anterior']}")
        lines.append(f"    Premolar  : {s['premolar']}")
        lines.append(f"    Molar     : {s['molar']}")

    lines.append("")
    lines.append(f"Overall MGI Score (worst observed) : {overall_mgi} — {mgi_text_map[overall_mgi]}")
    lines.append(f"Gingivitis Referral                : {gingi_refer}")
    lines.append(f"Oral Hygiene Instructions Provided : {ohi_provided}")

    lines.append("")
    lines.append("--- COMBINED CLINICAL SUMMARY ---")
    lines.append("[GENERATING...]")  # placeholder — SLM fills this next
    lines.append("")
    lines.append("Reviewed by (operator / clinician): _____________________________")
    lines.append("Signature: __________________________  Date: ____________________")
    lines.append("")
    lines.append("This note is AI-assisted and must be reviewed and countersigned")
    lines.append("by a qualified dental professional before use in clinical care.")

    return "\n".join(lines)

# ── Retrieve KB context for summary generation ───────────────────────────────
findings_text = ", ".join([label_map[l] for l in detected_labels if l in label_map])
query = (f"Referral criteria for {findings_text} "
         f"with MGI {overall_mgi} {mgi_text_map[overall_mgi]} priority {priority}")

matching_docs = db.similarity_search(query, k=4)
seen = set()
unique_docs = [d for d in matching_docs
               if d.page_content not in seen and not seen.add(d.page_content)]
context = "\n".join([d.page_content for d in unique_docs])

# ── SLM writes only the summary ───────────────────────────────────────────────
summary_prompt = f"""
You are an NDCS dental screening assistant.
Write ONLY a 2-3 sentence clinical summary for the combined findings below.
Be plain English. No bullet points. No headers. Stop after 3 sentences.

Findings:
- Malocclusion: {findings_text}
- Overall MGI Score: {overall_mgi} ({mgi_text_map[overall_mgi]})
- Maxilla MGI: {mgi_scores['maxilla']['overall']}, Mandible MGI: {mgi_scores['mandible']['overall']}
- Malocclusion referral: {mal_refer} — {refer_to} ({mal_urgency})
- Gingivitis referral: {gingi_refer}

NDCS Guidelines:
{context}

Summary:
"""

print("Generating clinical summary...")
response = ollama.generate(
    model="phi3",
    prompt=summary_prompt,
    options={"num_predict": 150}
)
summary = response["response"].strip()

# ── Assemble final note ───────────────────────────────────────────────────────
note = build_partial_note().replace("[GENERATING...]", summary)
print("\n" + "=" * 60)
print(note)
print("=" * 60)