import os
import json

# 📁 Directory where all your numbered verse files are stored
VERSE_DIR = "./bible_verses"  # change as needed

# 📖 Load the full list of expected KJV references
with open("verse_counts_kjv.json", "r") as f:
    verse_structure = json.load(f)

expected_refs = []
for book, chapters in verse_structure.items():
    for chapter, verse_count in chapters.items():
        for verse in range(1, verse_count + 1):
            expected_refs.append(f"{book} {chapter}:{verse}")

# 📘 Set to keep track of all found references
found_refs = set()

# 📂 Read files 1.json through 9000.json
for i in range(1, 9001):
    file_path = os.path.join(VERSE_DIR, f"{i}.json")
    if not os.path.isfile(file_path):
        print(f"⚠️ Missing file: {i}.json")
        continue

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            ref = data.get("reference")
            if ref:
                found_refs.add(ref.strip())
            else:
                print(f"⚠️ No reference in file: {i}.json")
    except Exception as e:
        print(f"❌ Error reading {i}.json: {e}")

# 🔍 Compare and find missing references
missing_refs = sorted(set(expected_refs) - found_refs)

# 📝 Report
if missing_refs:
    print(f"\n❌ Missing {len(missing_refs)} verses:")
    for ref in missing_refs:
        print(ref)
    with open("missing_verses_report.txt", "w") as f:
        f.write("\n".join(missing_refs))
    print("\n✅ Saved missing references to missing_verses_report.txt")
else:
    print("✅ All verses accounted for!")

