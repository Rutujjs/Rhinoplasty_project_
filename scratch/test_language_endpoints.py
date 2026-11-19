import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from backend.app.main import app
import fitz

client = TestClient(app)

print("--- Testing GET /languages ---")
resp = client.get("/languages")
assert resp.status_code == 200, f"Status {resp.status_code}"
langs = resp.json()
print("Total languages:", len(langs))
assert len(langs) >= 20
spanish = next(l for l in langs if l["language"] == "Spanish")
assert spanish["target_countries"] == ["Spain", "Mexico", "Colombia"]
assert spanish["priority"] == "High"
assert spanish["phase"] == "Phase 1"

print("\n--- Testing GET /languages/countries ---")
resp_c = client.get("/languages/countries")
assert resp_c.status_code == 200
countries = resp_c.json()
print("Total supported countries:", len(countries))
assert len(countries) >= 30
ch = next(c for c in countries if c["country"] == "Switzerland")
print("Switzerland mapped languages:", ch["associated_languages"])
assert "German" in ch["associated_languages"]
assert "French" in ch["associated_languages"]
assert "Italian" in ch["associated_languages"]

in_c = next(c for c in countries if c["country"] == "India")
print("India mapped languages:", in_c["associated_languages"])
assert "Hindi" in in_c["associated_languages"]
assert "Tamil" in in_c["associated_languages"]
assert "Malayalam" in in_c["associated_languages"]
assert "Telugu" in in_c["associated_languages"]

print("\n--- Testing GET /languages/phases ---")
resp_p = client.get("/languages/phases")
assert resp_p.status_code == 200
phases = resp_p.json()
assert len(phases) == 3

print("\n--- Testing GET /languages/report/pdf ---")
resp_pdf = client.get("/languages/report/pdf")
assert resp_pdf.status_code == 200
assert resp_pdf.headers["content-type"] == "application/pdf"
assert "attachment" in resp_pdf.headers["content-disposition"]
print("PDF size bytes:", len(resp_pdf.content))

# Verify PDF text extraction
doc = fitz.open(stream=resp_pdf.content, filetype="pdf")
text = ""
for page in doc:
    text += page.get_text()

assert "FACEIFY LABS" in text
assert "Language Expansion Strategy" in text
assert "Spanish" in text
assert "Spain, Mexico, Colombia" in text
assert "UAE, Saudi Arabia, Qatar" in text
assert "Phase 1" in text
assert "Phase 2" in text
assert "Phase 3" in text
print("PDF verified: contains all strategic language tables and rollout phases!")

print("\nALL LANGUAGE TESTS PASSED!")
