import pytest
import io
from fastapi.testclient import TestClient
from api.index import app as api_app

@pytest.fixture
def client():
    return TestClient(api_app)

def test_health_endpoint(client):
    """Test the /health endpoint returns healthy status and metadata."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "Briefly AI" in data["service"]
    assert "quick" in data["supported_styles"]
    assert "medium" in data["supported_lengths"]

def test_root_endpoint_html(client):
    """Test the root / endpoint serves index.html with all core workspace features."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html = res.text
    assert "Briefly AI" in html
    assert "Turn long text into" in html
    assert "Executive Summary" in html
    assert "Recent Summaries" in html
    assert "theme-toggle-btn" in html
    assert "btn-copy-summary" in html
    assert "btn-download-txt" in html
    assert "file-upload-input" in html

def test_summarize_default(client):
    """Test standard summarization with default parameters."""
    text = (
        "Artificial intelligence and machine learning have rapidly expanded across multiple sectors. "
        "Healthcare teams use specialized models to spot anomalies in radiological scans. "
        "Automated algorithms process complex diagnostics with high accuracy and minimal latency. "
        "Ethical considerations around data privacy and transparency remain an important conversation."
    )
    res = client.post("/summarize/", json={"dialogue": text})
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert len(data["summary"]) > 10
    assert data["style"] == "quick"
    assert data["length"] == "medium"

def test_all_four_summary_styles(client):
    """Test that all 4 summary styles (quick, detailed, bullet, takeaways) work and format correctly."""
    sample = (
        "Quantum computing harnesses the strange properties of quantum mechanics to solve complex problems. "
        "Traditional computers use bits that represent either zeros or ones. "
        "Quantum processors use qubits that can exist in a superposition of multiple states simultaneously. "
        "This allows quantum algorithms to process exponentially larger solution spaces for encryption and materials science. "
        "Substantial challenges remain in error correction, cryogenic cooling, and quantum coherence."
    )
    
    # 1. Quick
    res_q = client.post("/summarize/", json={"dialogue": sample, "style": "quick", "length": "medium"})
    assert res_q.status_code == 200
    assert len(res_q.json()["summary"]) > 0

    # 2. Detailed
    res_d = client.post("/summarize/", json={"dialogue": sample, "style": "detailed", "length": "long"})
    assert res_d.status_code == 200
    assert len(res_d.json()["summary"]) > 0

    # 3. Bullet Points
    res_b = client.post("/summarize/", json={"dialogue": sample, "style": "bullet", "length": "medium"})
    assert res_b.status_code == 200
    summary_b = res_b.json()["summary"]
    assert "•" in summary_b, "Bullet style must contain bullet points"

    # 4. Key Takeaways
    res_t = client.post("/summarize/", json={"dialogue": sample, "style": "takeaways", "length": "medium"})
    assert res_t.status_code == 200
    summary_t = res_t.json()["summary"]
    assert any(tag in summary_t for tag in ["Takeaway", "Finding", "Insight", "Point"]), "Takeaways style must contain semantic takeaway tags"

def test_summary_lengths(client):
    """Test short, medium, and long length settings modulate the summary depth."""
    sample = (
        "Autonomous driving systems merge LiDAR, radar, and computer vision cameras into real-time perception engines. "
        "High-definition digital mapping provides centimeter-accurate localization across complex urban road grids. "
        "Machine learning models predict pedestrian motion vectors up to five seconds in advance. "
        "Redundant brake actuators and dual steering controllers protect against hardware failures. "
        "Regulatory validation frameworks evaluate edge-case scenarios under rain, snow, and dense fog."
    )
    res_short = client.post("/summarize/", json={"dialogue": sample, "style": "quick", "length": "short"})
    res_long = client.post("/summarize/", json={"dialogue": sample, "style": "quick", "length": "long"})

    assert res_short.status_code == 200
    assert res_long.status_code == 200

    len_short = len(res_short.json()["summary"])
    len_long = len(res_long.json()["summary"])
    assert len_short <= len_long, "Short summary must be less than or equal to long summary"

def test_empty_and_invalid_input(client):
    """Test proper validation on empty or whitespace text."""
    res_empty = client.post("/summarize/", json={"dialogue": ""})
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()

    res_spaces = client.post("/summarize/", json={"dialogue": "   \n\t  "})
    assert res_spaces.status_code == 400
    assert "empty" in res_spaces.json()["detail"].lower()

def test_file_upload_txt(client):
    """Test uploading a valid TXT document."""
    txt_content = b"This is a test plain text document containing relevant summary information for testing."
    files = {"file": ("document.txt", io.BytesIO(txt_content), "text/plain")}
    res = client.post("/upload/", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "document.txt"
    assert "test plain text document" in data["text"]
    assert data["words"] > 5

def test_file_upload_pdf(client):
    """Test uploading a generated PDF document."""
    import pypdf
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    
    # We can write text or test the PDF upload handling
    bio = io.BytesIO()
    writer.write(bio)
    bio.seek(0)
    
    files = {"file": ("sample.pdf", bio, "application/pdf")}
    res = client.post("/upload/", files=files)
    # If the blank pdf has no text layer, it returns 400 with a clear message
    assert res.status_code in [200, 400]
    if res.status_code == 400:
        assert "scanned" in res.json()["detail"].lower() or "unable" in res.json()["detail"].lower()

def test_file_upload_unsupported(client):
    """Test uploading an unsupported file format like .exe or .png."""
    files = {"file": ("test.exe", io.BytesIO(b"fake binary"), "application/octet-stream")}
    res = client.post("/upload/", files=files)
    assert res.status_code == 400
    assert "unsupported" in res.json()["detail"].lower()

def test_local_app_endpoints():
    """Verify that app.py works identically for local execution."""
    from app import app as local_app
    with TestClient(local_app) as local_client:
        res = local_client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

        res_sum = local_client.post(
            "/summarize/",
            json={
                "dialogue": "Natural language processing enables computers to comprehend human language effectively.",
                "style": "bullet"
            }
        )
        assert res_sum.status_code == 200
        assert "•" in res_sum.json()["summary"]

