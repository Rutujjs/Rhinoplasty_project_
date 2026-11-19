import urllib.request
from pathlib import Path

target_dir = Path("nasal-deformation-pipeline-main/models/bisenet/weights")
target_dir.mkdir(parents=True, exist_ok=True)
target_file = target_dir / "resnet18.onnx"

urls = [
    "https://huggingface.co/PayamFard123/dermaintel-face-parsing/resolve/main/resnet18.onnx",
    "https://huggingface.co/thedemons/mirayu-models/resolve/main/bisenet_resnet18.onnx"
]

success = False
for url in urls:
    try:
        print(f"Attempting download from {url}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, str(target_file))
        size_mb = target_file.stat().st_size / (1024 * 1024)
        print(f"Downloaded {target_file.name}, size: {size_mb:.2f} MB")
        if size_mb > 10:  # Valid model should be ~50MB
            success = True
            break
        else:
            print("File too small, possibly an error page.")
    except Exception as e:
        print(f"Failed from {url}: {e}")

if success:
    import onnxruntime as ort
    session = ort.InferenceSession(str(target_file))
    print("ONNX session successfully loaded!")
    for inp in session.get_inputs():
        print(f"Input: name={inp.name}, shape={inp.shape}, type={inp.type}")
    for out in session.get_outputs():
        print(f"Output: name={out.name}, shape={out.shape}, type={out.type}")
else:
    print("Could not download valid ONNX model.")

