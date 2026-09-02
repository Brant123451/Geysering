from pathlib import Path

root = Path(__file__).resolve().parent
count = 0
for path in root.glob("*.sh"):
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    path.write_bytes(data)
    path.chmod(0o755)
    count += 1
print(f"scripts_normalized {count}")
