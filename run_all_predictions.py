import subprocess

commands = [
    ["python", "src/predict_humidity.py", "--temperature", "22.5", "--wind_speed", "10.5", "--pressure", "1012"],
    ["python", "src/predict_wind_speed.py", "--temperature", "22.5", "--humidity", "0.75", "--pressure", "1012"],
    ["python", "src/predict_pressure.py", "--temperature", "22.5", "--humidity", "0.75", "--wind_speed", "10.5"],
    ["python", "src/predict_temperature.py", "--humidity", "0.75", "--wind_speed", "10.5", "--pressure", "1012"],
]

for cmd in commands:
    print(f"Kör: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Fel:", result.stderr)
