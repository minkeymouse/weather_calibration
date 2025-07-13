import os
import sys
import subprocess
import PySimpleGUI as sg
import pandas as pd
import torch

# — helper to open files or URLs cross-platform —
def open_path(path):
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.call(["open", path])
    else:
        subprocess.call(["xdg-open", path])

# — your real model loader & runner (you can replace with your own imports) —
def load_model(device):
    model = torch.load("model.pt", map_location=device)
    model.to(device).eval()
    return model

# — main wizard —
def main():
    # Step 1 & 2: GPU & (optional) retrain decision
    layout1 = [
        [sg.Text("1) VRAM 16GB 이상의 GPU가 준비되어 있습니까?")],
        [sg.Radio("Yes", "GPU", key="-GPU_YES-"), sg.Radio("No", "GPU", key="-GPU_NO-", default=True)],
        [sg.Text("2) 예측 성능을 새로운 데이터에 최적화 하겠습니까?")],
        [sg.Radio("Yes", "TRAIN_YES", key="-TRAIN_YES-"), sg.Radio("No", "TRAIN_NO", key="-TRAIN_NO-", default=True)],
        [sg.Button("Next")]
    ]
    win1 = sg.Window("Setup Options", layout1)
    ev, vals = win1.read(); win1.close()
    has_gpu = vals["-GPU_YES-"]
    do_train = vals["-TRAIN_YES-"]
    device = torch.device("cuda" if has_gpu and torch.cuda.is_available() else "cpu")

    # Step 3: check if they have data
    layout2 = [
        [sg.Text("3) 관측 및 예측용 CSV 파일이 준비되어 있습니까?")],
        [sg.Radio("Yes", "DATA", key="-DATA_YES-"), sg.Radio("No", "DATA", key="-DATA_NO-", default=True)],
        [sg.Button("Next")]
    ]
    win2 = sg.Window("Data Available?", layout2)
    ev, vals = win2.read(); win2.close()
    if not vals["-DATA_YES-"]:
        sg.popup(
            "데이터가 없습니다.\n"
            "• GitHub 레포: https://github.com/yourname/yourrepo\n"
            "• 예시 파일 위치:\n"
            "  - input/관측데이터(예시)\n"
            "  - input/예측데이터(예시)\n\n"
            "준비 후 다시 실행해주세요."
        )
        sys.exit(0)

    # Step 4: let them upload the two CSVs
    layout3 = [
        [sg.Text("4a) 관측 데이터(obs.csv)를 선택하세요:")],
        [sg.Input(key="-OBS-"), sg.FileBrowse(file_types=(("CSV","*.csv"),))],
        [sg.Text("4b) 예측 데이터(fcst.csv)를 선택하세요:")],
        [sg.Input(key="-FCST-"), sg.FileBrowse(file_types=(("CSV","*.csv"),))],
        [sg.Button("Load & Preprocess"), sg.Button("Exit")]
    ]
    win3 = sg.Window("Upload Your Data", layout3)
    ev, vals = win3.read()
    win3.close()
    if ev != "Load & Preprocess":
        sys.exit(0)

    # save uploads into input/
    os.makedirs("input", exist_ok=True)
    try:
        df_obs = pd.read_csv(vals["-OBS-"], encoding="utf-8")
        df_fcst = pd.read_csv(vals["-FCST-"], encoding="utf-8")
        df_obs.to_csv("input/obs.csv", index=False)
        df_fcst.to_csv("input/fcst.csv", index=False)
    except Exception as e:
        sg.popup("파일 저장 오류", str(e))
        sys.exit(1)

    # Step 5: run preprocessing
    ret = subprocess.call([sys.executable, "inputprocess.py"])
    if ret != 0:
        sg.popup("전처리 오류: inputprocess.py가 실패했습니다.")
        sys.exit(1)

    # Step 6: prediction (with or without fine-tuning)
    if has_gpu and do_train:
        # retrain & predict
        cmd = [sys.executable, "predict.py", "--train", "--epochs", "20"]
    else:
        # pure predict
        cmd = [sys.executable, "predict.py"]
    ret = subprocess.call(cmd)
    if ret != 0:
        sg.popup("오류: predict.py가 실패했습니다.")
        sys.exit(1)

    sg.popup("✅ 완료", "output/prediction.csv 에 결과가 저장되었습니다.")

if __name__ == "__main__":
    main()
