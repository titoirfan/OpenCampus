# OpenCampus 2024

## Memo
```
git add .
git commit -m "update"
git push
```

## セットアップ

### 必要なもの
- Unitree A1
- 充電器
- LAN ケーブル
- コントローラー (A1)
- コントローラー (xbox)
- 障害物


### 1. 仮想環境作成
※ python 環境があるのが前提
```
python -m venv env
```
#### Activate
```
env\Scripts\activate
```

### 2. ライブラリインストール

```
pip install --upgrade pip
pip install -r src/requirements.txt
pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. LANケーブルでA1とPCを接続

<img src="C:\Users\hayas\Desktop\OpenCumpus2024\img\LAN_port.svg" alt="LAN Port" style="width:50%;">