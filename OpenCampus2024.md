# OpenCampus 2024

## Memo



## セットアップ

### 必要なもの
- Unitree A1
- 充電器
- LAN ケーブル
- コントローラー (A1)
- コントローラー (xbox)
- 障害物

### 1. LANケーブルでA1とPCを接続

<img src="C:\Users\hayas\Desktop\OpenCumpus2024\img\LAN_port.svg" alt="LAN Port" style="width:50%;">

### 2. 仮想環境作成
※python 環境があるのが前提
```
python -m venv env
```
#### Activate
```
env\Scripts\activate
```

### 3. ライブラリインストール
```
pip install -r src/requirements.txt
```