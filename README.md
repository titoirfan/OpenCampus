# OpenCampus 2024



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

pytorchのバージョンはcudaのバージョンにあわせる．
```
pip install --upgrade pip
pip install -r src/requirements.txt
pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. LANケーブルでA1とPCを接続

<img src="./img/LAN_port.svg" width="50%">

イーサネットのIPアドレスを確認
```
ipconfig
```

[RL_server.py](./src/RL_server.py)のhostを変更

```
def start_server():
    
    ######### 通信 #########################################
    
    # host = "169.254.122.147" # 有線
    host = "169.254.250.232"
    # host = "192.168.12.136" # 無線
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server is listening on {host}:{port}")
    
    ########################################################
```

[cpg.cpp](src\unitree_legged_sdk\code\cpg.cpp)のipアドレスを変更

```
void Custom::TCPClient() {
    int sock = 0;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "Socket creation error" << std::endl;
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(12345);

    // 有線
    if (inet_pton(AF_INET, "169.254.250.232", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported" << std::endl;
        return;
    }
```

### unitree_legged_sdk の build

```
scp -r unitree_legged_sdk pi@169.254.187.189:/home/OpenCampus2024/ryosei
```
ssh のパスワードは　123
```
ssh pi@169.254.187.189
cd /home/OpenCampus2024/ryosei/unitree_legged_sdk
mkdir build
cd build
cmake ..
make
```

## Start

### unitreeのコントローラーとA1を起動
どちらも電源ボタンを２回押す（２回目は長押し）
A1の起動が完了するまで待つ（立ち上がるまで）
L2+A を３回ほど押して，胴体の高さを下げる
L2+B で脱力させる

### 自作コントローラーを起動

xboxコントローラーをPCに接続

ターミナルを２つ開く

１つ目のターミナル

```
python src/RL_server.py
```

２つ目のターミナル

```
ssh pi@169.254.187.189
cd /home/OpenCampus2024/ryosei/unitree_legged_sdk/build
./cpg
```

## Control

<img src="./img/xboxController.svg" width="100%">

Forward + Turn も可

## Memo
```
git add .
git commit -m "update"
git push
```