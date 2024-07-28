import socket
import struct
import numpy as np

import sys
import os
import yaml

import pygame
import pytz
import datetime
import csv

script_dir = os.path.dirname(__file__)    
parent_dir = os.path.dirname(script_dir)  
sys.path.append(parent_dir)


from SoftActorCritic.SAC import SAC

train_log_Forward = rf"{script_dir}/TrainLog/SACTrain_240705_0/forward_00_05/240708_144207"
networks_Forward = "episode_6200.pt"

train_log_Backward = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrain_240705_0\Backward_03_08\240708_145445"
networks_Backward = "episode_5600.pt"

train_log_Left = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrain_240705_0\Left_03_08\240705_204533"
networks_Left = "episode_5600.pt"

train_log_Right = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrain_240705_0\Right_03_08\240705_204631"
networks_Right = "episode_5600.pt"


train_log_TL = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrain_240705_0\TurnLeft_03_08\240705_203739"
networks_TL = "episode_6200.pt"

train_log_TR = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrain_240705_0\TurnRight_03_08\240705_203910"
networks_TR = "episode_5600.pt"


train_log_TLF = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrain_240705_0\TurnLeftForward_03_08_00_05\240705_204039"
networks_TLF = "episode_5600.pt"

train_log_TRF = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrain_240705_0\TurnRightForward_03_08_00_05\240708_144959"
networks_TRF = "episode_5600.pt"

train_log_Step = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrain_240705_0\Step_omega_10_45\240705_205004"
networks_Step = "episode_5700.pt"

# train_log_delta = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrainDualplus_240710_0\delta_size_10-10-05_dekoboko00-12\240711_015231"
# networks_delta = "episode_3000.pt"

train_log_delta = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrainDualplus_240710_0\delta_size_10-10-05_dekoboko00-12\240715_014032"
networks_delta = "episode_5300.pt"

train_log_delta2 = r"C:\Users\hayas\workspace_A1Real\Log2\SACTrainDual_240711_0\delta_size_10-10-05_dekoboko00-12\240715_022240"
networks_delta2 = "episode_5100.pt"

# make log dir
timezone = pytz.timezone('Asia/Tokyo')
start_datetime = datetime.datetime.now(timezone)    
start_formatted = start_datetime.strftime("%y%m%d_%H%M%S")



def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def start_server():
    
    ######### 通信 #########################################
    
    # host = "169.254.122.147" # 有線
    host = "169.254.250.232" # 有線
    # host = "192.168.12.136" # 無線
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server is listening on {host}:{port}")
    
    ########################################################


    ######### コントローラー #################################
    pygame.init()
    pygame.joystick.init()
    
    try:
        joy = pygame.joystick.Joystick(0)  # create a joystick instance
        joy.init()  # init instance
        print(f"Joystick name: {joy.get_name()}")
    except pygame.error:
        print("No joystick found.")
        return
    
    ########################################################
    
    ######### エージェント ###################################
    observation_space = Box(63)
    action_space = Box(12)
    
    
    # Agent Forward 0.0 ~ 0.5 [m/s]
    config_Forward = os.path.join(train_log_Forward, "config.yaml")
    print(f"Config path: {config_Forward}")
    with open(config_Forward, 'r') as yml:
        cfg_Forward = yaml.safe_load(yml)
        
    cfg_Forward["gpu"] = 0
    
    agent_Forward = SAC(observation_space.shape, action_space, cfg_Forward)
    checkpoint_Forward = os.path.join(train_log_Forward, "Networks", networks_Forward)
    agent_Forward.load_checkpoint(ckpt_path=checkpoint_Forward, evaluate=True)

    mu_high_Forward = np.array(cfg_Forward["env"]["CPG"]["muHigh"])
    mu_low_Forward = np.array(cfg_Forward["env"]["CPG"]["muLow"])
    omega_high_Forward = 2* np.pi * np.array(cfg_Forward["env"]["CPG"]["omegaHigh"])
    omega_low_Forward = 2* np.pi * np.array(cfg_Forward["env"]["CPG"]["omegaLow"])
    psi_high_Forward = 2* np.pi * np.array(cfg_Forward["env"]["CPG"]["psiHigh"])
    psi_low_Forward = 2* np.pi * np.array(cfg_Forward["env"]["CPG"]["psiLow"])
    
    # Agent Backward -0.8 ~ -0.3 [m/s]
    config_Backward = os.path.join(train_log_Backward, "config.yaml")
    print(f"Config path: {config_Backward}")
    with open(config_Backward, 'r') as yml:
        cfg_Backward = yaml.safe_load(yml)
        
    cfg_Backward["gpu"] = 0
    
    agent_Backward = SAC(observation_space.shape, action_space, cfg_Backward)
    checkpoint_Backward = os.path.join(train_log_Backward, "Networks", networks_Backward)
    agent_Backward.load_checkpoint(ckpt_path=checkpoint_Backward, evaluate=True)

    mu_high_Backward = np.array(cfg_Backward["env"]["CPG"]["muHigh"])
    mu_low_Backward = np.array(cfg_Backward["env"]["CPG"]["muLow"])
    omega_high_Backward = 2* np.pi * np.array(cfg_Backward["env"]["CPG"]["omegaHigh"])
    omega_low_Backward = 2* np.pi * np.array(cfg_Backward["env"]["CPG"]["omegaLow"])
    psi_high_Backward = 2* np.pi * np.array(cfg_Backward["env"]["CPG"]["psiHigh"])
    psi_low_Backward = 2* np.pi * np.array(cfg_Backward["env"]["CPG"]["psiLow"])


    # Agent Left 0.3 ~ 0.8 [m/s]
    config_Left = os.path.join(train_log_Left, "config.yaml")
    print(f"Config path: {config_Left}")
    with open(config_Left, 'r') as yml:
        cfg_Left = yaml.safe_load(yml)
        
    cfg_Left["gpu"] = 0
    
    agent_Left = SAC(observation_space.shape, action_space, cfg_Left)
    checkpoint_Left = os.path.join(train_log_Left, "Networks", networks_Left)
    agent_Left.load_checkpoint(ckpt_path=checkpoint_Left, evaluate=True)

    mu_high_Left = np.array(cfg_Left["env"]["CPG"]["muHigh"])
    mu_low_Left = np.array(cfg_Left["env"]["CPG"]["muLow"])
    omega_high_Left = 2* np.pi * np.array(cfg_Left["env"]["CPG"]["omegaHigh"])
    omega_low_Left = 2* np.pi * np.array(cfg_Left["env"]["CPG"]["omegaLow"])
    psi_high_Left = 2* np.pi * np.array(cfg_Left["env"]["CPG"]["psiHigh"])
    psi_low_Left = 2* np.pi * np.array(cfg_Left["env"]["CPG"]["psiLow"])
    
    # Agent Right -0.8 ~ -0.3 [m/s]
    config_Right = os.path.join(train_log_Right, "config.yaml")
    print(f"Config path: {config_Right}")
    with open(config_Right, 'r') as yml:
        cfg_Right = yaml.safe_load(yml)
        
    cfg_Right["gpu"] = 0
    
    agent_Right = SAC(observation_space.shape, action_space, cfg_Right)
    checkpoint_Right = os.path.join(train_log_Right, "Networks", networks_Right)
    agent_Right.load_checkpoint(ckpt_path=checkpoint_Right, evaluate=True)

    mu_high_Right = np.array(cfg_Right["env"]["CPG"]["muHigh"])
    mu_low_Right = np.array(cfg_Right["env"]["CPG"]["muLow"])
    omega_high_Right = 2* np.pi * np.array(cfg_Right["env"]["CPG"]["omegaHigh"])
    omega_low_Right = 2* np.pi * np.array(cfg_Right["env"]["CPG"]["omegaLow"])
    psi_high_Right = 2* np.pi * np.array(cfg_Right["env"]["CPG"]["psiHigh"])
    psi_low_Right = 2* np.pi * np.array(cfg_Right["env"]["CPG"]["psiLow"])
    
    # Agent TurnLeft 0.3 ~ 0.8 [rad/s]
    config_TL = os.path.join(train_log_TL, "config.yaml")
    print(f"Config path: {config_TL}")
    with open(config_TL, 'r') as yml:
        cfg_TL = yaml.safe_load(yml)
        
    cfg_TL["gpu"] = 0
    
    agent_TL = SAC(observation_space.shape, action_space, cfg_TL)
    checkpoint_TL = os.path.join(train_log_TL, "Networks", networks_TL)
    agent_TL.load_checkpoint(ckpt_path=checkpoint_TL, evaluate=True)

    mu_high_TL = np.array(cfg_TL["env"]["CPG"]["muHigh"])
    mu_low_TL = np.array(cfg_TL["env"]["CPG"]["muLow"])
    omega_high_TL = 2* np.pi * np.array(cfg_TL["env"]["CPG"]["omegaHigh"])
    omega_low_TL = 2* np.pi * np.array(cfg_TL["env"]["CPG"]["omegaLow"])
    psi_high_TL = 2* np.pi * np.array(cfg_TL["env"]["CPG"]["psiHigh"])
    psi_low_TL = 2* np.pi * np.array(cfg_TL["env"]["CPG"]["psiLow"])
    
    # Agent TurnRight -0.8 ~ -0.3 [rad/s]
    config_TR = os.path.join(train_log_TR, "config.yaml")
    print(f"Config path: {config_TR}")
    with open(config_TR, 'r') as yml:
        cfg_TR = yaml.safe_load(yml)
        
    cfg_TR["gpu"] = 0
    
    agent_TR = SAC(observation_space.shape, action_space, cfg_TR)
    checkpoint_TR = os.path.join(train_log_TR, "Networks", networks_TR)
    agent_TR.load_checkpoint(ckpt_path=checkpoint_TR, evaluate=True)

    mu_high_TR = np.array(cfg_TR["env"]["CPG"]["muHigh"])
    mu_low_TR = np.array(cfg_TR["env"]["CPG"]["muLow"])
    omega_high_TR = 2* np.pi * np.array(cfg_TR["env"]["CPG"]["omegaHigh"])
    omega_low_TR = 2* np.pi * np.array(cfg_TR["env"]["CPG"]["omegaLow"])
    psi_high_TR = 2* np.pi * np.array(cfg_TR["env"]["CPG"]["psiHigh"])
    psi_low_TR = 2* np.pi * np.array(cfg_TR["env"]["CPG"]["psiLow"])
    
    # Agent TurnLeftForward 0.3 ~ 0.8 [rad/s] 0.0 ~ 0.5 [m/s]
    config_TLF = os.path.join(train_log_TLF, "config.yaml")
    print(f"Config path: {config_TLF}")
    with open(config_TLF, 'r') as yml:
        cfg_TLF = yaml.safe_load(yml)
        
    cfg_TLF["gpu"] = 0
    
    agent_TLF = SAC(observation_space.shape, action_space, cfg_TLF)
    checkpoint_TLF = os.path.join(train_log_TLF, "Networks", networks_TLF)
    agent_TLF.load_checkpoint(ckpt_path=checkpoint_TLF, evaluate=True)

    mu_high_TLF = np.array(cfg_TLF["env"]["CPG"]["muHigh"])
    mu_low_TLF = np.array(cfg_TLF["env"]["CPG"]["muLow"])
    omega_high_TLF = 2* np.pi * np.array(cfg_TLF["env"]["CPG"]["omegaHigh"])
    omega_low_TLF = 2* np.pi * np.array(cfg_TLF["env"]["CPG"]["omegaLow"])
    psi_high_TLF = 2* np.pi * np.array(cfg_TLF["env"]["CPG"]["psiHigh"])
    psi_low_TLF = 2* np.pi * np.array(cfg_TLF["env"]["CPG"]["psiLow"])
    
    # Agent TurnLeftForward -0.8 ~ -0.3 [rad/s] 0.0 ~ 0.5 [m/s]
    config_TRF = os.path.join(train_log_TRF, "config.yaml")
    print(f"Config path: {config_TRF}")
    with open(config_TRF, 'r') as yml:
        cfg_TRF = yaml.safe_load(yml)
        
    cfg_TRF["gpu"] = 0
    
    agent_TRF = SAC(observation_space.shape, action_space, cfg_TRF)
    checkpoint_TRF = os.path.join(train_log_TRF, "Networks", networks_TRF)
    agent_TRF.load_checkpoint(ckpt_path=checkpoint_TRF, evaluate=True)

    mu_high_TRF = np.array(cfg_TRF["env"]["CPG"]["muHigh"])
    mu_low_TRF = np.array(cfg_TRF["env"]["CPG"]["muLow"])
    omega_high_TRF = 2* np.pi * np.array(cfg_TRF["env"]["CPG"]["omegaHigh"])
    omega_low_TRF = 2* np.pi * np.array(cfg_TRF["env"]["CPG"]["omegaLow"])
    psi_high_TRF = 2* np.pi * np.array(cfg_TRF["env"]["CPG"]["psiHigh"])
    psi_low_TRF = 2* np.pi * np.array(cfg_TRF["env"]["CPG"]["psiLow"])
    
    # Agent Step
    config_Step = os.path.join(train_log_Step, "config.yaml")
    print(f"Config path: {config_Step}")
    with open(config_Step, 'r') as yml:
        cfg_Step = yaml.safe_load(yml)
        
    cfg_Step["gpu"] = 0
    
    agent_Step = SAC(observation_space.shape, action_space, cfg_Step)
    checkpoint_Step = os.path.join(train_log_Step, "Networks", networks_Step)
    agent_Step.load_checkpoint(ckpt_path=checkpoint_Step, evaluate=True)

    mu_high_Step = np.array(cfg_Step["env"]["CPG"]["muHigh"])
    mu_low_Step = np.array(cfg_Step["env"]["CPG"]["muLow"])
    omega_high_Step = 2* np.pi * np.array(cfg_Step["env"]["CPG"]["omegaHigh"])
    omega_low_Step = 2* np.pi * np.array(cfg_Step["env"]["CPG"]["omegaLow"])
    psi_high_Step = 2* np.pi * np.array(cfg_Step["env"]["CPG"]["psiHigh"])
    psi_low_Step = 2* np.pi * np.array(cfg_Step["env"]["CPG"]["psiLow"])
    
    # Agent Delta
    config_delta = os.path.join(train_log_delta, "config.yaml")
    print(f"Config path: {config_delta}")
    with open(config_delta, 'r') as yml:
        cfg_delta = yaml.safe_load(yml)
        
    cfg_delta["gpu"] = 0
    
    delta_x_max = cfg_delta["env"]["Delta"]["dx_max"]
    delta_x_min = cfg_delta["env"]["Delta"]["dx_min"]   
    delta_y_max = cfg_delta["env"]["Delta"]["dy_max"]
    delta_y_min = cfg_delta["env"]["Delta"]["dy_min"]
    delta_z_max = cfg_delta["env"]["Delta"]["dz_max"]
    delta_z_min = cfg_delta["env"]["Delta"]["dz_min"]
    action_space_delta = Box(dim=12,
                            low=np.array([delta_x_min, delta_y_min, delta_z_min,
                                        delta_x_min, delta_y_min, delta_z_min,
                                        delta_x_min, delta_y_min, delta_z_min,
                                        delta_x_min, delta_y_min, delta_z_min]),
                            high=np.array([delta_x_max, delta_y_max, delta_z_max,
                                        delta_x_max, delta_y_max, delta_z_max,
                                        delta_x_max, delta_y_max, delta_z_max,
                                        delta_x_max, delta_y_max, delta_z_max]))
    
    
    agent_delta = SAC(observation_space.shape, action_space_delta, cfg_delta)
    checkpoint_delta = os.path.join(train_log_delta, "Networks", networks_delta)
    agent_delta.load_checkpoint(ckpt_path=checkpoint_delta, evaluate=True)
    
    
    # delta2
    config_delta2 = os.path.join(train_log_delta2, "config.yaml")
    print(f"Config path: {config_delta2}")
    with open(config_delta2, 'r') as yml:
        cfg_delta2 = yaml.safe_load(yml)
        
    cfg_delta2["gpu"] = 0
    
    delta_x_max2 = cfg_delta2["env"]["Delta"]["dx_max"]
    delta_x_min2 = cfg_delta2["env"]["Delta"]["dx_min"]   
    delta_y_max2 = cfg_delta2["env"]["Delta"]["dy_max"]
    delta_y_min2 = cfg_delta2["env"]["Delta"]["dy_min"]
    delta_z_max2 = cfg_delta2["env"]["Delta"]["dz_max"]
    delta_z_min2 = cfg_delta2["env"]["Delta"]["dz_min"]
    action_space_delta2 = Box(dim=12,
                            low=np.array([delta_x_min2, delta_y_min2, delta_z_min2,
                                        delta_x_min2, delta_y_min2, delta_z_min2,
                                        delta_x_min2, delta_y_min2, delta_z_min2,
                                        delta_x_min2, delta_y_min2, delta_z_min2]),
                            high=np.array([delta_x_max2, delta_y_max2, delta_z_max2,
                                            delta_x_max2, delta_y_max2, delta_z_max2,
                                        delta_x_max2, delta_y_max2, delta_z_max2,
                                        delta_x_max2, delta_y_max2, delta_z_max2]))
    
    agent_delta2 = SAC(observation_space.shape, action_space_delta2, cfg_delta2)
    checkpoint_delta2 = os.path.join(train_log_delta2, "Networks", networks_delta2)
    agent_delta2.load_checkpoint(ckpt_path=checkpoint_delta2, evaluate=True)
    
    #########################################################################################

    
    mode = 0.0
    step = 0
    policy_active = False
    delta_policy_active = True
    command = np.array([0.0, 0.0, 0.0])
    h = 0.25
    gc = 0.1
    walk_gc = 0.1
    stop_count = 0
    
    print_count = 0
    
    while True:
        client_socket, client_address = server_socket.accept()
        # print(f"Connection from {client_address} has been established.")
        
        data = client_socket.recv(1024)
        # print(f"Received data size: {len(data)} bytes")
        if len(data) < 488:  # 61 doubles * 8 bytes each
            print("Received data is less than expected")
            client_socket.close()
            continue
        
        array = struct.unpack('d' * 61, data[:488])  # 61 * 8 = 488 bytes
        
        if array[0] < 10:
            mode = 0.0
            print("Mode is set")
        
        observation = np.array(array[1:])

        # 位相を2piで割る
        observation[40:48] = observation[40:48] % (2 * np.pi)


        # xbox360controllerの状態をチェック
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == 5 and event.value > 0.9:  # ZRボタンが押された
                    mode = 0.0
                    print("ZR button is pressed")
                if event.axis == 4 and event.value > 0.9:  # ZLボタンが押された
                    mode = 1.0
                    print("ZL button is pressed")
                
        #         if event.axis == 1 and event.value < -0.001:
        #             command[0] = - 0.3 * event.value + 0.5
        #             command[1] = 0.0
        #             command[2] = 0.0
        #             print(f"x_vel: {command}")
                    
        #         elif event.axis == 2 and abs(event.value) > 0.001:
        #             command[0] = 0.0
        #             command[1] = 0.0
        #             command[2] = - 1.0 * event.value
        #             print(f"w_vel: {command}")
                
        #         else:
        #             command[0] = 0.0
        #             command[1] = 0.0
        #             command[2] = 0.0
        
        
            # Left joy stick
            if joy.get_axis(1) < -0.3:
                command[0] = - 0.5 * (joy.get_axis(1) +0.3)/0.7
                command[1] = 0.0
            elif joy.get_axis(1) > 0.3:
                command[0] = - 0.2 * (joy.get_axis(1)-0.3)/0.7 -0.3
                command[1] = 0.0
            elif joy.get_axis(0) < -0.3:
                command[1] = -0.2 * (joy.get_axis(0)+0.3)/0.7 + 0.3
            elif joy.get_axis(0) > 0.3:
                command[1] = -0.2 * (joy.get_axis(0)-0.3)/0.7 - 0.3
            else:
                command[0] = 0.0
                command[1] = 0.0
                
            # joy0 = joy.get_axis(0)
            # joy1 = joy.get_axis(1)
            # joy2 = joy.get_axis(2)
                

            # Right joy stick
            if joy.get_axis(2) < - 0.09 and abs(joy.get_axis(0)) <= 0.09:
                command[1] = 0.0
                command[2] = - 0.5 * joy.get_axis(2) + 0.3
                
            elif joy.get_axis(2) > 0.09 and abs(joy.get_axis(0)) <= 0.09:
                command[1] = 0.0
                command[2] = - 0.5 * joy.get_axis(2) - 0.3
            else:
                command[2] = 0.0
                
            
                
                
            # 方向ボタン (ハットスイッチ) の処理を追加
            hat_input = joy.get_hat(0)
            if hat_input[0] == -1:  # 左
                walk_gc -= 0.01
                walk_gc = np.clip(walk_gc, 0.0, 0.12)
            elif hat_input[0] == 1:  # 右
                walk_gc += 0.01
                walk_gc = np.clip(walk_gc, 0.0, 0.12)

            if hat_input[1] == 1:  # 上
                h += 0.01
                h = np.clip(h, 0.19, 0.30)
            elif hat_input[1] == -1:  # 下
                h -= 0.01
                h = np.clip(h, 0.19, 0.30)
                
            if joy.get_button(4) == 1:
                policy_active = True
                
            elif joy.get_button(5) == 1:
                policy_active = False
                
            if joy.get_button(2) == 1:
                delta_policy_active = True
                
            elif joy.get_button(3) == 1:
                delta_policy_active = False
                
            # print(f"hat_input: {hat_input}")

        
        
        
        # NNの処理
        if policy_active:
            observation = np.concatenate([observation, command])
            
            if delta_policy_active:
                action_deltas = agent_delta.select_action(observation,evaluate=True)
                delta = ((action_space_delta.high - action_space_delta.low)/2) * action_deltas + ((action_space_delta.low + action_space_delta.high)/2)
                
            else:
                delta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            if command[0] > 0.0 and command[1] == 0.0:
                if command[2] > 0.1:
                    # Left Turn Forward
                    action = agent_TLF.select_action(observation,evaluate=True)
                    _mu = ((mu_high_TLF - mu_low_TLF) / 2) * action[0:4] + ((mu_high_TLF + mu_low_TLF) / 2)
                    _omega = ((omega_high_TLF - omega_low_TLF) / 2) * action[4:8] + ((omega_high_TLF + omega_low_TLF) / 2)
                    _psi = ((psi_high_TLF - psi_low_TLF) / 2) * action[8:12] + ((psi_high_TLF + psi_low_TLF) / 2)
                
                elif command[2] < -0.1:
                    # Right Turn Forward
                    action = agent_TRF.select_action(observation,evaluate=True)
                    _mu = ((mu_high_TRF - mu_low_TRF) / 2) * action[0:4] + ((mu_high_TRF + mu_low_TRF) / 2)
                    _omega = ((omega_high_TRF - omega_low_TRF) / 2) * action[4:8] + ((omega_high_TRF + omega_low_TRF) / 2)
                    _psi = ((psi_high_TRF - psi_low_TRF) / 2) * action[8:12] + ((psi_high_TRF + psi_low_TRF) / 2)
                    

                else:
                    action = agent_Forward.select_action(observation,evaluate=True)
                    _mu = ((mu_high_Forward - mu_low_Forward) / 2) * action[0:4] + ((mu_high_Forward + mu_low_Forward) / 2)
                    _omega = ((omega_high_Forward - omega_low_Forward) / 2) * action[4:8] + ((omega_high_Forward + omega_low_Forward) / 2)
                    _psi = ((psi_high_Forward - psi_low_Forward) / 2) * action[8:12] + ((psi_high_Forward + psi_low_Forward) / 2)
                    
                    if delta_policy_active:
                        action_deltas = agent_delta2.select_action(observation,evaluate=True)
                        delta = ((action_space_delta2.high - action_space_delta2.low)/2) * action_deltas + ((action_space_delta2.low + action_space_delta2.high)/2)
                
                    else:
                        delta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
                    
            elif command[0] < -0.3 and command[1] == 0.0:
                action = agent_Backward.select_action(observation,evaluate=True)
                _mu = ((mu_high_Backward - mu_low_Backward) / 2) * action[0:4] + ((mu_high_Backward + mu_low_Backward) / 2)
                _omega = ((omega_high_Backward - omega_low_Backward) / 2) * action[4:8] + ((omega_high_Backward + omega_low_Backward) / 2)
                _psi = ((psi_high_Backward - psi_low_Backward) / 2) * action[8:12] + ((psi_high_Backward + psi_low_Backward) / 2)
                
            elif command[1] > 0.3 and command[0] == 0.0:
                action = agent_Left.select_action(observation,evaluate=True)
                _mu = ((mu_high_Left - mu_low_Left) / 2) * action[0:4] + ((mu_high_Left + mu_low_Left) / 2)
                _omega = ((omega_high_Left - omega_low_Left) / 2) * action[4:8] + ((omega_high_Left + omega_low_Left) / 2)
                _psi = ((psi_high_Left - psi_low_Left) / 2) * action[8:12] + ((psi_high_Left + psi_low_Left) / 2)
                
            elif command[1] < -0.3 and command[0] == 0.0:
                action = agent_Right.select_action(observation,evaluate=True)
                _mu = ((mu_high_Right - mu_low_Right) / 2) * action[0:4] + ((mu_high_Right + mu_low_Right) / 2)
                _omega = ((omega_high_Right - omega_low_Right) / 2) * action[4:8] + ((omega_high_Right + omega_low_Right) / 2)
                _psi = ((psi_high_Right - psi_low_Right) / 2) * action[8:12] + ((psi_high_Right + psi_low_Right) / 2)
            
            elif command[2] > 0.3 and command[0] == 0.0 and command[1] == 0.0:
                action = agent_TL.select_action(observation,evaluate=True)
                _mu = ((mu_high_TL - mu_low_TL) / 2) * action[0:4] + ((mu_high_TL + mu_low_TL) / 2)
                _omega = ((omega_high_TL - omega_low_TL) / 2) * action[4:8] + ((omega_high_TL + omega_low_TL) / 2)
                _psi = ((psi_high_TL - psi_low_TL) / 2) * action[8:12] + ((psi_high_TL + psi_low_TL) / 2)
                
            elif command[2] < -0.3 and command[0] == 0.0 and command[1] == 0.0:
                action = agent_TR.select_action(observation,evaluate=True)
                _mu = ((mu_high_TR - mu_low_TR) / 2) * action[0:4] + ((mu_high_TR + mu_low_TR) / 2)
                _omega = ((omega_high_TR - omega_low_TR) / 2) * action[4:8] + ((omega_high_TR + omega_low_TR) / 2)
                _psi = ((psi_high_TR - psi_low_TR) / 2) * action[8:12] + ((psi_high_TR + psi_low_TR) / 2)

                
            else:
                action = agent_Step.select_action(observation,evaluate=True)
                _mu = ((mu_high_Step - mu_low_Step) / 2) * action[0:4] + ((mu_high_Step + mu_low_Step) / 2)
                _omega = ((omega_high_Step - omega_low_Step) / 2) * action[4:8] + ((omega_high_Step + omega_low_Step) / 2)
                _psi = ((psi_high_Step - psi_low_Step) / 2) * action[8:12] + ((psi_high_Step + psi_low_Step) / 2)
                
            
            
            stop_count = 0
            gc = walk_gc
        
        else:
            _command = np.array([0.0, 0.0, 0.0])
            observation = np.concatenate([observation, _command])
            
            stop_count += 1
            if stop_count < 300:
                stop_rate = stop_count / 300
                if stop_count > 100:
                    gc = 0.03 * (1 - stop_rate) + 0.02
                action = agent_Step.select_action(observation,evaluate=True)
                _mu = ((mu_high_Step - mu_low_Step) / 2) * action[0:4] + ((mu_high_Step + mu_low_Step) / 2)
                _omega = ((omega_high_Step - omega_low_Step) / 2) * action[4:8] + ((omega_high_Step + omega_low_Step) / 2)
                _psi = ((psi_high_Step - psi_low_Step) / 2) * action[8:12] + ((psi_high_Step + psi_low_Step) / 2)
                
                
            else:
                _mu = np.array([1.0, 1.0, 1.0, 1.0])
                _omega = np.array([0.0, 0.0, 0.0, 0.0])
                _psi = np.array([0.0, 0.0, 0.0, 0.0])
                
                threshold = 0.5
                if observation[40] > threshold:
                    _omega[0] = np.pi
                if observation[41] < np.pi and observation[41] < np.pi + threshold:
                    _omega[1] = np.pi
                if observation[42] < np.pi and observation[42] < np.pi + threshold:
                    _omega[2] = np.pi
                if observation[43] > threshold:
                    _omega[3] = np.pi
                    
                if observation[44] > threshold:
                    _psi[0] = -np.pi/2
                elif observation[44] < -threshold:
                    _psi[0] = np.pi/2
                if observation[45] > threshold:
                    _psi[1] = -np.pi/2
                elif observation[45] < -threshold:
                    _psi[1] = np.pi/2
                if observation[46] > threshold:
                    _psi[2] = -np.pi/2
                elif observation[46] < -threshold:
                    _psi[2] = np.pi/2
                if observation[47] > threshold:
                    _psi[3] = -np.pi/2
                elif observation[47] < -threshold:
                    _psi[3] = np.pi/2
                
                action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
            
            delta = np.zeros(12)
            action_deltas = np.zeros(12)
            
        
        # print(f"mu: {_mu}, omega: {_omega}, psi: {_psi}")
        
        if array[0] > 10000:
            mu =  _mu
            omega = _omega
            psi = _psi
        else:
            omega = np.array([0.0, 0.0, 0.0, 0.0])
            mu = np.array([1.2, 1.2, 1.2, 1.2])
            psi = np.array([0.0, 0.0, 0.0, 0.0])
            
        
        
        
        if print_count % 50 == 0:
            clear_terminal()
            # print(f"joy0: {joy0}, joy1: {joy1}, joy2: {joy2}")
            print(f"Delta Policy:{delta_policy_active}")
            print(f"mode: {mode}, command(vx,vy,wz): {command[0], command[1], command[2]}, h: {h}, walk_gc: {walk_gc}, policy active: {policy_active}")
            print(f"delta: {delta[0], delta[1], delta[2], delta[3], delta[4], delta[5], delta[6], delta[7], delta[8], delta[9], delta[10], delta[11]}, gc: {gc}")
        print_count += 1
        
        
        response = struct.pack('d' * 27, *(mu.tolist() + omega.tolist() + psi.tolist() + delta.tolist() + [mode, h, gc] ))
        
        step += 1
       
        
        client_socket.send(response)
        client_socket.close()

class Box:
    def __init__(self, dim, low=None, high=None):
        self.low = low
        self.high = high
        self.shape = dim

if __name__ == "__main__":
    start_server()