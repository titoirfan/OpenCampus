/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

//all foot position control

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>

#include <sys/time.h>

// **** ADD *************
#include <cmath>
#include <array>

// TCP
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

//*******************

struct timeval tbegin;
struct timeval tend;

using namespace std;
using namespace UNITREE_LEGGED_SDK;

class Custom
{
public:
    // Custom(): udp(LOW_CMD_LENGTH, LOW_STATE_LENGTH){}
    Custom(): control(LeggedType::A1, LOWLEVEL), udp() {
        control.InitCmdData(cmd);
    }
    void UDPRecv();
    void UDPSend();
    void RobotControl();
    void TCPClient();
    void CPG();
    std::array<double, 3> Trajectory(double y_bias, int leg);
    void Observations();
    std::array<double, 3> SaftyCheck(const std::array<double, 3>& leg_des_q, const std::array<double, 3>& leg_joint_q, const std::string& leg = "FR", double threshold = 0.5, double hip_max = 0.8, double hip_min = -0.8, double thigh_max = 3.927, double thigh_min = -0.524, double calf_max = -0.611, double calf_min = -2.775);

    Control control;
    UDP udp;
    LowCmd cmd = {0};
    LowState state = {0};


    double mode = 0.0;


    std::array<double, 3> FR_pos_init = {{0.0, -0.0838, -0.25}};
    std::array<double, 3> FR_pos_stop = {{0.0, -0.0838, -0.10}};
    std::array<double, 3> FR_pos_des = {{0.0, -0.0838, -0.25}};
    std::array<double, 3> FR_sin_mid_pos = {{0.0, -0.0838, -0.25}};
    std::array<double, 3> FR_pos = {{0.0, -0.0838, -0.25}};

    std::array<double, 3> FL_pos_init = {{0.0, 0.0838, -0.25}};
    std::array<double, 3> FL_pos_stop = {{0.0, 0.0838, -0.10}};
    std::array<double, 3> FL_pos_des = {{0.0, 0.0838, -0.25}};
    std::array<double, 3> FL_sin_mid_pos = {{0.0, 0.0838, -0.25}};
    std::array<double, 3> FL_pos = {{0.0, 0.0838, -0.25}};

    std::array<double, 3> RR_pos_init = {{0.0, -0.0838, -0.25}};
    std::array<double, 3> RR_pos_stop = {{0.0, -0.0838, -0.10}};
    std::array<double, 3> RR_pos_des = {{0.0, -0.0838, -0.25}};
    std::array<double, 3> RR_sin_mid_pos = {{0.0, -0.0838, -0.25}};
    std::array<double, 3> RR_pos = {{0.0, -0.0838, -0.25}};

    std::array<double, 3> RL_pos_init = {{0.0, 0.0838, -0.25}};
    std::array<double, 3> RL_pos_stop = {{0.0, 0.0838, -0.10}};
    std::array<double, 3> RL_pos_des = {{0.0, 0.0838, -0.25}};
    std::array<double, 3> RL_sin_mid_pos = {{0.0, 0.0838, -0.25}};
    std::array<double, 3> RL_pos = {{0.0, 0.0838, -0.25}};

    float Kp[3] = {0};  
    float Kd[3] = {0};
    double time_consume = 0;
    int rate_count = 0;
    int stop_rate_count = 0;
    int sin_count = 0;
    int motiontime = 0;
    int control_count = 0;
    int stop_count = 0;
    int gain_rate_count = 0;
    double dt = 0.001;

    // CPG parameters
    double gc = 0.1;
    double gp = 0.02;
    double h = 0.25;
    double d = 0.15;
    double a = 150;

    // CPG state
    // std::array<double, 4> mu = {{1.0, 1.0, 1.0, 1.0}};
    // std::array<double, 4> mu = {{1.50, 1.50, 1.50, 1.50}};
    std::array<double, 4> mu = {{1.20, 1.20, 1.20, 1.20}};
    std::array<double, 4> omega = {{0.00, 0.00, 0.00, 0.00}};
    std::array<double, 4> psi = {{0.00, 0.00, 0.00, 0.00}};

    // std::array<double, 4> r = {{1.00, 1.00, 1.00, 1.00}};
    // std::array<double, 4> r = {{1.50, 1.50, 1.50, 1.50}};
    std::array<double, 4> r = {{1.20, 1.20, 1.20, 1.20}};
    std::array<double, 4> r_dot = {{0.0, 0.0, 0.0, 0.0}};
    std::array<double, 4> theta = {{0.0, M_PI, M_PI, 0.0}};
    std::array<double, 4> phi = {{0.0, 0.0, 0.0, 0.0}};

    std::array<double, 12> _delta = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    std::array<double, 12> delta = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    std::array<double, 12> delta_dot = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    // Sensor data
    std::array<double, 12> joint_q = {{0.0, 0.8, -1.6, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6}};
    std::array<double, 12> joint_dq = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    std::array<double, 4> foot_force = {{0.0, 0.0, 0.0, 0.0}};
    std::array<double, 3> euler = {{0.0, 0.0, 0.0}};
    std::array<double, 3> angular_vel = {{0.0, 0.0, 0.0}};
    std::array<double, 3> linear_acc = {{0.0, 0.0, 0.0}};
        
};

void Custom::UDPRecv()
{  
    udp.Recv();
}

void Custom::UDPSend()
{  
    udp.Send();
}

void Custom::TCPClient() {
    int sock = 0;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "Socket creation error" << std::endl;
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(12345);

    if (inet_pton(AF_INET, "169.254.187.12", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported" << std::endl;
        return;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection Failed" << std::endl;
        return;
    }

    // Send data
    std::array<double, 61> data_to_send = {
        static_cast<double>(motiontime),
        joint_q[0], joint_q[1], joint_q[2], joint_q[3], joint_q[4], joint_q[5],
        joint_q[6], joint_q[7], joint_q[8], joint_q[9], joint_q[10], joint_q[11],
        joint_dq[0], joint_dq[1], joint_dq[2], joint_dq[3], joint_dq[4], joint_dq[5],
        joint_dq[6], joint_dq[7], joint_dq[8], joint_dq[9], joint_dq[10], joint_dq[11],
        foot_force[0], foot_force[1], foot_force[2], foot_force[3],
        euler[0], euler[1], //euler[2],
        angular_vel[0], angular_vel[1], angular_vel[2],
        linear_acc[0], linear_acc[1], linear_acc[2],
        r[0], r[1], r[2], r[3],
        theta[0], theta[1], theta[2], theta[3],
        phi[0], phi[1], phi[2], phi[3],
        r_dot[0], r_dot[1], r_dot[2], r_dot[3],
        omega[0], omega[1], omega[2], omega[3],
        psi[0], psi[1], psi[2], psi[3]
    };

    // std::cout << "Sending data of size: " << data_to_send.size() * sizeof(double) << " bytes" << std::endl;

    if (send(sock, data_to_send.data(), data_to_send.size() * sizeof(double), 0) < 0) {
        std::cerr << "Send failed" << std::endl;
        close(sock);
        return;
    }
    // std::cout << "Data sent" << std::endl;

    // Receive processed data
    double buffer[27] = {0};
    int valread = read(sock, buffer, sizeof(buffer));
    if (valread < 0) {
        std::cerr << "Read failed" << std::endl;
    }
    
    if (control_count >= 10000){
        mu = {buffer[0], buffer[1], buffer[2], buffer[3]};
        omega = {buffer[4], buffer[5], buffer[6], buffer[7]};
        psi = {buffer[8], buffer[9], buffer[10], buffer[11]};
        _delta = {buffer[12], buffer[13], buffer[14], buffer[15], buffer[16], buffer[17], buffer[18], buffer[19], buffer[20], buffer[21], buffer[22], buffer[23]};
        h = buffer[25];
        gc = buffer[26];
    }else{
        mu = {1.2, 1.2, 1.2, 1.2};
        omega = {0, 0, 0, 0};
        psi = {0, 0, 0, 0};
    }
    
    if (mode != buffer[24]){
        printf("mode: %.2f\n", buffer[24]);
    }

    mode = buffer[24];



    shutdown(sock, SHUT_RDWR);
    close(sock);
}

std::array<double, 3> FootPosLinearInterpolation(const std::array<double, 3>& initPos, const std::array<double, 3>& targetPos, double rate)
{
    std::array<double, 3> p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    for (int i = 0; i < 3; ++i) {
        p[i] = initPos[i] * (1 - rate) + targetPos[i] * rate;
    }
    return p;
}

std::array<double, 3> ForwardKinematics(const std::array<double, 3>& leg_joint_q, double L1 = 0.0838, double L2 = 0.2, double L3 = 0.2) {
    double th1 = leg_joint_q[0];
    double th2 = leg_joint_q[1];
    double th3 = leg_joint_q[2];

    double sin_th1 = std::sin(th1);
    double cos_th1 = std::cos(th1);
    double sin_th2 = std::sin(th2);
    double cos_th2 = std::cos(th2);
    double sin_th23 = std::sin(th2 + th3);
    double cos_th23 = std::cos(th2 + th3);

    double x = -L3 * sin_th23 - L2 * sin_th2;
    double y = L3 * sin_th1 * cos_th23 + L2 * sin_th1 * cos_th2 - L1 * cos_th1;
    double z = -L3 * cos_th1 * cos_th23 - L2 * cos_th1 * cos_th2 - L1 * sin_th1;

    return {x, y, z};
}

std::array<double, 3> InverseKinematics(const std::array<double, 3>& target_position, const std::string& LR = "R")
{
    double L1 = 0.0838;
    double L2 = 0.2;
    double L3 = 0.2;

    double th1, th_f_yz;

    // printf("target_position: %.2f %.2f %.2f\n", target_position[0], target_position[1], target_position[2]);
    // printf("LR: %s\n", LR.c_str());

    if (LR == "R") {
        th_f_yz = std::atan2(-target_position[2], -target_position[1]);
        th1 = th_f_yz - std::acos(L1 / std::sqrt(target_position[1] * target_position[1] + target_position[2] * target_position[2]));
    } else if (LR == "L") {
        th_f_yz = std::atan2(target_position[2], target_position[1]);
        th1 = th_f_yz + std::acos(L1 / std::sqrt(target_position[1] * target_position[1] + target_position[2] * target_position[2]));
    } else {
        throw std::invalid_argument("Invalid value for LR. Use 'R' or 'L'.");
    }

    std::array<std::array<double, 3>, 3> R_x = {{
        {1, 0, 0},
        {0, std::cos(th1), -std::sin(th1)},
        {0, std::sin(th1), std::cos(th1)}
    }};

    std::array<std::array<double, 3>, 3> R_x_inv;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_x_inv[i][j] = R_x[j][i];
        }
    }

    std::array<double, 3> rotated_target_pos = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            rotated_target_pos[i] += R_x_inv[i][j] * target_position[j];
        }
    }

    double phi = std::acos((L2 * L2 + L3 * L3 - rotated_target_pos[0] * rotated_target_pos[0] - rotated_target_pos[2] * rotated_target_pos[2]) / (2 * L2 * L3));
    double th_f_xz = std::atan2(-rotated_target_pos[0], -rotated_target_pos[2]);

    double th2 = th_f_xz + (M_PI - phi) / 2;
    double th3 = -M_PI + phi;

    return {th1, th2, th3};
}

void Custom::CPG() 
{
    for (size_t i = 0; i < r.size(); ++i) {
        r_dot[i] = a * (a * (mu[i] - r[i]) / 4 - r_dot[i]) * dt;
        r[i] = r_dot[i] * dt + r[i];
        theta[i] = omega[i] * dt + theta[i];
        phi[i] = psi[i] * dt + phi[i];
    }
    for (size_t i = 0; i < delta.size(); ++i) {
        delta_dot[i] = a * (a * (_delta[i] - delta[i]) / 4 - delta_dot[i]) * dt;
        delta[i] = delta_dot[i] * dt + delta[i];
    }
}

std::array<double, 3> Custom::Trajectory(double y_bias, int leg)
{
    std::array<double, 3> pos;
    double g = (std::sin(theta[leg]) > 0) ? gc : gp;
    pos[0] = -d * (r[leg] - 1) * std::cos(theta[leg]) * std::cos(phi[leg]);
    pos[1] = -d * (r[leg] - 1) * std::cos(theta[leg]) * std::sin(phi[leg]) + y_bias;
    pos[2] = -h + g * std::sin(theta[leg]);
    return pos;
}

void Custom::Observations()
{
    joint_q = {{state.motorState[FR_0].q, state.motorState[FR_1].q, state.motorState[FR_2].q,
    state.motorState[FL_0].q, state.motorState[FL_1].q, state.motorState[FL_2].q, 
    state.motorState[RR_0].q, state.motorState[RR_1].q, state.motorState[RR_2].q, 
    state.motorState[RL_0].q, state.motorState[RL_1].q, state.motorState[RL_2].q}};

    joint_dq = {{state.motorState[FR_0].dq, state.motorState[FR_1].dq, state.motorState[FR_2].dq, 
    state.motorState[FL_0].dq, state.motorState[FL_1].dq, state.motorState[FL_2].dq, 
    state.motorState[RR_0].dq, state.motorState[RR_1].dq, state.motorState[RR_2].dq, 
    state.motorState[RL_0].dq, state.motorState[RL_1].dq, state.motorState[RL_2].dq}};

    foot_force = {{
        state.footForce[FR_] > 0 ? 1.0 : 0.0,
        state.footForce[FL_] > 0 ? 1.0 : 0.0,
        state.footForce[RR_] > 0 ? 1.0 : 0.0,
        state.footForce[RL_] > 0 ? 1.0 : 0.0
    }};
    euler = {{state.imu.rpy[0], state.imu.rpy[1], state.imu.rpy[2]}};
    angular_vel = {{state.imu.gyroscope[0], state.imu.gyroscope[1], state.imu.gyroscope[2]}};
    linear_acc = {{state.imu.accelerometer[0], state.imu.accelerometer[1], state.imu.accelerometer[2]}};
}

std::array<double, 3> Custom::SaftyCheck(const std::array<double, 3>& leg_des_q, const std::array<double, 3>& leg_joint_q, const std::string& leg, double threshold, double hip_max, double hip_min, double thigh_max, double thigh_min, double calf_max, double calf_min)
{
    std::array<double, 3> leg_des_q_checked = leg_des_q;

    if (leg_des_q[0] - leg_joint_q[0] > threshold)
    {
        leg_des_q_checked[0] = leg_joint_q[0] + threshold;
        printf("des_q[%s_0] - joint_q[%s_0] is too large\n", leg.c_str(), leg.c_str());
    }
    else if (leg_des_q[0] - leg_joint_q[0] < -threshold)
    {
        leg_des_q_checked[0] = leg_joint_q[0] - threshold;
        printf("des_q[%s_0] - joint_q[%s_0] is too small\n", leg.c_str(), leg.c_str());
    }

    if (leg_des_q[1] - leg_joint_q[1] > threshold)
    {
        leg_des_q_checked[1] = leg_joint_q[1] + threshold;
        printf("des_q[%s_1] - joint_q[%s_1] is too large\n", leg.c_str(), leg.c_str());
    }
    else if (leg_des_q[1] - leg_joint_q[1] < -threshold)
    {
        leg_des_q_checked[1] = leg_joint_q[1] - threshold;
        printf("des_q[%s_1] - joint_q[%s_1] is too small\n", leg.c_str(), leg.c_str());
    }

    if (leg_des_q[2] - leg_joint_q[2] > threshold)
    {
        leg_des_q_checked[2] = leg_joint_q[2] + threshold;
        printf("des_q[%s_2] - joint_q[%s_2] is too large\n", leg.c_str(), leg.c_str());
    }
    else if (leg_des_q[2] - leg_joint_q[2] < -threshold)
    {
        leg_des_q_checked[2] = leg_joint_q[2] - threshold;
        printf("des_q[%s_2] - joint_q[%s_2] is too small\n", leg.c_str(), leg.c_str());
    }

    if (leg_des_q_checked[0] > hip_max)
    {
        leg_des_q_checked[0] = hip_max;
        printf("des_q[%s_0] is too large\n", leg.c_str());
    }
    else if (leg_des_q_checked[0] < hip_min)
    {
        leg_des_q_checked[0] = hip_min;
        printf("des_q[%s_0] is too small\n", leg.c_str());
    }

    if (leg_des_q_checked[1] > thigh_max)
    {
        leg_des_q_checked[1] = thigh_max;
        printf("des_q[%s_1] is too large\n", leg.c_str());
    }
    else if (leg_des_q_checked[1] < thigh_min)
    {
        leg_des_q_checked[1] = thigh_min;
        printf("des_q[%s_1] is too small\n", leg.c_str());
    }

    if (leg_des_q_checked[2] > calf_max)
    {
        leg_des_q_checked[2] = calf_max;
        printf("des_q[%s_2] is too large\n", leg.c_str());
    }
    else if (leg_des_q_checked[2] < calf_min)
    {
        leg_des_q_checked[2] = calf_min;
        printf("des_q[%s_2] is too small\n", leg.c_str());
    }

    return leg_des_q_checked;
}

void Custom::RobotControl() 
{
    motiontime++;
    udp.GetRecv(state);
    Observations();

    
    std::array<double, 3> FR_joint_q = {state.motorState[FR_0].q, state.motorState[FR_1].q, state.motorState[FR_2].q};
    std::array<double, 3> FL_joint_q = {state.motorState[FL_0].q, state.motorState[FL_1].q, state.motorState[FL_2].q};
    std::array<double, 3> RR_joint_q = {state.motorState[RR_0].q, state.motorState[RR_1].q, state.motorState[RR_2].q};
    std::array<double, 3> RL_joint_q = {state.motorState[RL_0].q, state.motorState[RL_1].q, state.motorState[RL_2].q};
    
    if( motiontime >= 0){

        CPG();

        if (mode == 0.0){
            
            if (stop_count < 10){
                FR_pos_init = ForwardKinematics(FR_joint_q,0.0838);
                FL_pos_init = ForwardKinematics(FL_joint_q,-0.0838);
                RR_pos_init = ForwardKinematics(RR_joint_q,0.0838);
                RL_pos_init = ForwardKinematics(RL_joint_q,-0.0838);
                if (motiontime < 100){
                    Kp[0] = 0.0; Kp[1] = 0.0; Kp[2] = 0.0;  
                    Kd[0] = 0.0; Kd[1] = 0.0; Kd[2] = 0.0;
                    
                }
                else{
                    Kp[0] = 100.0; Kp[1] = 100.0; Kp[2] = 100.0;  
                    Kd[0] = 2.0; Kd[1] = 2.0; Kd[2] = 2.0;
                }
            }
            if (stop_count >= 10 && stop_count < 5000){
                stop_rate_count++;
                double rate = stop_rate_count/4000.0;  // needs count to 4000
                FR_pos_des = FootPosLinearInterpolation(FR_pos_init, FR_pos_stop, rate);
                FL_pos_des = FootPosLinearInterpolation(FL_pos_init, FL_pos_stop, rate);
                RR_pos_des = FootPosLinearInterpolation(RR_pos_init, RR_pos_stop, rate);
                RL_pos_des = FootPosLinearInterpolation(RL_pos_init, RL_pos_stop, rate);
                Kp[0] = 100.0; Kp[1] = 100.0; Kp[2] = 100.0;  
                Kd[0] = 2.0; Kd[1] = 2.0; Kd[2] = 2.0;
                gain_rate_count = 0;
            }
            if (stop_count >= 5000){
                stop_rate_count = 0;
                gain_rate_count++;
                double rate = std::min(std::max(gain_rate_count/3000.0, 0.0), 1.0);
                Kp[0] = 100.0 *(1-rate); Kp[1] = 100.0 *(1-rate); Kp[2] = 100.0 *(1-rate); 
                Kd[0] = 2.0 *(1-rate); Kd[1] = 2.0 *(1-rate); Kd[2] = 2.0 *(1-rate);
            }

            stop_count++;
            control_count = 0;
            
        }
        
        if (mode == 1.0){

            if( control_count >= 0 && control_count < 10){
                
                FR_pos_init = ForwardKinematics(FR_joint_q,0.0838);
                FL_pos_init = ForwardKinematics(FL_joint_q,-0.0838);
                RR_pos_init = ForwardKinematics(RR_joint_q,0.0838);
                RL_pos_init = ForwardKinematics(RL_joint_q,-0.0838);
                Kp[0] = 0.0; Kp[1] = 0.0; Kp[2] = 0.0; 
                Kd[0] = 0.0; Kd[1] = 0.0; Kd[2] = 0.0;
                
            }

            if( control_count >= 10 && control_count < 10000){
                rate_count++;
                double rate = rate_count/9000.0;  // needs count to 200

                Kp[0] = 100.0; Kp[1] = 100.0; Kp[2] = 100.0; 
                Kd[0] = 2.0; Kd[1] = 2.0; Kd[2] = 2.0;

                std::array<double, 3> _FR_pos_des = Trajectory(-0.0838, 0);
                std::array<double, 3> _FL_pos_des = Trajectory(0.0838, 1);
                std::array<double, 3> _RR_pos_des = Trajectory(-0.0838, 2);
                std::array<double, 3> _RL_pos_des = Trajectory(0.0838, 3);

                FR_pos_des = FootPosLinearInterpolation(FR_pos_init, _FR_pos_des, rate);
                FL_pos_des = FootPosLinearInterpolation(FL_pos_init, _FL_pos_des, rate);
                RR_pos_des = FootPosLinearInterpolation(RR_pos_init, _RR_pos_des, rate);
                RL_pos_des = FootPosLinearInterpolation(RL_pos_init, _RL_pos_des, rate);
            }

            if( control_count >= 10000){

                FR_pos_des = Trajectory(-0.0838, 0);
                FL_pos_des = Trajectory(0.0838, 1);
                RR_pos_des = Trajectory(-0.0838, 2);
                RL_pos_des = Trajectory(0.0838, 3);

                FR_pos_des[0] += delta[0];
                FR_pos_des[1] += delta[1];
                FR_pos_des[2] += delta[2];

                FL_pos_des[0] += delta[3];
                FL_pos_des[1] += delta[4];
                FL_pos_des[2] += delta[5];

                RR_pos_des[0] += delta[6];
                RR_pos_des[1] += delta[7];
                RR_pos_des[2] += delta[8];

                RL_pos_des[0] += delta[9];
                RL_pos_des[1] += delta[10];
                RL_pos_des[2] += delta[11];


                Kp[0] = 100.0; Kp[1] = 100.0; Kp[2] = 100.0;  
                Kd[0] = 2.0; Kd[1] = 2.0; Kd[2] = 2.0;

                rate_count = 0;
            }
            
            control_count++;
            stop_count = 0;
        }

        


        std::array<double, 3> FR_q_des = InverseKinematics(FR_pos_des, "R");
        std::array<double, 3> FL_q_des = InverseKinematics(FL_pos_des, "L");
        std::array<double, 3> RR_q_des = InverseKinematics(RR_pos_des, "R");
        std::array<double, 3> RL_q_des = InverseKinematics(RL_pos_des, "L");

        double threshold = 0.4;
        FR_q_des = Custom::SaftyCheck(FR_q_des, FR_joint_q, "FR", threshold);
        FL_q_des = Custom::SaftyCheck(FL_q_des, FL_joint_q, "FL", threshold);
        RR_q_des = Custom::SaftyCheck(RR_q_des, RR_joint_q, "RR", threshold);
        RL_q_des = Custom::SaftyCheck(RL_q_des, RL_joint_q, "RL", threshold);


        cmd.motorCmd[FR_0].q = FR_q_des[0];
        cmd.motorCmd[FR_0].dq = 0;
        cmd.motorCmd[FR_0].Kp = Kp[0];
        cmd.motorCmd[FR_0].Kd = Kd[0];
        cmd.motorCmd[FR_0].tau = 0.0f;

        cmd.motorCmd[FR_1].q = FR_q_des[1];
        cmd.motorCmd[FR_1].dq = 0;
        cmd.motorCmd[FR_1].Kp = Kp[1];
        cmd.motorCmd[FR_1].Kd = Kd[1];
        cmd.motorCmd[FR_1].tau = 0.0f;

        cmd.motorCmd[FR_2].q =  FR_q_des[2];
        cmd.motorCmd[FR_2].dq = 0;
        cmd.motorCmd[FR_2].Kp = Kp[2];
        cmd.motorCmd[FR_2].Kd = Kd[2];
        cmd.motorCmd[FR_2].tau = 0.0f;

        cmd.motorCmd[FL_0].q = FL_q_des[0];
        cmd.motorCmd[FL_0].dq = 0;
        cmd.motorCmd[FL_0].Kp = Kp[0];
        cmd.motorCmd[FL_0].Kd = Kd[0];
        cmd.motorCmd[FL_0].tau = 0.0f;

        cmd.motorCmd[FL_1].q = FL_q_des[1];
        cmd.motorCmd[FL_1].dq = 0;
        cmd.motorCmd[FL_1].Kp = Kp[1];
        cmd.motorCmd[FL_1].Kd = Kd[1];
        cmd.motorCmd[FL_1].tau = 0.0f;

        cmd.motorCmd[FL_2].q =  FL_q_des[2];
        cmd.motorCmd[FL_2].dq = 0;
        cmd.motorCmd[FL_2].Kp = Kp[2];
        cmd.motorCmd[FL_2].Kd = Kd[2];
        cmd.motorCmd[FL_2].tau = 0.0f;

        cmd.motorCmd[RR_0].q = RR_q_des[0];
        cmd.motorCmd[RR_0].dq = 0;
        cmd.motorCmd[RR_0].Kp = Kp[0];
        cmd.motorCmd[RR_0].Kd = Kd[0];
        cmd.motorCmd[RR_0].tau = 0.0f;

        cmd.motorCmd[RR_1].q = RR_q_des[1];
        cmd.motorCmd[RR_1].dq = 0;
        cmd.motorCmd[RR_1].Kp = Kp[1];
        cmd.motorCmd[RR_1].Kd = Kd[1];
        cmd.motorCmd[RR_1].tau = 0.0f;

        cmd.motorCmd[RR_2].q =  RR_q_des[2];
        cmd.motorCmd[RR_2].dq = 0;
        cmd.motorCmd[RR_2].Kp = Kp[2];
        cmd.motorCmd[RR_2].Kd = Kd[2];
        cmd.motorCmd[RR_2].tau = 0.0f;

        cmd.motorCmd[RL_0].q = RL_q_des[0];
        cmd.motorCmd[RL_0].dq = 0;
        cmd.motorCmd[RL_0].Kp = Kp[0];
        cmd.motorCmd[RL_0].Kd = Kd[0];
        cmd.motorCmd[RL_0].tau = 0.0f;

        cmd.motorCmd[RL_1].q = RL_q_des[1];
        cmd.motorCmd[RL_1].dq = 0;
        cmd.motorCmd[RL_1].Kp = Kp[1];
        cmd.motorCmd[RL_1].Kd = Kd[1];
        cmd.motorCmd[RL_1].tau = 0.0f;

        cmd.motorCmd[RL_2].q =  RL_q_des[2];
        cmd.motorCmd[RL_2].dq = 0;
        cmd.motorCmd[RL_2].Kp = Kp[2];
        cmd.motorCmd[RL_2].Kd = Kd[2];
        cmd.motorCmd[RL_2].tau = 0.0f;

    }


    udp.SetSend(cmd);

}


int main(void)
{   
    Custom custom;
    
    LoopFunc loop_control("control_loop", 0.001, 1, boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udpRecv("udp_recv", 0.001, 2, boost::bind(&Custom::UDPRecv, &custom));
    LoopFunc loop_udpSend("udp_send", 0.001, 3, boost::bind(&Custom::UDPSend, &custom));
    LoopFunc loop_tcp("tcp", 0.01, 3, boost::bind(&Custom::TCPClient, &custom));

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();
    loop_tcp.start();

    while(1){
        sleep(10);
    };

    return 0; 
}
