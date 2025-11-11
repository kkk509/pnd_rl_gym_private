# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


from pnd_humanoid_robot_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class pnd_humanoid_robot_adam_Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 8192
        num_observations = 202 #175
        num_actions = 23
    
    class terrain( LeggedRobotCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
    class commands( LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.0, 0.0] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0, 0]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.92] # x,y,z [m]
        default_joint_angles = { 
            'hipPitch_Left': -0.0,
            'hipRoll_Left': -0.085,
            'hipYaw_Left': -0.0,
            'kneePitch_Left': 0.9,
            'anklePitch_Left': -0.0,
            'ankleRoll_Left': 0.002,

            'hipPitch_Right': -0.0,
            'hipRoll_Right': 0.085,
            'hipYaw_Right': 0.0,
            'kneePitch_Right': 0.9,
            'anklePitch_Right': -0.0,
            'ankleRoll_Right': -0.002,

            'waistRoll': 0.0,
            'waistPitch': 0.0,
            'waistYaw': 0.0,

            'shoulderPitch_Left':0.0,
            'shoulderRoll_Left':0.,
            'shoulderYaw_Left':0.0,
            'elbow_Left':-0.3,


            'shoulderPitch_Right':0.0,
            'shoulderRoll_Right':-0.,
            'shoulderYaw_Right':0.0,
            'elbow_Right':-0.3
        }

    class control( LeggedRobotCfg.control ):
        stiffness = {   'hipPitch': 305., 'hipRoll': 700.0, 'hipYaw': 405.0,'kneePitch': 305., 'anklePitch': 20.0,'ankleRoll': 0.,
                        'waistRoll': 405.0, 'waistPitch': 405.0, 'waistYaw': 205.0,
                        'shoulderPitch':18.0,  'shoulderRoll':9.0,  'shoulderYaw':9.0, 'elbow':9.0
                    }  # [N*m/rad]
        damping = { 'hipPitch': 6.1, 'hipRoll': 30.0, 'hipYaw': 6.1,'kneePitch': 6.1, 'anklePitch': 2.5,'ankleRoll': 0.35,
                    'waistRoll': 6.1, 'waistPitch': 6.1, 'waistYaw': 4.1,
                    'shoulderPitch':0.9,  'shoulderRoll':0.9,  'shoulderYaw':0.9, 'elbow':0.9
                    }  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2
        
    class asset( LeggedRobotCfg.asset ):
        file = '{PND_HUMANOID_ROBOT_ROOT_DIR}/resources/robots/adam/urdf/adam.urdf'
        name = "adam"
        foot_name = 'toe'
        thigh_name = 'thigh'
        shin_name = 'shin'
        torso_name = 'torso'
        upper_arm_name = 'shoulderYaw'
        lower_arm_name = 'elbow'

        terminate_after_contacts_on = ['pelvis','thigh','shoulder','elbow']
        flip_visual_attachments = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.8
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 500.
        only_positive_rewards = False
        base_height_target = 0.90
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 3.0
            torques = -35.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = -2
            dof_pos_limits = -1.
            no_fly = 0.8
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            arm_pose = -0.3
            torso_yaw = -0.3
            torso_orientation_diff_osu = -0.8
            torso_ang_vel_xy_osu = -0.3

    class noise:
        add_noise = False
        noise_level = 0.1 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1    
        
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
        clip_actions = 100.0

    class sim(LeggedRobotCfg.sim):
        dt =  0.005
    


class pnd_humanoid_robot_adam_Cfg_PPO( LeggedRobotCfgPPO):
    
    class runner( LeggedRobotCfgPPO.runner ):
        # policy_class_name = 'ActorCriticRecurrent'
        num_steps_per_env = 64#100 # per iteration
        run_name = ''
        experiment_name = 'pnd_humanoid_robot_adam'
        max_iterations = 4000 # number of policy updates
        save_interval = 200 # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class algorithm( LeggedRobotCfgPPO.algorithm):
        # training params
        num_learning_epochs = 8
        num_mini_batches = 40 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #5.e-4



  
