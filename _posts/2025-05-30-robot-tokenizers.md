---
layout: post
title: "Overview of robots learning actions"
subtitle: "from language to action tokens"
date: 2025-05-30 23:45:13 -0400
background: '/assets/img/boulder-bg.jpg'
---

# What puts the 'A' in VLA?

There's been an algorithmic shift towards VLA (Vision Language Action) models in robotics. The idea behind them is that we can re-use the visual world understanding of VLMs (Vision Language Models) and teach them to speak robot actions.

I don't have a robotics background so all of this is quite new to me, and I've always been quite miffed by the bridging of what must happen between a model outputting tokens to an actual arm flipping a pancake. So what are robot actions?

## How do tokens map to motor actions

For beginners (like myself) Hugging Face's [LeRobot project](https://github.com/huggingface/lerobot) was a great starting point for learning the essentials of robotics ML. You can order a kit and 3D print yourself some parts to screw onto the motors, and alltogether you have a pair of arms for ~$300-$600 (depending on which kit you go for). I went for the the [koch v1.1](https://github.com/jess-moss/koch-v1-1) which is based around dynamixel motors.

### Generating movement

Think of a single motor as a thing that can turn. The kit comes with 6 motors per arm, you daisy chain them together on a single serial bus then you can use the same interface to speak to each motor with its assigned ID.

![](/assets/img/posts/robot_tokenizers/single_motor.png)

Note that under the hood you don't actually write degrees to these things, but each motor will typically have a fixed resolution. The Dynamixel XL330's have 12bit encoders which means there's 4096 possible position values you can write to them.

Below is a Python snippet to get an appreciation for how it works for a single motor. I use this to move my gripper up 45 degrees.

```python
from dynamixel_sdk import PortHandler, PacketHandler

protocol = 2.0 # seems to be default
device = '/dev/ttyACM1' # usb device connected to dynamixel motors
motor_id = 6 # I set this as the gripper ID
baudrate = 1_000_000 # Default in lerobot

# Addresses from 
# https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/#control-table
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# Boilerplate communication setup
portHandler = PortHandler(device)
packetHandler = PacketHandler(protocol)
portHandler.openPort()
portHandler.setBaudRate(baudrate)

# Read current position of motor
current_position, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_PRESENT_POSITION)

# Calculate goal position by adding how much we want to rotate
goal_position = current_position + int((45/360) * 4095) # move 45 degrees

# The bit that makes the motor move
packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_GOAL_POSITION, goal_position)
```

![](/assets/img/posts/robot_tokenizers/real_movement.png)

### Moving together

Hugging Face LeRobot examples and code snippets controlling dynamixel motor. Policy output is in range of [-180,180], for example action chunking transformer, then it gets converted to motor resolution, e.g. if 4096 is max. That drives motors 

![](/assets/img/posts/robot_tokenizers/arm_overview.png)

Here's an example of running the lerobot APIs to do teleop:

```python
while True:
    action = teleop_device.get_action()
    robot.send_action(action)
```

Where your actions are simply dicts containing the robot joint positions (or motor states normalized to [-100,100]).

```python
print(teleop_device.get_action())
# Output:
{'shoulder_pan.pos': 4.322344322344335,
 'shoulder_lift.pos': -99.59514170040485,
 'elbow_flex.pos': -90.30362389813908,
 'wrist_flex.pos': 4.133685136323663,
 'wrist_roll.pos': 4.224664224664238,
 'gripper.pos': 49.2676431424767}
```

I was a bit shocked how simple moving the robot really was. You just write numbers and the arm moves. The challenge is the smarts -- how do you make those movements autonomous, smooth, precise and safe. Each concern being a huge topic of research!

Solving robot intelligence or "physical AI" has always been a challenging area. Note that I'm writing floating point numbers above, not discrete tokens, and for each degree of freedom you add a motor -- it's a huge search space for next actions. Availability of training data also hampers progress because robots come in various embodiments: single arm, bimanual, humanoid, dog, etc. Sensors are part of the embodiment and just switching from one camera to two changes your embodiment and you need to either finetune or re-train your policy.

**However**, recently there's been a lot of really cool developments in VLA land, like pre-trained base robotics models, using diffusion to plan trajectories and new tokenization techniques.

## From VLM to VLA

Lets take a moment to discern between a typical VLM and VLA setup.

### VLM

VLMs are a multimodal model that has been trained on image & text data simultaneously. The images will go through some image embedding step which will convert it to image tokens which the base transformer model will process alongside text tokens.

The result -- a model that you can feed a picture to and it will be able to answer questions about the contents. You can repurpose these for a variety of tasks, like segmentation, classification, etc.

![](/assets/img/posts/robot_tokenizers/vlm_overview.png)

These have become the backbone of training VLA models because of their world understanding capabilities. Though some might argue present-day VLMs are not great are understanding the physical world and there is still a looot of work to be done in 3D world understanding which is essential for robotics.

### VLA

VLAs add an additional modality to VLMs and have to output actions to control the various moving bits of a robot -- joints, grippers, wheels, etc. 

This adds an additional input, scarily named "proprioception", which generally just means the robot state. Usually you can just query your robot motors via API call and find out what position it's in, i.e. how much it has rotated. And the output of these models will be the next action (or set of actions) to be performed by the robot, which, again, will just be motor instructions.

![](/assets/img/posts/robot_tokenizers/vla_overview.png)

## How has action tokenization evolved

Lets do a little history overview.

### Language instead of tokens

### PaLM-E

When LLMs took over people tried to apply these text generators via prompting and finetuning for a variety of tasks. One was to use LLMs to output low-level instructions for a robot to execute. However since these weren't actual motor commands, models like PaLM-E had to be supplemented by an additional controller module that would translate instructions to motor commands.

### RT-1 / RT-2
The first VLAs we've seen came from Google. RT-1 was trained from scratch on an internal dataset of 130k demonstrations. RT-2 is the first time we saw this transfer learning of webscale data from a pre-trained VLM take place to a robotics policy.

RT-2 had multiple variants based on different VLMs, namely PaLI-X and PaLM-E. 

RT-1 - Each dimension is discretized into 256 bins
* Seven for arm (x, y, z, roll, pitch, yaw, opening of the gripper)
* Three for base (x, y, yaw)
* 1 discrete variable for terminating ep (don't think this gets binned)

RT-2 - Vocab expansion to include actions next to text tokens (like RT-1)
* Everything discretized to 256 bins, except the special termination signal
* Action can be represented as 8 integer numbers
* Need to reserve 256 tokens as special action tokens
* PaLI-X already has 1000 tokens as integers, so they reuse those tokens. For PaLM-E they overwrite the 256 least used tokens (otherwise you need to change arch) - this is a form of symbol tuning.
* At inference output vocab is constrained to only sample from action tokens.

OpenVLA also had this approach of replacing the vocab.

### Diffusion for continous output action space
Started seeing action experts attached to VLMs, Octo, GR00T, Pi0. Diffusion works better than autoregressive approach for continuous motions.

### Pi0-FAST
First time we're seeing robot-specific tokenizer. Very excited about this.

Honestly I just wanted to do a little writeup of the FAST tokenizer



<!-- 
Currently I have a [koch v1.1](https://github.com/jess-moss/koch-v1-1) clamped onto my desk, but there's cheaper ones ones like the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) available as well.


# Transformers as robotics policies

In robotics there is an exciting new paradigm shift towards VLAs (Vision Language Action) models. This is transformer models learning to output commands directly into the motor controllers based on visual and language inputs. VLAs are hot now but I also like some of the earlier works on action cloning - these typically don't have a language instruction they just learn to do 1 task well. HuggingFace's LeRobot has some really nice examples of this with my favorite being the ACT policy, since it's tiny and easy to grok.  -->