🚂 This code is based on the official starter kit - NeurIPS 2020 Flatland Challenge
---


You can using for your own experiments 
```python
set_action_size_full()
```
or 
```python
set_action_size_reduced()
```
action space. The reduced action space removes DO_NOTHING. 

---
Have a look into the [run.py](.\run.py) file. There you can select using PPO or DDDQN as RL agents
 
```python
####################################################
# EVALUATION PARAMETERS
set_action_size_full()

# Print per-step logs
VERBOSE = True
USE_FAST_TREEOBS = True

if False:
    # -------------------------------------------------------------------------------------------------------
    # RL solution
    # -------------------------------------------------------------------------------------------------------
    # 116591 adrian_egli
    # graded	71.305	0.633	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/51
    # Fri, 22 Jan 2021 23:37:56
    set_action_size_reduced()
    load_policy = "DDDQN"
    checkpoint = "./checkpoints/210122120236-3000.pth"  # 17.011131341978228
    EPSILON = 0.0

if False:
    # -------------------------------------------------------------------------------------------------------
    # RL solution
    # -------------------------------------------------------------------------------------------------------
    # 116658 adrian_egli
    # graded	73.821	0.655	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/52
    # Sat, 23 Jan 2021 07:41:35
    set_action_size_reduced()
    load_policy = "PPO"
    checkpoint = "./checkpoints/210122235754-5000.pth"  # 16.00113400887389
    EPSILON = 0.0

if True:
    # -------------------------------------------------------------------------------------------------------
    # RL solution
    # -------------------------------------------------------------------------------------------------------
    # 116659 adrian_egli
    # graded	80.579	0.715	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/53
    # Sat, 23 Jan 2021 07:45:49
    set_action_size_reduced()
    load_policy = "DDDQN"
    checkpoint = "./checkpoints/210122165109-5000.pth"  # 17.993750197899438
    EPSILON = 0.0

if False:
    # -------------------------------------------------------------------------------------------------------
    # !! This is not a RL solution !!!!
    # -------------------------------------------------------------------------------------------------------
    # 116727 adrian_egli
    # graded	106.786	0.768	RL	Successfully Graded ! More details about this submission can be found at:
    # http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/54
    # Sat, 23 Jan 2021 14:31:50
    set_action_size_reduced()
    load_policy = "DeadLockAvoidance"
    checkpoint = None
    EPSILON = 0.0
```

---
A deadlock avoidance agent is implemented. The agent only lets the train take the shortest route. And it tries to avoid as many deadlocks as possible.
* [dead_lock_avoidance_agent.py](.\utils\dead_lock_avoidance_agent.py)


---
The policy interface has changed, please have a look into 

---
See the tensorboard training output to get some insights:
```
tensorboard --logdir ./runs_bench 
```

---
If you have any questions write me on the official discord channel **aiAdrian**    
(Adrian Egli - adrian.egli@gmail.com) 

Main links
---

* [Flatland documentation](https://flatland.aicrowd.com/)
* [Flatland Challenge](https://www.aicrowd.com/challenges/flatland)

Communication
---

* [Discord Channel](https://discord.com/invite/hCR3CZG)
* [Discussion Forum](https://discourse.aicrowd.com/c/neurips-2020-flatland-challenge)
* [Issue Tracker](https://gitlab.aicrowd.com/flatland/flatland/issues/)