# CityFlow-HiLight-HRL
A hierarchical reinforcement learning (HRL) framework for **multi-intersection traffic signal control** using the **HiLight** architecture, implemented and evaluated in the **CityFlow** simulator.  
This repository is built for research and experimentation with high-level (regional) and low-level (intersection) agent coordination for dynamic traffic signal optimization.

---

## Overview
CityFlow-HiLight-HRL provides a reproducible environment for running and extending the HiLight method inside **CityFlow** with **gym-cityflow** compatibility.  
All setup steps are performed inside a **safe, isolated Ubuntu (WSL2) environment** on Windows 11.

---

## Prerequisites
- Windows 11 (with admin access)  
- Internet connection (to fetch WSL + packages)  
- Git and Python ≥ 3.8 (installed within Ubuntu)  
- ~5 GB free disk space  

---

## Step 1 Enable WSL and Install Ubuntu

### 1.1 Open PowerShell as Administrator
Press `Start → type "PowerShell" → right click → "Run as administrator"`.

### 1.2 Install WSL with Ubuntu 20.04
Run:
```bash
wsl --install -d Ubuntu-20.04
```
> [!NOTE]
> If it’s slow or stalls at 0%, you can alternatively install **Ubuntu 22.04 LTS** from the **Microsoft Store** (same process). But for me I just right clicked after waiting 5 minutes and it moved on.
### 1.3 Wait for Installation

You’ll see messages like:

`Downloading Ubuntu 20.04 LTS... Installing, this may take a few minutes...`

This can take **5–15 minutes** depending on your connection and CPU speed.

### 1.4 Launch Ubuntu for the First Time

Once installed, search **“Ubuntu”** in the Start Menu and open it.  
You’ll be prompted to create a **Linux user**.

Follow the prompts:

`Enter new UNIX username: (e.g.) jonathan` 
`Enter new UNIX password:`
`Retype new UNIX password:`

⚠️ Password will not show any characters as you type, that’s normal.  
This username and password are only for your Ubuntu environment (not your Windows login).

When you’re done, you’ll see a prompt like:

`jonathan@DESKTOP-XXXX:~$`

That means you’re now inside your Linux shell.
## Step 2 Update Ubuntu and Install Build Tools

Keep your system up to date and install the required compilers/libraries for CityFlow.

`sudo apt update && sudo apt install -y build-essential cmake git python3-venv python3-pip`

## Step 3  Create a Dedicated Folder and Virtual Environment

We’ll isolate CityFlow inside a self-contained environment.

`mkdir ~/cityflow_env cd ~/cityflow_env python3 -m venv venv source venv/bin/activate`

Your command line should now start with `(venv)`:

`(venv) jonathan@DESKTOP:~/cityflow_env$`

## Step 4 Clone and Install CityFlow

CityFlow is the main traffic simulator used in this project.

`git clone https://github.com/cityflow-project/CityFlow.git cd CityFlow pip install .`

⏳ The first build will take several minutes as it compiles C++ extensions.

If you see an error like:

`error: invalid command 'bdist_wheel'`

it’s harmless it means `wheel` isn’t installed. CityFlow will still finish building.

To avoid this in future:

`pip install wheel`

> [!NOTE]
> I recommend rerunning the git clone command after you install wheel

## Step 5 Install gym-cityflow and Reinforcement Learning Dependencies

This integrates CityFlow with OpenAI Gym for RL experiments.

`cd .. git clone https://github.com/MaxVanDijck/gym-cityflow.git cd gym-cityflow pip install -e . pip install gym`

If you see a **Gym deprecation warning**, you can use its drop-in replacement:

`pip install gymnasium`

Then, in code:

`import gymnasium as gym`

## Step 6 Verify Installation

Let’s confirm everything installed correctly.

`python`

Inside Python:

```python
import cityflow
import gym
import gym_cityflow
print("✅ CityFlow and gym-cityflow installed successfully!") exit()
```

If no errors appear, you’re good to go.

## Step 7 Test CityFlow with a Sample Simulation

CityFlow comes with example configs to test your setup.

`cd ~/cityflow_env/CityFlow/examples python`

Inside Python:

``` python
import cityflow 
eng = cityflow.Engine("config.json", thread_num=1) 
for i in range(5):     
	eng.next_step() 
print("✅ CityFlow ran 5 simulation steps successfully!")
```

You should see the confirmation message without errors.

## Step 8 Access Ubuntu Files from Windows

Your Linux files are accessible through File Explorer:

`\\wsl$\Ubuntu-20.04\home\<your_username>\cityflow_env`

You can open this path in VS Code (with the Remote WSL extension) or drag files in/out just like normal.

If you ever want to see your Windows Desktop from Ubuntu:

`cd /mnt/c/Users/<YourWindowsUser>/Desktop`



## Reference:
https://medium.com/@EvanMath/how-to-install-the-cityflow-simulator-in-windows-11-a89ef4ea2397
