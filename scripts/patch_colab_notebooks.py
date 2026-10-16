"""Patch Colab notebooks to support private repository access."""
import json
from pathlib import Path

# 1. Update meetra_core_colab.ipynb
colab_path = Path("meetra_core_colab.ipynb")
if colab_path.exists():
    with open(colab_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and any("git clone" in line for line in cell.get("source", [])):
            cell["source"] = [
                "# 1. Check GPU Availability\n",
                "!nvidia-smi\n",
                "\n",
                "# 2. Clone the Meetra Core repository (Private or Public)\n",
                "import os\n",
                "from getpass import getpass\n",
                "\n",
                "if not os.path.exists('meetra_core'):\n",
                "    token = None\n",
                "    try:\n",
                "        from google.colab import userdata\n",
                "        token = userdata.get('GITHUB_TOKEN')\n",
                "    except Exception:\n",
                "        pass\n",
                "    ret = os.system('git clone https://github.com/GLITXHY9999/meetra_core.git')\n",
                "    if ret != 0:\n",
                "        print('\\n[!] Private repository detected. Enter your GitHub Personal Access Token:')\n",
                "        print('    (You can create one at: https://github.com/settings/tokens)')\n",
                "        if not token:\n",
                "            token = getpass('GitHub Personal Access Token: ').strip()\n",
                "        !git clone https://{token}@github.com/GLITXHY9999/meetra_core.git\n",
                "    %cd meetra_core\n",
                "else:\n",
                "    %cd meetra_core\n",
                "    !git pull origin main\n",
                "\n",
                "print('\\n[+] Current Working Directory:', os.getcwd())\n"
            ]
            break

    with open(colab_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Updated meetra_core_colab.ipynb successfully!")

# 2. Update Experiment_1_2_EDA.ipynb
eda_path = Path("Experiment_1_2_EDA.ipynb")
if eda_path.exists():
    with open(eda_path, "r", encoding="utf-8") as f:
        eda_nb = json.load(f)

    for cell in eda_nb["cells"]:
        if cell["cell_type"] == "code" and any("employee_data.csv" in line for line in cell.get("source", [])):
            cell["source"] = [
                "import os\n",
                "import pandas as pd\n",
                "\n",
                "# Load dataset (handles local workspace, Colab upload, and GitHub raw)\n",
                "if os.path.exists('employee_data.csv'):\n",
                "    df = pd.read_csv('employee_data.csv')\n",
                "elif os.path.exists('meetra_core/employee_data.csv'):\n",
                "    df = pd.read_csv('meetra_core/employee_data.csv')\n",
                "else:\n",
                "    try:\n",
                "        url = 'https://raw.githubusercontent.com/GLITXHY9999/meetra_core/main/employee_data.csv'\n",
                "        df = pd.read_csv(url)\n",
                "    except Exception:\n",
                "        print('[!] Private repository: Please upload employee_data.csv using the files panel, or upload now:')\n",
                "        from google.colab import files\n",
                "        uploaded = files.upload()\n",
                "        filename = list(uploaded.keys())[0]\n",
                "        df = pd.read_csv(filename)\n",
                "\n",
                "print(df.head(10).to_string())\n"
            ]
            break

    with open(eda_path, "w", encoding="utf-8") as f:
        json.dump(eda_nb, f, indent=2)
    print("Updated Experiment_1_2_EDA.ipynb successfully!")
