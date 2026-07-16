# 🚀 Local to RunPod Ollama Automation Guide

This guide explains the architectural rules, billing behaviors, and automation steps for offloading heavy LLM inference workloads (like `DeepSeek-R1-Distill-Llama-70B`) from your local Ubuntu PC (RTX 4070 Ti Super) to an on-demand cloud GPU via RunPod.

---

## 🧠 What You Must Know About Local-to-Cloud Execution

Before running the automation scripts, keep these four operational principles in mind to avoid unexpected costs or performance drops:

### 1. Where the Model Files are Saved
*   **The Reverse SSH Tunnel Script (Script 1)**: The cloud GPU functions as a remote compute engine, but it routes through your local server instance. When you execute `ollama run`, the ~43 GB model files are downloaded directly into the **RunPod instance's ephemeral network storage volume**. This saves your local home internet bandwidth and takes under 60 seconds using the cloud data center’s multi-gigabit speeds.
*   **Persistent Storage vs. Billing**: When you **Terminate** a pod on RunPod to stop the hourly billing clock, the local model storage volume is wiped. Every time you spin up a brand-new pod, Ollama will re-download the model file. Because data-center downloads are incredibly fast, this is highly recommended over paying for expensive "persistent network volume storage" while idle.

### 2. The Context Window Memory Tax
As your conversation with DeepSeek-R1 grows longer, Ollama stores the conversation history in a temporary VRAM buffer called the **KV Cache**. 
*   An 8B model uses very little KV Cache. 
*   A 70B model requires significant VRAM just to "remember" past messages. 
*   By renting a **48GB VRAM card (like the RTX A6000)**, you have roughly 5GB of extra VRAM safety buffer beyond the 43GB model footprint. This allows for long, multi-turn reasoning conversations without crashing the remote GPU due to Out-Of-Memory (OOM) faults.

### 3. Mixing and Sharding is Disabled Here
Because you are running the cloud instance over a network connection rather than a physical motherboard slot, **Ollama cannot split layers between your local RTX 4070 Ti Super and the cloud GPU**. The automation scripts handle this cleanly by shifting the execution target *entirely* up to the cloud node. Your local 4070 Ti Super will drop to a 0% idle workload while the cloud node does 100% of the processing.

### 4. Guard Your Wallet: The Billing Meter
RunPod charges by the exact second, but **it will continue billing you indefinitely if you just close your local terminal window.** Closing your terminal kills the text stream interface, but the remote cloud worker remains running in the data center. You must manually log into your browser dashboard and click **Terminate** on the pod to freeze charges.

---

## 🛠️ Script Setup & Deployment Instructions

Choose the file location on your Ubuntu machine where you want to keep your scripts, then use the instructions below.

### 1. The Reverse SSH Tunnel Automator (`runpod_tunnel.sh`)
*Use this option if you deployed a blank **RunPod CUDA Base** image and want your system to cleanly bind ports headlessly.*

Create the file:
```bash
nano runpod_tunnel.sh
```
Paste the following code:
```bash
#!/bin/bash

clear
echo "=========================================================="
echo "🚀 OLLAMA RUNPOD REVERSE TUNNEL AUTOMATOR"
echo "=========================================================="

read -p "🔹 Enter RunPod IP Address: " RUNPOD_IP
read -p "🔹 Enter RunPod Port: " RUNPOD_PORT

if [ -z "\(RUNPOD_IP" ] \vert{}\vert{} [ -z "\)RUNPOD_PORT" ]; then
    echo "❌ Error: IP and Port cannot be empty."
    exit 1
fi

echo -e "\n🔄 Establishing secure reverse tunnel to RunPod..."

# Start SSH Tunnel in background and capture process ID
ssh -f -N -R 11434:localhost:11434 -p "\(RUNPOD_PORT" root@"\)RUNPOD_IP"
TUNNEL_PID=\$!

sleep 3

echo "✅ Tunnel successfully established in background."
echo "📥 Initializing DeepSeek-R1:70B via cloud GPU..."
echo "----------------------------------------------------------"

export OLLAMA_HOST=127.0.0.1:11434
ollama run deepseek-r1:70b

echo "----------------------------------------------------------"
echo "🛑 Exited Ollama session."
echo "🔄 Teardown: Killing background SSH tunnel..."

pkill -f "ssh -f -N -R 11434:localhost:11434"

echo "✅ Tunnel disconnected safely."
echo "🖥️  System reverted back to local hardware tracking."
echo "⚠️  Remember to TERMINATE your pod in the web dashboard!"
```

### 2. The Cloud API Endpoint Switcher (`runpod_api.sh`)
*Use this option if you used RunPod's pre-made **Ollama Template** and want to point your terminal client directly to the generated proxy address web link.*

Create the file:
```bash
nano runpod_api.sh
```
Paste the following code:
```bash
#!/bin/bash

clear
echo "=========================================================="
echo "🌐 OLLAMA REMOTE CLOUD API ROUTER"
echo "=========================================================="

read -p "🔹 Paste your RunPod Public Proxy URL (Port 11434): " PROXY_URL

if [ -z "\$PROXY_URL" ]; then
    echo "❌ Error: Proxy URL cannot be empty."
    exit 1
fi

PROXY_URL="\${PROXY_URL%/}"

echo -e "\n🔗 Routing terminal client to cloud proxy..."
echo "----------------------------------------------------------"

export OLLAMA_HOST="\$PROXY_URL"
ollama run deepseek-r1:70b

echo "----------------------------------------------------------"
echo "🖥️  Session Closed. Global environment variables cleared."
echo "⚠️  Remember to TERMINATE your RunPod instance via your dashboard!"
```

---

## ⚡ Execution Activation

Before executing either file, you must explicitly grant execution rights via the Ubuntu system shell:

```bash
# 1. Move to the directory containing your script files
cd /path/to/your/scripts

# 2. Grant permissions
chmod +x runpod_tunnel.sh
chmod +x runpod_api.sh

# 3. Trigger execution (example for Script 1)
./runpod_tunnel.sh
```

---

## 🔧 Troubleshooting Connectivity Hangups

If your local terminal crashes or your internet drops while the script is active, your local port `11434` can become locked or "stuck" in a ghost listening loop. 

If your script throws a **"Port already in use"** error when you restart it, run this cleanup snippet to forcefully evict hanging SSH processes from your network card memory:

```bash
sudo kill -9 \$(lsof -t -i:11434) 2>/dev/null
pkill -f "ssh -f -N -R 11434"
```
