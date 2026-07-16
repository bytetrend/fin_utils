#!/bin/bash

# Clear the screen for a clean UI
clear
echo "=========================================================="
echo "🚀 OLLAMA RUNPOD REVERSE TUNNEL AUTOMATOR"
echo "=========================================================="

# 1. Get connection details from user
read -p "🔹 Enter RunPod IP Address: " RUNPOD_IP
read -p "🔹 Enter RunPod Port: " RUNPOD_PORT

if [ -z "$RUNPOD_IP" ] || [ -z "$RUNPOD_PORT" ]; then
    echo "❌ Error: IP and Port cannot be empty."
    exit 1
fi

echo -e "\n🔄 Establishing secure reverse tunnel to RunPod..."

# 2. Start the SSH Tunnel in the background and grab its Process ID (PID)
# The -f flag tells SSH to go into the background right after authentication
ssh -f -N -R 11434:localhost:11434 -p "$RUNPOD_PORT" root@"$RUNPOD_IP"
TUNNEL_PID=$!

# Give the tunnel a couple of seconds to negotiate the handshake safely
sleep 3

echo "✅ Tunnel successfully established in background (PID: $TUNNEL_PID)."
echo "📥 Initializing DeepSeek-R1:70B via the cloud GPU..."
echo "----------------------------------------------------------"

# 3. Export the host variable and pass execution straight to Ollama
export OLLAMA_HOST=127.0.0.1:11434
ollama run deepseek-r1:70b

# ==========================================================
# Everything below this line executes AFTER you exit Ollama (/bye)
# ==========================================================
echo "----------------------------------------------------------"
echo "🛑 Exited Ollama session."
echo "🔄 Teardown: Killing background SSH tunnel..."

# 4. Clean up background processes
kill $TUNNEL_PID 2>/dev/null
pkill -f "ssh -f -N -R 11434:localhost:11434"

echo "✅ Tunnel disconnected safely."
echo "🖥️  Your system is now reverted back to your local RTX 4070 Ti Super."
echo "⚠️  Remember to go to your RunPod dashboard and TERMINATE your pod to stop billing!"
