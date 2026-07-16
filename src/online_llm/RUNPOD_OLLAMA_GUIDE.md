# MultiCharts AI Parameter Optimizer Deployment Guide

This document maps out the architecture for securely analyzing private MultiCharts PowerLanguage files and massive optimization reports (2000+ rows) using open-weight models on specialized cloud hardware.

---

## 1. Hardware Architecture (RunPod Setup)

To execute local code environments and load uncompromised reasoning models natively, bypass public API services to avoid data leakage of your proprietary trading logic.

### Hardware Selection Blueprint
*   **Recommended Hardware**: `8x NVIDIA H100 (80GB VRAM)` or `8x NVIDIA A100 (80GB VRAM)`.
*   **Total System VRAM**: Minimum 640GB of VRAM is required to run uncompressed, full-scale **DeepSeek-R1 (671B parameters)** or **Llama-3-405B** with an active 128k context configuration.
*   **Alternative Hardware**: If scaling down to `Llama-3.3-70B-Instruct` or `Qwen2.5-Coder-32B`, deploy a single or dual `NVIDIA H100 (80GB/160GB VRAM)` cluster.

### Container Template Selection
Select a **Community Template** or **Base Template** equipped with:
1.  **vLLM Inference Server Engine**: Implements PagedAttention mechanics to avoid Out-Of-Memory (OOM) errors during heavy context extraction loops.
2.  **Jupyter Notebook Interface**: Provides an active local kernel to save and run Python data science pipelines directly beside your data files.

---

## 2. Advanced Environment Tuning Variables

Input these parameters exactly into the RunPod Environment Variables configuration panel prior to booting the instance to maximize system resources:

```env
MAX_MODEL_LEN=131072
GPU_MEMORY_UTILIZATION=0.95
TENSOR_PARALLEL_SIZE=8
```
*(Note: Shift `TENSOR_PARALLEL_SIZE` to match your physical GPU count. If utilizing 2 GPUs, set to 2).*

---

## 3. Workflow Execution Loop

To execute **Option B (Code Execution Environment)** successfully, follow this interaction sequence on your running workspace:

[Your Local Machine] ---> Uploads MultiCharts_Results.csv & PowerLanguage.txt ---> [RunPod Jupyter Server]|[RunPod Jupyter Kernel] <--- Executes Scikit-Learn Script Locally <--------------------------+|v Generates: Feature Importances & Parameter Boundaries|[DeepSeek-R1 / LLM Engine] <--- Fed with Code Rules, Metadata Headers & Summary Logs (No Data Bloat)|v Outputs: Structural Edge Assessments & Curve-Fitting Verification Report