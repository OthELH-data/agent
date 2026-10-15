#!/usr/bin/env bash
# Shared connection settings for the deploy scripts. Override any of these in your shell.
VM_HOST="${VM_HOST:-azureuser@20.84.91.17}"
# SHELL_JV_AZURE_VM_KEY is the preferred project-specific override. VM_KEY remains
# supported for existing shells and automation. The project-specific value wins when
# both are supplied.
VM_KEY="${SHELL_JV_AZURE_VM_KEY:-${VM_KEY:-$HOME/Downloads/Shell JV Audit/JOA-VM_key.pem}}"
VM_BASE="${VM_BASE:-/home/azureuser/shell-jv-audit}"
PY_VERSION="${PY_VERSION:-3.11}"

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_PARENT="$(dirname "$LOCAL_REPO")"

SSH_OPTS=(-i "$VM_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
# rsync's -e option takes one shell command string, unlike ssh_vm below, so
# quote the key path explicitly. This matters on macOS where the key lives
# under "Downloads/Shell JV Audit".
SSH_CMD="ssh -i \"$VM_KEY\" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

ssh_vm() { ssh "${SSH_OPTS[@]}" "$VM_HOST" "$@"; }
