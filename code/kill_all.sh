#!/usr/bin/env bash
# Kill all training processes and free DGL/torch ports on every cluster node.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IP_CONFIG="$DIR/ip_config.txt"

KILL='pkill -9 -f "python3.*(train_|launch)" 2>/dev/null || true; \
      pkill -9 -f "torch.distributed" 2>/dev/null || true; \
      pkill -9 -f "dgl.distributed" 2>/dev/null || true; \
      for p in 30050 30051 30052 30053 29500 29501; do fuser -k ${p}/tcp 2>/dev/null || true; done'

eval "$KILL"
if [[ -f "$IP_CONFIG" ]]; then
    while IFS= read -r ip || [[ -n "$ip" ]]; do
        ip=$(echo "$ip" | tr -d '[:space:]')
        [[ -z "$ip" ]] && continue
        echo "Cleaning $ip..."
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$ip" "$KILL" 2>/dev/null || true
    done < "$IP_CONFIG"
fi
echo "Cleanup complete."
