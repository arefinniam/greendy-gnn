#!/usr/bin/env bash
# Stop artifact-owned DGL training processes and free DGL/Torch ports.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IP_CONFIG="$DIR/ip_config.txt"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5)

KILL='pkill -TERM -f "python3.*(train_|launch)" 2>/dev/null || true; \
      pkill -TERM -f "torch.distributed" 2>/dev/null || true; \
      pkill -TERM -f "dgl.distributed" 2>/dev/null || true; \
      sleep 2; \
      pkill -KILL -f "python3.*(train_|launch)" 2>/dev/null || true; \
      pkill -KILL -f "torch.distributed" 2>/dev/null || true; \
      pkill -KILL -f "dgl.distributed" 2>/dev/null || true; \
      for p in 30050 30051 30052 30053 29500 29501; do fuser -k -TERM ${p}/tcp 2>/dev/null || true; done'

pkill -TERM -f "python3.*(train_|launch)" 2>/dev/null || true
pkill -TERM -f "torch.distributed" 2>/dev/null || true
pkill -TERM -f "dgl.distributed" 2>/dev/null || true
sleep 2
pkill -KILL -f "python3.*(train_|launch)" 2>/dev/null || true
pkill -KILL -f "torch.distributed" 2>/dev/null || true
pkill -KILL -f "dgl.distributed" 2>/dev/null || true
for p in 30050 30051 30052 30053 29500 29501; do
    fuser -k -TERM ${p}/tcp 2>/dev/null || true
done

if [[ -f "$IP_CONFIG" ]]; then
    while IFS= read -r ip || [[ -n "$ip" ]]; do
        ip=$(echo "$ip" | tr -d '[:space:]')
        [[ -z "$ip" ]] && continue
        echo "Cleaning $ip..."
        ssh "${SSH_OPTS[@]}" "$ip" "$KILL" 2>/dev/null || true
    done < "$IP_CONFIG"
fi
echo "Cleanup complete."
