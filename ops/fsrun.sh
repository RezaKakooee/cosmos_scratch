#!/usr/bin/env bash
set -euo pipefail

# Optional partition filter:
#   fsrun.sh h200
#   PREFERRED_PARTITION=h200 fsrun.sh
preferred_partition="${1:-${PREFERRED_PARTITION:-}}"

candidate="$({
     scontrol show node | awk -v want="$preferred_partition" '
          function reset_node() {
               node = ""
               part = ""
               state = ""
               cfg = 0
               alloc = 0
          }

          function has_partition(parts, p,     i, n, a) {
               if (p == "") return 1
               n = split(parts, a, ",")
               for (i = 1; i <= n; i++) {
                    if (a[i] == p) return 1
               }
               return 0
          }

          function first_partition(parts,     a) {
               split(parts, a, ",")
               return a[1]
          }

          function node_unhealthy(s,     t) {
               t = tolower(s)
               return (t ~ /down|drain|not_responding|reboot|fail|maint|planned/)
          }

          BEGIN {
               best_idle = -1
               best_node = ""
               best_part = ""
               reset_node()
          }

          /^NodeName=/ {
               split($1, a, "=")
               node = a[2]
          }

          /State=/ {
               for (i = 1; i <= NF; i++) {
                    if ($i ~ /^State=/) {
                         split($i, a, "=")
                         state = a[2]
                         break
                    }
               }
          }

          /Partitions=/ {
               for (i = 1; i <= NF; i++) {
                    if ($i ~ /^Partitions=/) {
                         split($i, a, "=")
                         part = a[2]
                         break
                    }
               }
          }

          /CfgTRES=/ {
               if (match($0, /gres\/gpu[^, ]*=([0-9]+)/, m)) {
                    cfg = m[1] + 0
               } else {
                    cfg = 0
               }
          }

          /AllocTRES=/ {
               if (match($0, /gres\/gpu[^, ]*=([0-9]+)/, m)) {
                    alloc = m[1] + 0
               } else {
                    alloc = 0
               }

               idle = cfg - alloc
               if (cfg <= 0 || idle <= 0) next
               if (node_unhealthy(state)) next
               if (!has_partition(part, want)) next

               if (idle > best_idle) {
                    best_idle = idle
                    best_node = node
                    best_part = (want != "" ? want : first_partition(part))
               }
          }

          END {
               if (best_node == "") exit 1
               printf "%s|%s|%d\n", best_node, best_part, best_idle
          }
     '
} 2>/dev/null)" || {
     if [[ -n "$preferred_partition" ]]; then
          echo "No idle GPU found in partition '$preferred_partition'."
     else
          echo "No idle GPU found in any healthy partition."
     fi
     exit 1
}

IFS='|' read -r node part idle <<< "$candidate"
echo "Selected node: $node (partition: $part, idle_gpus: $idle)"

srun --job-name=cosmos \
           --time=0-02:00:00 \
           --partition="$part" \
           --nodelist="$node" \
           --gres=gpu:1 \
           --export=ALL \
           --pty bash --init-file /home2/reza/sim/.sim_init.sh