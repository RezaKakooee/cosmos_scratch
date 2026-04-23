#!/bin/bash
# Show idle GPU slots across all partitions.

echo "=========================================="
echo "Idle GPU slots  —  $(date)"
echo "=========================================="

sinfo -o "%-12P %-20N %-14G %-14G %-10T" \
      --Format="partition,nodelist,gres,gresused,statelong" \
      --noheader \
| awk '
{
    partition = $1
    node      = $2
    gres      = $3   # e.g. gpu:rtxA4500:4
    gresused  = $4   # e.g. gpu:rtxA4500:2(IDX:0-1)
    state     = $5

    # skip non-GPU or down nodes
    if (gres == "(null)" || gres == "N/A") next
    if (state ~ /down|drain|reboot/) next

    # parse total GPUs from gres field  (gpu:type:N  or  gpu:N)
    total = 0
    n = split(gres, parts, ":")
    if (n == 3) total = parts[3] + 0
    if (n == 2) total = parts[2] + 0

    # parse used GPUs from gresused field
    used = 0
    m = split(gresused, uparts, ":")
    if (m >= 3) used = uparts[3] + 0
    if (m == 2) used = uparts[2] + 0

    idle = total - used
    if (idle > 0)
        printf "  %-14s %-20s %-20s  total=%-3d used=%-3d idle=%d\n",
               partition, node, gres, total, used, idle
}
' \
| sort

echo "=========================================="
echo "Summary by partition:"
sinfo --Format="partition,gres,gresused,statelong" --noheader \
| awk '
{
    partition = $1; gres = $2; gresused = $3; state = $4
    if (gres == "(null)" || state ~ /down|drain|reboot/) next

    total = 0
    n = split(gres, gp, ":"); if (n==3) total=gp[3]+0; if (n==2) total=gp[2]+0

    used = 0
    m = split(gresused, gu, ":"); if (m>=3) used=gu[3]+0; if (m==2) used=gu[2]+0

    tot[partition] += total
    use[partition] += used
}
END {
    for (p in tot)
        printf "  %-14s  total=%-4d used=%-4d idle=%d\n", p, tot[p], use[p], tot[p]-use[p]
}
' \
| sort
echo "=========================================="
