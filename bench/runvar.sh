#!/bin/bash
# Usage: ./runvar.sh <variant-substr> <ne00> <ne01> [reps] [iters]
cd "$(dirname "$0")"
V="$1"; NE00="$2"; NE01="$3"; REPS="${4:-5}"; ITERS="${5:-20}"
declare -a all
for i in $(seq 1 "$REPS"); do
  line=$(./bench "$NE00" "$NE01" "$ITERS" "$V" 2>/dev/null | grep -E "BW=" | tail -1)
  bw=$(echo "$line" | sed -E 's/.*BW= *([0-9.]+) GB\/s.*/\1/')
  ms=$(echo "$line" | sed -E 's/.* +([0-9.]+) ms\/iter.*/\1/')
  if [ -z "$bw" ]; then echo "no measurement for $V (line=[$line])"; exit 1; fi
  all+=("$bw")
  printf "  rep %d: %s GB/s (%s ms)\n" "$i" "$bw" "$ms"
done
sorted=$(printf '%s\n' "${all[@]}" | sort -n)
min=$(echo "$sorted" | head -1)
cnt=$(echo "$sorted" | wc -l | tr -d ' ')
mid=$(( (cnt+1)/2 ))
med=$(echo "$sorted" | sed -n "${mid}p")
echo "=== $V @ ${NE00}x${NE01}: MIN ${min} GB/s  MEDIAN ${med} GB/s ==="
