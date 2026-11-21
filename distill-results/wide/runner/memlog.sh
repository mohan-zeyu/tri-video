#!/bin/sh
# Passive host-memory log for the widened-head study: every 10 s MemAvailable/SwapFree (GB); top RSS when low.
while :; do
  a=$(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo); s=$(awk "/SwapFree/{printf \"%.2f\", \$2/1048576}" /proc/meminfo)
  step=$(grep -o "\"step\": [0-9]*" train-c0*.log 2>/dev/null | tail -1 | grep -o "[0-9]*$")
  echo "$(date +%s) avail_GB=$a swapfree_GB=$s wide_step=$step" >> memlog.txt
  if [ "${a%.*}" -lt 60 ]; then ps -eo rss,user,pid,comm --sort=-rss | head -6 | tr "\n" ";" >> memlog.txt; echo >> memlog.txt; fi
  sleep 10
done
