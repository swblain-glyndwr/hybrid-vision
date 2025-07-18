#!/usr/bin/env bash
# Example: 5 G (~100 Mbit, 20 ms RTT)
tc qdisc replace dev lo root netem rate 100mbit delay 20ms